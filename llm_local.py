
import asyncio
import os
import json
import re
import warnings
import torch
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, Dict, Iterator, List, Optional, Sequence, Type, Union

from transformers import BitsAndBytesConfig, AutoModelForCausalLM
warnings.filterwarnings(
    "ignore",
    message=".*Passing.*generation_config.*together with generation-related arguments.*",
    category=UserWarning,
)

from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    
)
# pyrefly: ignore [missing-import]
from langchain_core.tools import BaseTool
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel



def _extract_json(text: str) -> Optional[dict]:
    
    text = text.strip()

    # 0. Strip <think>...</think> blocks emitted by reasoning/thinking SLMs
    #    (e.g. DeepSeek-R1, Qwen3-thinking) before looking for JSON.
    #    Case A: properly closed block  →  <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    #    Case B: unclosed block (model hit max_new_tokens mid-thought)
    #            Everything from <think> to end-of-string is the thinking
    #            chain; there is no JSON after it, so strip and let the
    #            caller raise a clear error rather than try to parse garbage.
    if "<think>" in text:
        text = text[: text.index("<think>")].strip()

    # 1. Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # 2. Try stripping markdown code fences
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        try:
            result = json.loads(fenced.group(1))
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # 3. Try extracting the first balanced {...} block
    start = text.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        result = json.loads(candidate)
                        if isinstance(result, dict):
                            return result
                    except json.JSONDecodeError:
                        break

    # 4. Partial / truncated JSON recovery
    #    When the model hits max_new_tokens mid-JSON (e.g. the closing }" is missing)
    #    attempt to extract any fully-written string fields via regex so that
    #    at least `issup` (or similar required fields) can still be recovered.
    start = text.find("{")
    if start != -1:
        fragment = text[start:]
        partial: dict = {}
        # Extract completed string values:  "key": "value"
        for m in re.finditer(r'"([^"]+)"\s*:\s*"([^"]*)"', fragment):
            partial[m.group(1)] = m.group(2)
        # Extract completed array of strings:  "key": ["a", "b"]
        for m in re.finditer(r'"([^"]+)"\s*:\s*\[([^\]]*)\]', fragment):
            items = re.findall(r'"([^"]*)"', m.group(2))
            partial[m.group(1)] = items
        if partial:
            return partial

    return None



class QwenChatHuggingFace(ChatHuggingFace):
   
    bound_tools: List[Any] = []
    
    def bind_tools(
        self,
        tools: Sequence[Union[BaseTool, Any]],
        **kwargs,
    ) -> "QwenChatHuggingFace":
        clone = QwenChatHuggingFace(llm=self.llm)
        clone.bound_tools = list(tools)
        return clone

    def with_structured_output(
        self,
        schema: Any,
        *,
        method: str = "json_mode",
        include_raw: bool = False,
        **kwargs,
    ) -> Runnable:
       
        llm_ref = self  # capture self for the closure

        # Build a JSON schema string for the prompt
        if hasattr(schema, "model_json_schema"):
            schema_str = json.dumps(schema.model_json_schema(), indent=2)
        elif hasattr(schema, "schema"):
            schema_str = json.dumps(schema.schema(), indent=2)
        else:
            schema_str = str(schema)

        def _invoke_structured(input_val: Any) -> Any:
            # Normalise input to a list of message dicts
            if isinstance(input_val, str):
                messages_raw = [{"role": "user", "content": input_val}]
            elif isinstance(input_val, list):
                messages_raw = []
                for m in input_val:
                    if isinstance(m, tuple) and len(m) == 2:
                        role_map = {"system": "system", "human": "user", "user": "user", "ai": "assistant", "assistant": "assistant"}
                        messages_raw.append({"role": role_map.get(m[0], m[0]), "content": m[1]})
                    elif hasattr(m, "content"):
                        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
                        role = "system" if isinstance(m, SystemMessage) else ("assistant" if isinstance(m, AIMessage) else "user")
                        messages_raw.append({"role": role, "content": m.content})
                    elif isinstance(m, dict):
                        messages_raw.append(m)
                    else:
                        messages_raw.append({"role": "user", "content": str(m)})
            elif isinstance(input_val, dict):
                messages_raw = [input_val]
            else:
                messages_raw = [{"role": "user", "content": str(input_val)}]

            # Inject a strict JSON-only system prompt
            json_system = (
                "You MUST respond with ONLY a valid JSON object matching this schema — "
                "no extra text, no markdown, no explanation:\n"
                f"{schema_str}"
            )
            # Prepend or replace existing system message
            if messages_raw and messages_raw[0].get("role") == "system":
                messages_raw[0]["content"] = json_system + "\n\n" + messages_raw[0]["content"]
            else:
                messages_raw.insert(0, {"role": "system", "content": json_system})

            # Call the already-built HF pipeline from the LLM wrapper.
            tokenizer = llm_ref.llm.pipeline.tokenizer
            hf_pipeline = llm_ref.llm.pipeline

            formatted_prompt: str = tokenizer.apply_chat_template(
                messages_raw,
                add_generation_prompt=True,
                tokenize=False,
            )
            raw_outputs = hf_pipeline(formatted_prompt, return_full_text=False)
            # pipeline returns generated tokens under "generated_text"
            text: str = raw_outputs[0]["generated_text"].strip()

            # Extract JSON from the response using multiple heuristics
            parsed = _extract_json(text)

            # Validate against Pydantic schema if applicable
            if parsed is not None:
                if hasattr(schema, "model_validate"):
                    try:
                        return schema.model_validate(parsed)
                    except Exception as exc:
                        raise ValueError(
                            f"LLM returned valid JSON but it failed Pydantic validation.\n"
                            f"Parsed JSON: {parsed}\n"
                            f"Validation error: {exc}\n"
                            f"Tip: check that the values in your prompt exactly match the schema enum values."
                        ) from exc
                elif hasattr(schema, "parse_obj"):
                    try:
                        return schema.parse_obj(parsed)
                    except Exception as exc:
                        raise ValueError(
                            f"LLM returned valid JSON but it failed Pydantic validation.\n"
                            f"Parsed JSON: {parsed}\n"
                            f"Validation error: {exc}"
                        ) from exc
                return parsed  # schema has no validation method — return raw dict

            raise ValueError(
                f"Could not extract valid JSON from LLM output:\n{text!r}\n"
                f"Expected schema:\n{schema_str}"
            )

        return RunnableLambda(_invoke_structured)

  
    def _tool_to_openai_schema(self, tool: Any) -> dict:
        """Convert a LangChain @tool / BaseTool to OpenAI function schema."""
        name = getattr(tool, "name", str(tool))
        desc = getattr(tool, "description", "No description.")

        props: dict = {}
        required: List[str] = []
        if hasattr(tool, "args_schema") and tool.args_schema is not None:
            try:
                raw = tool.args_schema.schema()
                props = raw.get("properties", {})
                required = raw.get("required", [])
            except Exception:
                pass
        elif hasattr(tool, "args") and tool.args:
            for k in tool.args:
                props[k] = {"type": "string", "description": k}
            required = list(tool.args.keys())

        return {
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
            },
        }


    def _lc_messages_to_dicts(self, messages: List[BaseMessage]) -> List[dict]:
        """
        Convert LangChain message objects to the plain-dict format that
        tokenizer.apply_chat_template() expects.
        """
        result: List[dict] = []
        for msg in messages:
            if isinstance(msg, dict):
                result.append(msg)
                continue
            if isinstance(msg, str):
                result.append({"role": "user", "content": msg})
                continue

            if isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": getattr(msg, "content", "")})

            elif isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": getattr(msg, "content", "")})

            elif isinstance(msg, AIMessage):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    # Re-encode tool calls in Qwen's <tool_call> format
                    tc_text = "\n".join(
                        f"<tool_call>\n{json.dumps({'name': tc['name'], 'arguments': tc.get('args', {})})}\n</tool_call>"
                        for tc in msg.tool_calls
                    )
                    result.append({"role": "assistant", "content": tc_text})
                else:
                    result.append({"role": "assistant", "content": getattr(msg, "content", "")})

            elif isinstance(msg, ToolMessage):
                # Qwen expects tool results as role="tool"
                result.append({
                    "role": "tool",
                    "content": getattr(msg, "content", ""),
                    "name": getattr(msg, "name", "tool"),
                })

            elif hasattr(msg, "type") and msg.type == "tool":
                result.append({
                    "role": "tool",
                    "content": getattr(msg, "content", ""),
                    "name": getattr(msg, "name", "tool"),
                })

            elif hasattr(msg, "content"):
                result.append({"role": "user", "content": str(msg.content)})
            else:
                result.append({"role": "user", "content": str(msg)})

        return result

  
    def _extract_tool_calls(self, text: str) -> List[dict]:
        """
        Parse <tool_call>...</tool_call> blocks from the generated text.
        Falls back to a plain-JSON heuristic for robustness.
        """

        tool_calls: List[dict] = []

        # --- 1. Native Qwen format: <tool_call>...</tool_call> ---
        for raw in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL):
            try:
                data = json.loads(raw)
                name = data.get("name")
                args = data.get("arguments", {})
                if name:
                    tool_calls.append({
                        "name": name,
                        "args": args,
                        "id": f"call_{name}_{len(tool_calls)}",
                        "type": "tool_call",
                    })
            except json.JSONDecodeError:
                pass

        if tool_calls:
            return tool_calls

        # --- 2. Fallback: bare JSON object with "tool" / "name" key ---
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        candidate = fenced.group(1) if fenced else text
        for pattern in [
            r'\{\s*"name"\s*:\s*"([^"]+)".*?"arguments"\s*:\s*(\{.*?\})\s*\}',
            r'\{\s*"tool"\s*:\s*"([^"]+)".*?"args"\s*:\s*(\{.*?\})\s*\}',
        ]:
            m = re.search(pattern, candidate, re.DOTALL)
            if m:
                try:
                    obj = json.loads(m.group())
                    name = obj.get("name") or obj.get("tool")
                    args = obj.get("arguments") or obj.get("args", {})
                    if name:
                        tool_calls.append({
                            "name": name,
                            "args": args,
                            "id": f"call_{name}_0",
                            "type": "tool_call",
                        })
                        return tool_calls
                except json.JSONDecodeError:
                    pass

        return tool_calls


    def invoke(
        self, input: Any, config: Optional[dict] = None, **kwargs
    ) -> AIMessage:

        if not self.bound_tools:
            # No tools bound — delegate entirely to parent (fast path)
            return super().invoke(input, config=config, **kwargs)

        # --- normalise input ---
        if isinstance(input, str):
            messages: List[BaseMessage] = [HumanMessage(content=input)]
        elif isinstance(input, list):
            messages = list(input)
        else:
            messages = [input]

        # --- convert to dicts ---
        messages_dict = self._lc_messages_to_dicts(messages)

        # --- get tokenizer from the underlying pipeline ---
        tokenizer = self.llm.pipeline.tokenizer

        # --- build tool schemas ---
        tool_schemas = [self._tool_to_openai_schema(t) for t in self.bound_tools]

        # --- apply_chat_template with native tools parameter ---
        # This generates the EXACT format Qwen2.5 was trained on.
        try:
            formatted_prompt: str = tokenizer.apply_chat_template(
                messages_dict,
                tools=tool_schemas,
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception as e:
            # Fallback if tokenizer doesn't support tools param
            print(f"[QwenChatHuggingFace] apply_chat_template(tools=...) failed: {e}")
            print("[QwenChatHuggingFace] Falling back to parent invoke()")
            return super().invoke(input, config=config, **kwargs)

        # --- call the raw HF pipeline directly ---
        raw_outputs = self.llm.pipeline(
            formatted_prompt,
            return_full_text=False,   # only newly generated tokens
        )
        generated_text: str = raw_outputs[0]["generated_text"]

        # --- parse tool calls ---
        tool_calls = self._extract_tool_calls(generated_text)

        if tool_calls:
            return AIMessage(
                content="",
                tool_calls=tool_calls,
            )

        return AIMessage(content=generated_text)

    # ── Async shims ──────────────────────────────────────────────────────────
    # RAGAS evaluation uses ainvoke / agenerate internally.
    # HuggingFacePipeline does not support native async, so we run the
    # synchronous path in a thread-pool executor to avoid blocking the
    # event loop and satisfy the async interface.

    async def ainvoke(
        self, input: Any, config: Optional[dict] = None, **kwargs
    ) -> AIMessage:
        """Async wrapper: runs sync invoke() in a thread-pool executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.invoke(input, config=config, **kwargs)
        )

    async def _agenerate(
        self,
        messages: List[Any],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs,
    ):
        """Async wrapper: runs sync _generate() in a thread-pool executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._generate(messages, stop=stop, run_manager=None, **kwargs),
        )


# Pipeline factory — cached so the model loads only once

@lru_cache(maxsize=1)
def _build_pipeline(model_path: str, temperature: float, max_new_tokens: int, quantize: bool = True):
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    print(f"[llm_local] Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, clean_up_tokenization_spaces=False)
    # Load model with optional 4-bit quantization (BitsAndBytes)
    if quantize:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
        )
    
    # Safely modify the loaded generation_config so we keep its eos_token_id, bos_token_id, etc.
    model.generation_config.max_new_tokens = max_new_tokens
    model.generation_config.max_length = None
    model.generation_config.repetition_penalty = 1.05 
    model.generation_config.do_sample = temperature > 0
    if temperature > 0:
        model.generation_config.temperature = temperature
        
 

    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        trust_remote_code=True,
        return_full_text=False,
        clean_up_tokenization_spaces=False
    )

    return hf_pipeline


def get_local_llm(
    model_path:str = r"E:\DS_Team_Share\hugging-face-model\Qwen2.5-14B-Instruct\models--Qwen--Qwen2.5-14B-Instruct\snapshots\cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
    temperature: float = 0,
    max_new_tokens: int = 2048,
    quantize: bool = True,
 ) -> QwenChatHuggingFace:
    """
    Load the local Qwen2.5-Instruct model and return a QwenChatHuggingFace.

    Usage:
        llm = get_local_llm()
        llm_with_tools = llm.bind_tools([tool1, tool2])
    """
    # model_path = os.environ.get(
    #     "LOCAL_MODEL_PATH",
    #     r"E:\DS_Team_Share\hugging-face-model\Qwen2.5-14B-Instruct\models--Qwen--Qwen2.5-14B-Instruct\snapshots\cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
        
    # ).strip()

    # if not model_path:
    #     raise EnvironmentError(
    #         "LOCAL_MODEL_PATH is not set.\n"
    #         "Add it to your .env file:  LOCAL_MODEL_PATH=/path/to/your/Qwen-model"
    #     )

    hf_pipeline = _build_pipeline(model_path, float(temperature), int(max_new_tokens), quantize=bool(quantize))
    llm_pipeline = HuggingFacePipeline(pipeline=hf_pipeline)
    return QwenChatHuggingFace(llm=llm_pipeline)
