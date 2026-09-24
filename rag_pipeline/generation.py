from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


DEFAULT_MODEL_PATH = r"E:\DS_Team_Share\hugging-face-model\Qwen2.5-14B-Instruct\models--Qwen--Qwen2.5-14B-Instruct\snapshots\cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_N_CONTEXT = 5

RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using ONLY the provided context.\n"
    "If the context does not contain enough information, say: "
    "'I don't have enough information to answer this.'\n"
    "Do not make up facts."
)

DIRECT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using your general knowledge.\n"
    "If you are unsure, say: 'I don't know.'"
)


class Generation:

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        quantize: bool = True,
    ) -> None:
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.quantize = quantize
        self._pipeline: Optional[Any] = None

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
                pipeline,
            )

            resolved = (
                self.model_path
                if os.path.exists(self.model_path)
                else "Qwen/Qwen2.5-14B-Instruct"
            )
            print(f"[Generation] Loading model from '{resolved}'...")

            tokenizer = AutoTokenizer.from_pretrained(
                resolved,
                trust_remote_code=True,
                clean_up_tokenization_spaces=False,
            )

            if self.quantize:
                quant_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    resolved,
                    quantization_config=quant_cfg,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    resolved,
                    device_map="auto",
                    trust_remote_code=True,
                )

            model.generation_config.max_new_tokens = self.max_new_tokens
            model.generation_config.max_length = None
            model.generation_config.do_sample = self.temperature > 0
            if self.temperature > 0:
                model.generation_config.temperature = self.temperature

            self._pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                trust_remote_code=True,
                return_full_text=False,
                clean_up_tokenization_spaces=False,
            )
            print("[Generation] Model loaded.")
        return self._pipeline

    def _build_context(self, docs: List[Dict]) -> str:
        return "\n\n".join(
            doc.get("page_content", "") for doc in docs if doc.get("page_content", "").strip()
        )

    def _call(self, system_prompt: str, user_message: str) -> str:
        pipe = self._get_pipeline()
        tokenizer = pipe.tokenizer
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ]
        formatted = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        output = pipe(formatted, return_full_text=False)
        return output[0]["generated_text"].strip()

    def generate(
        self,
        query: str,
        docs: List[Dict],
        system_prompt: Optional[str] = None,
    ) -> Dict[str, str]:
        context = self._build_context(docs)

        if context:
            sys_prompt = system_prompt or RAG_SYSTEM_PROMPT
            user_message = f"Context:\n{context}\n\nQuestion:\n{query}"
        else:
            sys_prompt = DIRECT_SYSTEM_PROMPT
            user_message = query

        print(f"[Generation] Generating answer (context_docs={len(docs)}, rag={'yes' if context else 'no'})...")
        answer = self._call(sys_prompt, user_message)
        print(f"[Generation] Done.")

        return {
            "answer": answer,
            "context": context,
            "query": query,
            "rag_used": bool(context),
        }

    def generate_direct(self, query: str) -> str:
        print("[Generation] Direct generation (no context)...")
        return self._call(DIRECT_SYSTEM_PROMPT, query)

    def __repr__(self) -> str:
        return (
            f"Generation("
            f"model='{os.path.basename(self.model_path)}', "
            f"max_new_tokens={self.max_new_tokens}, "
            f"temperature={self.temperature})"
        )
