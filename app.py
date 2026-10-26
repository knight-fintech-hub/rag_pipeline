from __future__ import annotations

import asyncio
import os
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"
import shutil
from contextlib import asynccontextmanager
from typing import Annotated, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from rag_pipeline.rag_agent import RAGAgent
from config import cfg
from database import init_db, close_pool


UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploaded_documents")
os.makedirs(UPLOAD_DIR, exist_ok=True)

_agent: Optional[RAGAgent] = None


def get_agent() -> RAGAgent:
    global _agent
    if _agent is None:
        try:
            qdrant_path = os.environ.get(
                "QDRANT_PATH", 
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_pipeline", "qdrant_store")
            )
            embedding_model_path = os.environ.get("EMBEDDING_MODEL_PATH", r"E:\ds-team\huggingface_models\Old BKP\test_models\bge_m3")
            reranker_model_path = os.environ.get("RERANKER_MODEL_PATH", r"E:\ds-team\huggingface_models\Old BKP\test_models\bge_reranker_v2_m3")
            generator_model_path = os.environ.get("GENERATOR_MODEL_PATH", r"E:\DS_Team_Share\hugging-face-model\Qwen2.5-14B-Instruct\models--Qwen--Qwen2.5-14B-Instruct\snapshots\cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8")
            
            _agent = RAGAgent(
                collection_name=os.environ.get("COLLECTION_NAME", "docs"),
                embedding_model_path=embedding_model_path,
                reranker_model_path=reranker_model_path,
                generator_model_path=generator_model_path,
                qdrant_path=qdrant_path,
                device=os.environ.get("DEVICE", "cuda"),
                top_k=int(os.environ.get("TOP_K", "8")),
                top_n=int(os.environ.get("TOP_N", "3")),
                score_threshold=float(os.environ.get("SCORE_THRESHOLD", "0.6")),
                gpu_batch_size=int(os.environ.get("GPU_BATCH_SIZE", "32")),
                max_new_tokens=int(os.environ.get("MAX_NEW_TOKENS", "2048")),
                quantize=os.environ.get("QUANTIZE", "True").lower() in ("true", "1", "yes"),
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Agent initialisation failed: {exc}")
    return _agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    print(f"[API] Server starting up. (USE_DB={cfg.USE_DB}, DEBUG={cfg.DEBUG})")
    if cfg.USE_DB:
        db_ok = init_db()
        if not db_ok:
            print("[API] Warning: DB unavailable — session history will be disabled.")
    else:
        print("[API] USE_DB=false — skipping database connection.")

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    print("[API] Server shutting down.")
    if cfg.USE_DB:
        close_pool()


app = FastAPI(
    title="Distributed RAG API",
    description="Production-grade RAG pipeline: ingest PDFs, query with hybrid retrieval + reranking + generation.",
    version="1.0.0",
    redoc_url="/",
    docs_url="/",
    lifespan=lifespan,
)


def _patch_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    for _, path_item in schema.get("paths", {}).items():
        for _, operation in path_item.items():
            for param in operation.get("parameters", []):
                if param.get("in") == "formData" and param.get("type") == "file":
                    param["format"] = "binary"
            req_body = operation.get("requestBody", {})
            content = req_body.get("content", {})
            for media_type, media_obj in content.items():
                props = media_obj.get("schema", {}).get("properties", {})
                for prop_name, prop_val in props.items():
                    if prop_val.get("type") == "string" and prop_val.get("contentMediaType") == "application/octet-stream":
                        prop_val["format"] = "binary"
                    if prop_val.get("type") == "array":
                        items = prop_val.get("items", {})
                        if items.get("contentMediaType") == "application/octet-stream":
                            items["format"] = "binary"
    app.openapi_schema = schema
    return schema


app.openapi = _patch_openapi


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to answer.")
    doc_ids: Optional[List[str]] = Field(default=None, description="Optional list of document IDs to filter the search by.")
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional session ID (UUID from the DB). When supplied the LLM receives "
            "the prior conversation turns for this session as context, and the new "
            "Q/A pair is saved back so future queries remain context-aware."
        ),
    )


# ── /retrieve request / response ─────────────────────────────────────────────

class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The search query to embed and retrieve against.")
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for audit / logging purposes.",
    )
    doc_ids: Optional[List[str]] = Field(
        default=None,
        description="Filter results to only chunks belonging to these document IDs.",
    )
    collection_name: str = Field(
        ...,
        description="Target Qdrant collection.",
    )
    score_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum rerank score a chunk must have to be included in the response. "
            "Overrides the agent-level threshold for this request."
        ),
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Number of candidate chunks fetched from the vector store before reranking.",
    )
    top_n: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Number of top chunks returned after reranking.",
    )
    fetch_k: int = Field(
        default=20,
        ge=1,
        description="Internal MMR candidate pool size (used inside hybrid search).",
    )
    lambda_mult: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="MMR diversity weight: 1.0 = pure relevance, 0.0 = maximum diversity.",
    )


class RetrievedChunk(BaseModel):
    chunk_id: str = Field(..., description="Qdrant point ID of the chunk.")
    page_content: str = Field(..., description="Raw text of the chunk.")
    metadata: Dict = Field(default_factory=dict, description="Chunk metadata (doc_id, page, source, …).")
    dense_score: Optional[float] = Field(default=None, description="Dense-vector similarity score.")
    rerank_score: Optional[float] = Field(default=None, description="Cross-encoder rerank score (0–1).")
    rrf_score: Optional[float] = Field(default=None, description="RRF fusion score (only set when hybrid search is used).")


class RetrieveResponse(BaseModel):
    query: str
    session_id: Optional[str] = None
    collection_name: str
    chunks: List[RetrievedChunk]
    total_retrieved: int = Field(..., description="Number of chunks returned after reranking / threshold filtering.")
    top_rerank_score: Optional[float] = None
    latencies: Dict[str, float]


# ── /generate request / response ─────────────────────────────────────────────

class ChunkInput(BaseModel):
    """A retrieved chunk passed in as context to the generation endpoint."""
    chunk_id: Optional[str] = Field(default=None, description="Qdrant point ID — informational only.")
    page_content: str = Field(..., description="Raw text of the chunk.")
    metadata: Optional[Dict] = Field(default=None, description="Optional chunk metadata.")
    rerank_score: Optional[float] = Field(default=None, description="Score from the retrieval stage (used to decide RAG vs. direct mode).")


class GenerateRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user question / prompt to answer.")
    chunks: List[ChunkInput] = Field(
        default_factory=list,
        description="Retrieved chunks from the /retrieve endpoint (or assembled externally).",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID used to load chat history and persist this turn.",
    )
    score_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "If the best chunk rerank_score is below this value the LLM falls back to "
            "direct (knowledge-only) generation. Defaults to the agent-level threshold."
        ),
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Custom system prompt. Overrides the built-in RAG/direct prompts when provided.",
    )
    max_new_tokens: Optional[int] = Field(
        default=None,
        ge=64,
        description="Override max_new_tokens for this request only.",
    )


class GenerateResponse(BaseModel):
    answer: str
    mode: str = Field(..., description="'rag' when context was used, 'direct' otherwise.")
    query: str
    context_used: str = Field(..., description="The concatenated context string fed to the LLM.")
    session_id: Optional[str] = None
    chunks_used: int = Field(..., description="Number of chunks actually fed into the prompt.")
    top_rerank_score: Optional[float] = None
    latencies: Dict[str, float]


class DocumentMetadata(BaseModel):
    filename: str
    doc_id: str


class IngestResponse(BaseModel):
    status: str
    documents_ingested: List[DocumentMetadata]
    documents: int
    chunks: int
    points_indexed: int
    collection: str
    latencies: Dict[str, float]


class QueryResponse(BaseModel):
    answer: str
    mode: str
    query: str
    context: str
    top_rerank_score: Optional[float] = None
    reason: Optional[str] = None
    latencies: Dict[str, float]





@app.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest PDF documents",
    description=(
        "Upload one or more PDF files via multipart/form-data. "
        "Files are saved to disk, chunked, embedded, and indexed into Qdrant. "
        "Set force_recreate=true to drop and rebuild the collection from scratch. "
        "Returns a unique doc_id for each uploaded file."
    ),
)
async def ingest(
    files: Annotated[
        List[UploadFile],
        File(description="One or more PDF documents to ingest."),
    ],
    force_recreate: Annotated[
        bool,
        Form(description="Drop  and recreate the Qdrant collection before indexing."),
    ] = False,
):
    import uuid
    agent = get_agent()

    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    saved_paths: List[str] = []
    rejected: List[str] = []
    doc_mapping: dict = {}
    docs_meta: List[dict] = []

    try:
        for upload in files:
            filename = upload.filename or ""

            if not filename:
                continue

            if not filename.lower().endswith(".pdf"):
                rejected.append(filename)
                continue

            safe_name = os.path.basename(filename)
            dest = os.path.join(UPLOAD_DIR, safe_name)

            with open(dest, "wb") as out_file:
                shutil.copyfileobj(upload.file, out_file)

            file_size = os.path.getsize(dest)
            
            doc_id = str(uuid.uuid4())[:12]
            doc_mapping[dest] = doc_id
            docs_meta.append({"filename": safe_name, "doc_id": doc_id})
            
            print(f"[API] Saved '{safe_name}' ({file_size} bytes) → {dest} [doc_id={doc_id}]")
            saved_paths.append(dest)

        if rejected:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type(s): {rejected}. Only PDF files are accepted.",
            )

        if not saved_paths:
            raise HTTPException(status_code=400, detail="No valid PDF files found in the request.")

        result = await asyncio.to_thread(agent.ingest, saved_paths, force_recreate, doc_mapping)
        result["documents_ingested"] = docs_meta
        return JSONResponse(status_code=200, content=result)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")


@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the RAG agent",
    description="Ask a question. Can be filtered by collection_name and doc_ids. Returns latency in seconds.",
)
async def query(request: QueryRequest):
    agent = get_agent()
    try:
        result = await asyncio.to_thread(
            agent.invoke,
            request.question,
            request.doc_ids,
            request.session_id,
        )

        return JSONResponse(
            status_code=200,
            content={
                "answer":           result.get("answer", ""),
                "mode":             result.get("mode", ""),
                "query":            result.get("query", request.question),
                "context":          result.get("context", ""),
                "top_rerank_score": result.get("top_rerank_score"),
                "reason":           result.get("reason"),
                "latencies":        result.get("latencies", {}),
                "session_id":       request.session_id,
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        print(f"[API /query] ERROR: {exc}\n{tb}")
        raise HTTPException(status_code=500, detail=f"Query failed: {type(exc).__name__}: {exc}")


# ── /retrieve endpoint ────────────────────────────────────────────────────────

@app.post(
    "/retrieve",
    response_model=RetrieveResponse,
    summary="Retrieve relevant chunks",
    description=(
        "Embeds the query, runs true hybrid search (dense semantic + BM25 keyword) over Qdrant, "
        "fuses results via RRF, reranks the candidates, optionally filters by a score threshold, and "
        "returns the top-N chunks with their scores. "
        "Use the returned chunks as input to POST /generate."
    ),
    tags=["RAG Pipeline"],
)
async def retrieve(request: RetrieveRequest):
    import time
    from qdrant_client.models import Filter, FieldCondition, MatchAny

    agent = get_agent()
    latencies: Dict[str, float] = {}

    try:
        # ── 1. Embed query (dense + sparse) ──────────────────────────────────
        t0 = time.perf_counter()
        query_vector = await asyncio.to_thread(agent.embedder.embed_query, request.query)
        latencies["query_embedding"] = round(time.perf_counter() - t0, 4)

        # Sparse (BM25) query vector — graceful fallback if unavailable
        query_sparse_vector = None
        try:
            t0 = time.perf_counter()
            query_sparse_vector = await asyncio.to_thread(agent.embedder.embed_query_sparse, request.query)
            latencies["query_sparse_embedding"] = round(time.perf_counter() - t0, 4)
        except Exception as e:
            print(f"[API /retrieve] Sparse query embedding failed ({e}). BM25 will be skipped.")

        # ── 2. Build optional payload filter ──────────────────────────────────
        payload_filter = None
        if request.doc_ids:
            payload_filter = Filter(
                must=[
                    FieldCondition(
                        key="metadata.doc_id",
                        match=MatchAny(any=request.doc_ids),
                    )
                ]
            )

        # ── 3. Resolve collection name ────────────────────────────────────────
        collection_name = request.collection_name or agent.collection_name

        # ── 4. True hybrid retrieval (Dense + BM25 + RRF) ────────────────────
        t0 = time.perf_counter()
        candidates = await asyncio.to_thread(
            agent.retriever.retrieve,
            query_vector,
            query_sparse_vector,    # sparse vector for BM25
            request.top_k,          # top_k override (None → agent default)
            request.fetch_k,
            request.lambda_mult,
            payload_filter,
            collection_name,
        )
        latencies["retrieval"] = round(time.perf_counter() - t0, 4)

        # ── 5. Rerank ─────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        reranked = await asyncio.to_thread(
            agent.reranker.rerank,
            request.query,
            candidates,
            request.top_n,          # top_n override (None → reranker default)
        )
        latencies["reranking"] = round(time.perf_counter() - t0, 4)

        # ── 6. Optional score threshold filter ────────────────────────────────
        threshold = request.score_threshold if request.score_threshold is not None else 0.0
        if threshold > 0.0:
            reranked = [c for c in reranked if c.get("rerank_score", 0.0) >= threshold]

        top_score = reranked[0].get("rerank_score") if reranked else None

        # ── 7. Serialise ──────────────────────────────────────────────────────
        chunks_out = [
            RetrievedChunk(
                chunk_id=c.get("id", ""),
                page_content=c.get("page_content", ""),
                metadata=c.get("metadata", {}),
                dense_score=c.get("score"),
                rerank_score=c.get("rerank_score"),
                rrf_score=c.get("rrf_score"),
            )
            for c in reranked
        ]

        latencies["total"] = round(sum(latencies.values()), 4)

        return RetrieveResponse(
            query=request.query,
            session_id=request.session_id,
            collection_name=collection_name,
            chunks=chunks_out,
            total_retrieved=len(chunks_out),
            top_rerank_score=top_score,
            latencies=latencies,
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}")


# ── /generate endpoint ────────────────────────────────────────────────────────

@app.post(
    "/generate",
    response_model=GenerateResponse,
    summary="Generate an answer from retrieved chunks",
    description=(
        "Takes a query and a list of retrieved chunks (e.g. from POST /retrieve) "
        "and generates a grounded answer. If the top chunk's rerank_score is below "
        "score_threshold (or no chunks are provided) the LLM answers from its "
        "general knowledge instead (direct mode)."
    ),
    tags=["RAG Pipeline"],
)
async def generate(request: GenerateRequest):
    import time

    agent = get_agent()
    latencies: Dict[str, float] = {}

    # ── Session history ───────────────────────────────────────────────────────
    try:
        from session_manager import load_history, save_turn
        from config import cfg as _cfg
        _use_db = _cfg.USE_DB
    except Exception:
        load_history = lambda *a, **kw: []
        save_turn    = lambda *a, **kw: None
        _use_db = False

    chat_history = []
    if request.session_id and _use_db:
        chat_history = await asyncio.to_thread(load_history, request.session_id)

    try:
        # ── Convert ChunkInput → internal doc dicts ───────────────────────────
        docs = [
            {
                "page_content": c.page_content,
                "metadata": c.metadata or {},
                "rerank_score": c.rerank_score,
                "id": c.chunk_id or "",
            }
            for c in request.chunks
        ]

        # ── Determine generation mode ─────────────────────────────────────────
        threshold = (
            request.score_threshold
            if request.score_threshold is not None
            else agent.score_threshold
        )
        top_score = max(
            (d["rerank_score"] for d in docs if d.get("rerank_score") is not None),
            default=None,
        )
        use_rag = bool(docs) and (top_score is None or top_score >= threshold)

        # ── Temporarily override max_new_tokens if requested ──────────────────
        original_max = agent.llm.max_new_tokens
        if request.max_new_tokens is not None:
            agent.llm.max_new_tokens = request.max_new_tokens
            if agent.llm._pipeline is not None:
                agent.llm._pipeline.model.generation_config.max_new_tokens = request.max_new_tokens

        try:
            t0 = time.perf_counter()
            if use_rag:
                gen_result = await asyncio.to_thread(
                    agent.llm.generate,
                    request.query,
                    docs,
                    request.system_prompt,
                    chat_history or None,
                )
                answer = gen_result["answer"]
                context_used = gen_result.get("context", "")
                mode = "rag"
            else:
                answer = await asyncio.to_thread(
                    agent.llm.generate_direct,
                    request.query,
                    chat_history or None,
                )
                context_used = ""
                mode = "direct"
            latencies["generation"] = round(time.perf_counter() - t0, 4)
        finally:
            if request.max_new_tokens is not None:
                agent.llm.max_new_tokens = original_max

        # ── Persist turn ──────────────────────────────────────────────────────
        if request.session_id and _use_db:
            await asyncio.to_thread(save_turn, request.session_id, request.query, answer)

        latencies["total"] = round(sum(latencies.values()), 4)

        return GenerateResponse(
            answer=answer,
            mode=mode,
            query=request.query,
            context_used=context_used,
            session_id=request.session_id,
            chunks_used=len(docs) if use_rag else 0,
            top_rerank_score=top_score,
            latencies=latencies,
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}")


@app.get(
    "/health",
    summary="Health check",
    description="Returns server status and whether the RAGAgent has been initialised.",
)
async def health():
    from rag_pipeline.vector_store import _qdrant_mode, _qdrant_host, _qdrant_port, _qdrant_url
    agent_ready = _agent is not None

    mode = _qdrant_mode()
    if mode == "server":
        qdrant_info = {"mode": "server", "host": _qdrant_host(), "port": _qdrant_port()}
    elif mode == "cloud":
        qdrant_info = {"mode": "cloud", "url": _qdrant_url()}
    else:
        qdrant_info = {"mode": "local", "path": str(_agent.vector_store.path) if agent_ready else ""}

    qdrant_ok = False
    if agent_ready:
        try:
            _agent.vector_store.client.get_collections()
            qdrant_ok = True
        except Exception:
            qdrant_ok = False

    return {
        "status": "ok",
        "agent_ready": agent_ready,
        "qdrant": {**qdrant_info, "reachable": qdrant_ok},
        "use_db": cfg.USE_DB,
        "debug": cfg.DEBUG,
    }


@app.get(
    "/collections/info",
    summary="List all collections with info and documents",
    description="Returns a list of all collections, their metadata, and their stored documents.",
)
async def get_all_collections_info():
    agent = get_agent()
    try:
        cols_resp = agent.vector_store.client.get_collections()
        result = []
        for c in cols_resp.collections:
            info = agent.vector_store.get_collection_info(c.name)
            docs = await asyncio.to_thread(agent.vector_store.list_documents, c.name)
            result.append({
                "collection_name": c.name,
                "info": info,
                "documents": docs
            })
        return JSONResponse(status_code=200, content={"collections": result})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch collections info: {exc}")


@app.delete(
    "/collection",
    summary="Delete collection",
    description="Permanently drops the initialized Qdrant collection and all its data.",
)
async def delete_collection():
    agent = get_agent()
    try:
        target_col = agent.collection_name
        agent.vector_store.delete_collection(collection_name=target_col)
        return {"status": "deleted", "collection": target_col}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Delete failed: {exc}")


@app.delete(
    "/document/{doc_id}",
    summary="Delete document",
    description="Deletes a document from the initialized collection by its doc_id.",
)
async def delete_document(doc_id: str):
    agent = get_agent()
    try:
        target_col = agent.collection_name
        await asyncio.to_thread(agent.vector_store.delete_document, doc_id, target_col)
        return {"status": "deleted", "doc_id": doc_id, "collection": target_col}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {exc}")


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
    )
