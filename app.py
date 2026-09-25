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
        raise HTTPException(status_code=503, detail="RAGAgent is not initialised. Call POST /init first.")
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


class InitRequest(BaseModel):
    collection_name: str = ""
    qdrant_path: str = ""
    embedding_model_path: str = r"E:\ds-team\huggingface_models\Old BKP\test_models\bge_m3"
    reranker_model_path: str = r"E:\ds-team\huggingface_models\Old BKP\test_models\bge_reranker_v2_m3"
    generator_model_path: str = r"E:\DS_Team_Share\hugging-face-model\Qwen2.5-14B-Instruct\models--Qwen--Qwen2.5-14B-Instruct\snapshots\cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
    device: str = "cuda"
    top_k: int = Field(default=8, ge=1, le=50)
    top_n: int = Field(default=3, ge=1, le=20)
    score_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    gpu_batch_size: int = Field(default=32, ge=1)
    max_new_tokens: int = Field(default=512, ge=64)
    quantize: bool = True


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to answer.")
    collection_name: Optional[str] = Field(default=None, description="Optional collection to query. Defaults to active collection.")
    doc_ids: Optional[List[str]] = Field(default=None, description="Optional list of document IDs to filter the search by.")
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional session ID (UUID from the DB). When supplied the LLM receives "
            "the prior conversation turns for this session as context, and the new "
            "Q/A pair is saved back so future queries remain context-aware."
        ),
    )


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
    "/init",
    summary="Initialise the RAG agent",
    description="Creates the RAGAgent singleton with the given config. Must be called before /ingest or /query.",
)
async def init_agent(request: InitRequest):
    global _agent
    try:
        qdrant_path = request.qdrant_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "rag_pipeline", "qdrant_store"
        )
        _agent = RAGAgent(
            embedding_model_path=request.embedding_model_path,
            reranker_model_path=request.reranker_model_path,
            generator_model_path=request.generator_model_path,
            qdrant_path=qdrant_path,
            collection_name=request.collection_name,
            device=request.device,
            top_k=request.top_k,
            top_n=request.top_n,
            score_threshold=request.score_threshold,
            gpu_batch_size=request.gpu_batch_size,
            max_new_tokens=request.max_new_tokens,
            quantize=request.quantize,
        )
        return JSONResponse(
            status_code=200,
            content={"status": "initialised", "collection": request.collection_name},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent initialisation failed: {exc}")


@app.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest PDF documents",
    description=(
        "Upload one or more PDF files via multipart/form-data. "
        "Files are saved to disk, chunked, embedded, and indexed into Qdrant. "
        "If collection_name is provided, it is used; otherwise a unique ID (e.g. s5abdd6cc8d92) is generated. "
        "Set force_recreate=true to drop and rebuild the collection from scratch. "
        "Returns a unique doc_id for each uploaded file."
    ),
)
async def ingest(
    files: Annotated[
        List[UploadFile],
        File(description="One or more PDF documents to ingest."),
    ],
    collection_name: Annotated[
        Optional[str],
        Form(description="Optional custom collection name. If empty, a unique ID is auto-generated."),
    ] = None,
    force_recreate: Annotated[
        bool,
        Form(description="Drop and recreate the Qdrant collection before indexing."),
    ] = False,
):
    import uuid
    agent = get_agent()

    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    if collection_name and collection_name.strip():
        target_collection = collection_name.strip()
    else:
        target_collection = f"s{uuid.uuid4().hex[:12]}"

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

        result = await asyncio.to_thread(agent.ingest, saved_paths, force_recreate, doc_mapping, target_collection)
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
            request.collection_name
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
                "collection_name":  request.collection_name
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")



@app.get(
    "/health",
    summary="Health check",
    description="Returns server status and whether the RAGAgent has been initialised.",
)
async def health():
    agent_ready = _agent is not None
    client_dir = []
    if agent_ready:
        client_dir = dir(_agent.vector_store.client)
    return {
        "status": "ok",
        "agent_ready": agent_ready,
        "collection": _agent.collection_name if _agent else None,
        "client_dir": client_dir,
    }


@app.get(
    "/collections",
    summary="List all collections",
    description="Returns a list of all collection names currently present in Qdrant.",
)
async def list_collections():
    agent = get_agent()
    try:
        cols_resp = agent.vector_store.client.get_collections()
        return {"collections": [c.name for c in cols_resp.collections]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {exc}")


@app.get(
    "/collection/info",
    summary="Collection stats and documents",
    description="Returns Qdrant collection metadata and a deduplicated list of every ingested document.",
)
async def collection_info(collection_name: Optional[str] = None):
    agent = get_agent()
    try:
        info = agent.vector_store.get_collection_info(collection_name=collection_name)
        docs = await asyncio.to_thread(agent.vector_store.list_documents)
        info["documents"] = docs
        info["total_documents"] = len(docs)
        return JSONResponse(status_code=200, content=info)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not fetch collection info and documents: {exc}")


@app.delete(
    "/collection",
    summary="Delete collection",
    description="Permanently drops a Qdrant collection and all its data.",
)
async def delete_collection(collection_name: Optional[str] = None):
    agent = get_agent()
    try:
        target_col = collection_name or agent.collection_name
        agent.vector_store.delete_collection(collection_name=target_col)
        return {"status": "deleted", "collection": target_col}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Delete failed: {exc}")


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
    )
