# api/knowledge_bases.py
from fastapi import APIRouter, BackgroundTasks, Depends, File, UploadFile, HTTPException, status, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from bson import ObjectId
from datetime import datetime
import os
import uuid
import aiofiles
from core.db import get_db
from core.config import settings

from services.ingestion import run_ingestion_task  

router = APIRouter()

# ---------- Utilities ----------
def oid_to_str(o: ObjectId) -> str:
    return str(o)


# ---------- Pydantic Schemas ----------
class KBCreateRequest(BaseModel):
    name: str = Field(..., example="Health Insurance Plan A")
    kb_type: Optional[str] = Field("generic", example="health_insurance")
    description: Optional[str] = Field("", example="Short description about this KB")


class KBResponse(BaseModel):
    kb_id: str
    name: str
    kb_type: str
    description: Optional[str]
    created_at: datetime
    last_updated: datetime


# ---------- Routes ----------
@router.post("/create", response_model=KBResponse, status_code=status.HTTP_201_CREATED)
async def create_kb(payload: KBCreateRequest):
    """
    Create a Knowledge Base metadata entry.
    """
    db = get_db()
    kb_doc = {
        "name": payload.name,
        "kb_type": payload.kb_type,
        "description": payload.description or "",
        "created_at": datetime.utcnow(),
        "last_updated": datetime.utcnow(),
    }
    res = await db.kbs.insert_one(kb_doc)
    created = await db.kbs.find_one({"_id": res.inserted_id})
    return KBResponse(
        kb_id=oid_to_str(created["_id"]),
        name=created["name"],
        kb_type=created["kb_type"],
        description=created.get("description", ""),
        created_at=created["created_at"],
        last_updated=created["last_updated"],
    )


@router.post("/{kb_id}/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_documents(
    kb_id: str,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    description: Optional[str] = Form(None),
):
    """
    Upload one or more files for a KB. Files are saved temporarily and ingestion is triggered in background.
    Returns quickly with 202 Accepted while ingestion runs.
    """
    db = get_db()

    # Validate KB exists
    kb = await db.kbs.find_one({"_id": ObjectId(kb_id)})
    if not kb:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge Base not found")

    # Save uploaded files to a temp folder (unique per request)
    upload_dir = os.path.join("uploads", kb_id, str(uuid.uuid4()))
    os.makedirs(upload_dir, exist_ok=True)
    saved_paths: List[str] = []

    try:
        for f in files:
            # Basic validation on file size/type can be added here
            filename = f.filename
            safe_name = filename.replace("..", "_")
            dest_path = os.path.join(upload_dir, safe_name)
            async with aiofiles.open(dest_path, "wb") as out_file:
                content = await f.read()
                await out_file.write(content)
            saved_paths.append(dest_path)

    except Exception as e:
        # Cleanup partially written files
        for p in saved_paths:
            try:
                os.remove(p)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    # update KB metadata last_updated and optional description
    update_doc = {"last_updated": datetime.utcnow()}
    if description:
        update_doc["description"] = description
    await db.kbs.update_one({"_id": ObjectId(kb_id)}, {"$set": update_doc})

    # Trigger background ingestion using the SYNC wrapper
    # This is the key fix - use run_ingestion_task (sync) instead of start_ingestion_task (async)
    background_tasks.add_task(run_ingestion_task, kb_id, saved_paths)

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "status": "accepted",
            "message": "Files received. Ingestion started in background.",
            "kb_id": kb_id,
            "files_received": [os.path.basename(p) for p in saved_paths],
        },
    )


@router.post("/{kb_id}/add-documents", status_code=status.HTTP_202_ACCEPTED)
async def add_documents(kb_id: str, background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """
    Alias to upload documents to an existing KB (keeps semantics explicit).
    """
    return await upload_documents(kb_id=kb_id, background_tasks=background_tasks, files=files)


@router.get("/", response_model=List[KBResponse])
async def list_kbs():
    """
    List all Knowledge Bases (basic metadata).
    """
    db = get_db()
    cursor = db.kbs.find({}, sort=[("created_at", -1)])
    results = []
    async for doc in cursor:
        results.append(
            KBResponse(
                kb_id=oid_to_str(doc["_id"]),
                name=doc["name"],
                kb_type=doc.get("kb_type", "generic"),
                description=doc.get("description", ""),
                created_at=doc["created_at"],
                last_updated=doc["last_updated"],
            )
        )
    return results


@router.get("/{kb_id}", response_model=KBResponse)
async def get_kb(kb_id: str):
    db = get_db()
    doc = await db.kbs.find_one({"_id": ObjectId(kb_id)})
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge Base not found")
    return KBResponse(
        kb_id=oid_to_str(doc["_id"]),
        name=doc["name"],
        kb_type=doc.get("kb_type", "generic"),
        description=doc.get("description", ""),
        created_at=doc["created_at"],
        last_updated=doc["last_updated"],
    )