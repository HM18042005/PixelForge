import os
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

import gridfs
from bson import ObjectId
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field
from pymongo import MongoClient

from generate import (
    CFG_SCALE,
    HEIGHT,
    LORA_OUTPUT_SCALE,
    LORA_SCALE,
    NEGATIVE_PROMPT,
    PROMPT,
    SEED,
    STEPS,
    WIDTH,
    generate_images,
)

MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGODB_DB", "pixelforge")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
fs = gridfs.GridFS(db)

app = FastAPI(title="PixelForge API", version="0.1.0")


class GenerateRequest(BaseModel):
    prompt: str = Field(default=PROMPT, description="Text prompt for generation")
    negative_prompt: str = Field(default=NEGATIVE_PROMPT, description="Negative prompt")
    steps: int = Field(default=STEPS, ge=1, le=150)
    guidance_scale: float = Field(default=CFG_SCALE, ge=0.0, le=30.0)
    width: int = Field(default=WIDTH, ge=64, le=1024, multiple_of=8)
    height: int = Field(default=HEIGHT, ge=64, le=1024, multiple_of=8)
    seed: int = Field(default=SEED, ge=0)
    lora_indoor_scale: float = Field(default=LORA_SCALE, ge=0.0, le=3.0)
    lora_output_scale: float = Field(default=LORA_OUTPUT_SCALE, ge=0.0, le=3.0)
    include_comparison: bool = Field(default=True)


class ImageSummary(BaseModel):
    id: str
    filename: Optional[str]
    kind: Optional[str]
    prompt: Optional[str]
    seed: Optional[int]
    width: Optional[int]
    height: Optional[int]
    lora_scales: Optional[Dict[str, float]]
    created_at: Optional[str]


class ImageListResponse(BaseModel):
    items: List[ImageSummary]


def _object_id(id_str: str) -> ObjectId:
    try:
        return ObjectId(id_str)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image id")


def _dt_to_iso(value: Any) -> Optional[str]:
    if isinstance(value, datetime):
        return value.isoformat() + "Z"
    return None


@app.post("/generate")
def generate_and_store(request: GenerateRequest):
    try:
        result = generate_images(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            steps=request.steps,
            guidance_scale=request.guidance_scale,
            width=request.width,
            height=request.height,
            seed=request.seed,
            lora_scales={
                "indoor": request.lora_indoor_scale,
                "output": request.lora_output_scale,
            },
            include_comparison=request.include_comparison,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {exc}") from exc

    created_at = datetime.utcnow()
    base_metadata = {**result.metadata, "created_at": created_at}

    ids: Dict[str, str] = {}
    for name, image in result.images().items():
        if image is None:
            continue
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        file_id = fs.put(
            buffer.getvalue(),
            filename=f"{name}.png",
            metadata={**base_metadata, "kind": name},
        )
        ids[name] = str(file_id)

    return {
        "image_ids": ids,
        "metadata": {**result.metadata, "created_at": created_at.isoformat() + "Z"},
    }


@app.get("/images/{image_id}")
def fetch_image(image_id: str):
    oid = _object_id(image_id)
    try:
        grid_out = fs.get(oid)
    except Exception:
        raise HTTPException(status_code=404, detail="Image not found")

    filename = grid_out.filename or "image.png"
    return Response(
        content=grid_out.read(),
        media_type="image/png",
        headers={"Content-Disposition": f"inline; filename=\"{filename}\""},
    )


@app.get("/images", response_model=ImageListResponse)
def list_images(limit: int = Query(default=20, ge=1, le=100)):
    cursor = db.fs.files.find().sort("uploadDate", -1).limit(limit)
    items: List[ImageSummary] = []
    for doc in cursor:
        metadata = doc.get("metadata", {})
        created_at = metadata.get("created_at") or doc.get("uploadDate")
        items.append(
            ImageSummary(
                id=str(doc.get("_id")),
                filename=doc.get("filename"),
                kind=metadata.get("kind"),
                prompt=metadata.get("prompt"),
                seed=metadata.get("seed"),
                width=metadata.get("width"),
                height=metadata.get("height"),
                lora_scales=metadata.get("lora_scales"),
                created_at=_dt_to_iso(created_at),
            )
        )

    return ImageListResponse(items=items)
