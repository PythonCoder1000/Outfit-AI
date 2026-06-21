"""
FastAPI backend for the OutfitAI web app.

Serves the frontend at / and exposes three API routes:
  GET  /api/wardrobe  — garment list (cached after startup)
  GET  /api/weather   — live weather with 5-min TTL
  POST /api/analyze   — SSE stream of tool events + outfit/gap results
"""
import asyncio
import dataclasses
import json
import os
import queue
import sys
import threading
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Load .env before any SDK client is instantiated
def _load_dotenv() -> None:
    dotenv = _ROOT / ".env"
    if not dotenv.exists():
        return
    for line in dotenv.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

_load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel as PydanticModel

from src.context import fetch_cached, to_string
from src.decider import analyze
from src.describer import (
    GarmentDescription,
    describe_single,
    describe_wardrobe,
    remove_from_cache,
)

app = FastAPI(title="OutfitAI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_FRONTEND = _ROOT / "frontend"
_WARDROBE = _ROOT / "wardrobe"
_ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

app.mount("/wardrobe", StaticFiles(directory=str(_WARDROBE)), name="wardrobe")

_garments: list[GarmentDescription] = []


@app.on_event("startup")
async def _startup() -> None:
    global _garments
    print("Scanning wardrobe…")
    _garments = await asyncio.to_thread(describe_wardrobe)
    print(f"Ready — {len(_garments)} garment(s) loaded.")


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(str(_FRONTEND / "index.html"))


@app.get("/api/wardrobe")
async def api_wardrobe() -> dict:
    return {
        "garments": [
            {
                **g.model_dump(),
                "source_file": g.source_file,
                "image_url": f"/wardrobe/{g.source_file}",
            }
            for g in _garments
        ]
    }


@app.post("/api/wardrobe/upload")
async def api_upload(file: UploadFile = File(...)) -> dict:
    global _garments
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_EXTS:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Use JPG, PNG, GIF, or WEBP.")

    # Strip any path components the client may have sent, then de-conflict.
    safe_name = Path(file.filename or "upload").name
    dest = _WARDROBE / safe_name
    if dest.exists():
        stem, ext = dest.stem, dest.suffix
        counter = 1
        while dest.exists():
            dest = _WARDROBE / f"{stem}_{counter}{ext}"
            counter += 1

    dest.write_bytes(await file.read())

    description = await asyncio.to_thread(describe_single, dest)
    if description is None:
        dest.unlink(missing_ok=True)
        raise HTTPException(422, "Image could not be described — try a clearer photo of a single garment.")

    _garments.append(description)
    return {
        **description.model_dump(),
        "source_file": description.source_file,
        "image_url": f"/wardrobe/{description.source_file}",
    }


@app.delete("/api/wardrobe/{filename}")
async def api_delete(filename: str) -> dict:
    global _garments
    # Prevent path traversal: only the bare filename is allowed.
    safe_name = Path(filename).name
    if safe_name != filename:
        raise HTTPException(400, "Invalid filename.")
    target = _WARDROBE / safe_name
    if not target.exists():
        raise HTTPException(404, "File not found.")

    await asyncio.to_thread(remove_from_cache, safe_name)
    target.unlink()
    _garments = [g for g in _garments if g.source_file != safe_name]
    return {"deleted": safe_name}


@app.get("/api/weather")
async def api_weather() -> dict:
    ctx = await asyncio.to_thread(fetch_cached)
    if ctx is None:
        return {"error": "unavailable"}
    return {
        "city": ctx.city,
        "country": ctx.country,
        "temp_f": ctx.temp_f,
        "feels_like_f": ctx.feels_like_f,
        "conditions": ctx.conditions,
        "wind_mph": ctx.wind_mph,
        "now": ctx.now,
    }


class AnalyzeRequest(PydanticModel):
    prompt: str
    profile_name: str = ""
    profile_gender: str = ""
    profile_age: str = ""


def _outfit_to_dict(outfit, garments: list[GarmentDescription]) -> dict:
    images = [garments[i].source_file for i in outfit.item_indices if i < len(garments)]
    return {
        "item_labels": outfit.item_labels,
        "item_images": images,
        "item_indices": outfit.item_indices,
        "supplementary_items": outfit.supplementary_items,
        "occasion_match": outfit.occasion_match,
        "weather_rationale": outfit.weather_rationale,
        "rule_applied": outfit.rule_applied,
        "suitability_score": outfit.suitability_score,
    }


@app.post("/api/analyze")
async def api_analyze(body: AnalyzeRequest) -> StreamingResponse:
    event_queue: queue.Queue = queue.Queue()
    garments_snapshot = list(_garments)

    def run() -> None:
        def on_tool(name: str, args: dict, result: str) -> None:
            event_queue.put(json.dumps({"type": "tool", "name": name, "result": result}))

        result = analyze(
            body.prompt,
            garments=garments_snapshot,
            on_tool_called=on_tool,
            profile_name=body.profile_name,
            profile_gender=body.profile_gender,
            profile_age=body.profile_age,
        )

        for outfit in result.outfits:
            event_queue.put(json.dumps({
                "type": "outfit",
                "data": _outfit_to_dict(outfit, garments_snapshot),
            }))

        if result.gap:
            event_queue.put(json.dumps({
                "type": "gap",
                "data": dataclasses.asdict(result.gap),
            }))

        event_queue.put(None)  # sentinel

    threading.Thread(target=run, daemon=True).start()

    async def stream():
        loop = asyncio.get_event_loop()
        while True:
            event = await loop.run_in_executor(None, event_queue.get)
            if event is None:
                yield 'data: {"type":"done"}\n\n'
                break
            yield f"data: {event}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
