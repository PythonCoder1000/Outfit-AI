"""
Describer: wardrobe image → structured GarmentDescription, with persistent cache.

Cache design
------------
- Stored at cache/garments.json.
- Key: SHA-256 of the image bytes (content-addressed — rename = same hit, edit = new hash).
- Each entry carries SCHEMA_VERSION; bump it to force a full re-describe on next run.
- A None description means the image was flagged unusable; it's cached so we don't retry it.
"""
import hashlib
import json
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel, Field

from src.model import vision_describe

SCHEMA_VERSION = "1.1"

_WARDROBE_DIR = Path(__file__).parent.parent / "wardrobe"
_CACHE_PATH   = Path(__file__).parent.parent / "cache" / "garments.json"
_IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

_TOOLS = [
    {
        "name": "describe_garment",
        "description": "Record a structured description of a garment from its photo.",
        "input_schema": {
            "type": "object",
            "properties": {
                "type":            {"type": "string",
                                    "description": "Garment type, e.g. 't-shirt', 'chinos', 'blazer', 'dress'"},
                "color":           {"type": "string",
                                    "description": "Primary color, e.g. 'navy blue', 'white'"},
                "secondary_color": {"type": "string",
                                    "description": "Secondary or accent color; empty string if none"},
                "pattern":         {"type": "string",
                                    "description": "Pattern: 'solid', 'striped', 'plaid', 'floral', 'checked', etc."},
                "material":        {"type": "string",
                                    "description": "Fabric, e.g. 'cotton', 'wool blend', 'denim', 'linen'"},
                "fit":             {"type": "string",
                                    "description": "Fit: 'slim', 'regular', 'relaxed', 'oversized'"},
                "formality":       {"type": "string",
                                    "description": "'casual', 'smart casual', 'business casual', or 'formal'"},
                "season":          {"type": "array", "items": {"type": "string"},
                                    "description": "Seasons this works for, e.g. ['spring', 'fall']"},
                "confidence":      {"type": "number",
                                    "description": "0.0–1.0 confidence in this description; lower if image is poor quality"},
            },
            "required": ["type", "color", "secondary_color", "pattern",
                         "material", "fit", "formality", "season", "confidence"],
        },
    },
    {
        "name": "flag_unusable_image",
        "description": (
            "Flag this image as unusable. Use when the photo is blurry, empty, "
            "shows something that isn't a garment, or is otherwise undescribable."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Why the image cannot be described"},
            },
            "required": ["reason"],
        },
    },
]


class GarmentDescription(BaseModel):
    type: str
    color: str
    secondary_color: str
    pattern: str
    material: str
    fit: str
    formality: str
    season: list[str]
    confidence: float
    source_file: str = Field(default="", exclude=True)  # set at runtime, never cached

    def label(self) -> str:
        """Short human-readable label for display and prompts."""
        parts = [self.fit.capitalize(), self.color]
        if self.secondary_color:
            parts.append(f"/ {self.secondary_color}")
        if self.pattern != "solid":
            parts.append(self.pattern)
        parts += [self.material, self.type]
        return " ".join(parts)


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_cache() -> dict:
    if _CACHE_PATH.exists():
        return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict) -> None:
    _CACHE_PATH.parent.mkdir(exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _describe_image(path: Path) -> Optional["GarmentDescription"]:
    """Call the vision API; return None if the model flags the image as unusable."""
    tool_name, raw = vision_describe(
        path,
        prompt=(
            "Describe this garment using describe_garment. "
            "If the image is blurry, empty, or not a garment, call flag_unusable_image instead."
        ),
        tools=_TOOLS,
    )
    if tool_name == "flag_unusable_image":
        return None
    return GarmentDescription(**raw)


def describe_single(image_path: Path) -> Optional[GarmentDescription]:
    """Describe one image, write it to the cache, and return the result.

    Returns None if the model flags the image as unusable. Callers should
    delete the file in that case — we still cache the None so a retry of the
    same bytes doesn't call the API again.
    """
    file_hash = _hash_file(image_path)
    cache = _load_cache()
    # Honour an existing valid cache entry (e.g. same image re-uploaded).
    entry = cache.get(file_hash)
    if entry and entry.get("schema_version") == SCHEMA_VERSION:
        desc_data = entry.get("description")
        if desc_data:
            return GarmentDescription(**desc_data, source_file=image_path.name)
        return None

    description = _describe_image(image_path)
    cache[file_hash] = {
        "schema_version": SCHEMA_VERSION,
        "file_name": image_path.name,
        "description": description.model_dump() if description else None,
    }
    _save_cache(cache)
    if description:
        return description.model_copy(update={"source_file": image_path.name})
    return None


def remove_from_cache(filename: str) -> None:
    """Remove all cache entries whose file_name matches filename."""
    cache = _load_cache()
    stale = [k for k, v in cache.items() if v.get("file_name") == filename]
    if stale:
        for k in stale:
            del cache[k]
        _save_cache(cache)


def describe_wardrobe(
    on_hit: Optional[Callable[[str], None]] = None,
    on_miss: Optional[Callable[[str], None]] = None,
    on_unusable: Optional[Callable[[str], None]] = None,
) -> list[GarmentDescription]:
    """Return descriptions for every usable image in the wardrobe directory.

    Cache hits skip the API entirely. Unusable images are cached as None so
    they aren't retried on future runs. Call describe_wardrobe() once at
    startup; the results stay valid for the session.
    """
    cache = _load_cache()
    results: list[GarmentDescription] = []
    cache_dirty = False

    for image_path in sorted(_WARDROBE_DIR.iterdir()):
        if image_path.suffix.lower() not in _IMAGE_EXTS:
            continue

        file_hash = _hash_file(image_path)
        entry = cache.get(file_hash)

        if entry and entry.get("schema_version") == SCHEMA_VERSION:
            desc_data = entry.get("description")
            if desc_data is None:
                if on_unusable:
                    on_unusable(image_path.name)
            else:
                if on_hit:
                    on_hit(image_path.name)
                results.append(GarmentDescription(**desc_data, source_file=image_path.name))
            continue

        # Cache miss — call the vision API
        if on_miss:
            on_miss(image_path.name)
        description = _describe_image(image_path)
        cache[file_hash] = {
            "schema_version": SCHEMA_VERSION,
            "file_name": image_path.name,
            "description": description.model_dump() if description else None,
        }
        cache_dirty = True
        if description:
            description = description.model_copy(update={"source_file": image_path.name})
            results.append(description)
        else:
            if on_unusable:
                on_unusable(image_path.name)

    if cache_dirty:
        _save_cache(cache)

    return results
