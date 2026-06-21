"""
Decider: tight-prompt + tool-structured outfit analysis.

Instead of streaming prose, the model calls structured tools:
  get_weather()           — fetch conditions at reasoning time
  lookup_dress_code(code) — web lookup for unusual dress codes only
  propose_outfit(...)     — record one viable outfit (called 0..N times)
  report_gap(...)         — record why nothing works (called instead of propose_outfit)

This makes "how many outfits" an explicit model decision, not a side-effect of
prose length. It also enforces the no-unlisted-items constraint at the schema level.
"""
import json
import ssl
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import certifi

from src.context import fetch_cached, to_string
from src.model import run_tool_loop

if TYPE_CHECKING:
    from src.describer import GarmentDescription

_KNOWLEDGE_DIR = Path(__file__).parent.parent / "knowledge"
_SSL_CTX = ssl.create_default_context(cafile=certifi.where())

# ── Output types ──────────────────────────────────────────────────────────────

@dataclass
class ProposedOutfit:
    item_indices: list[int]       # 0-based indices into the garments list (for image lookup)
    item_labels: list[str]        # human-readable garment labels, e.g. "Slim navy wool blazer"
    supplementary_items: list[str]  # needed items NOT in the wardrobe, e.g. ["white crew socks"]
    occasion_match: str           # formality level matched
    weather_rationale: str        # how this handles current conditions
    rule_applied: str             # exact rule from the Styling Reference
    suitability_score: float      # 0.0–1.0

@dataclass
class GapReport:
    explanation: str
    missing_items: list[str]

@dataclass
class AnalysisResult:
    outfits: list[ProposedOutfit] = field(default_factory=list)
    gap: Optional[GapReport] = None

# ── Tools ─────────────────────────────────────────────────────────────────────

_TOOLS = [
    {
        "name": "get_weather",
        "description": (
            "Get current weather: temperature, conditions, wind. "
            "Call this before proposing outfits whenever the occasion is outdoors "
            "or weather-sensitive."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "lookup_dress_code",
        "description": (
            "Look up an unusual dress code on the web. "
            "Only call for codes NOT covered in the Styling Reference "
            "(e.g. 'festive attire', 'garden party casual', 'resort formal'). "
            "Standard codes — casual, smart casual, business casual, business formal, "
            "cocktail, black tie — need no lookup."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dress_code": {"type": "string", "description": "The unusual dress code to look up"},
            },
            "required": ["dress_code"],
        },
    },
    {
        "name": "propose_outfit",
        "description": (
            "Propose a complete, wearable outfit. "
            "Call only when the outfit genuinely works for the occasion and weather — "
            "1 call is correct if only 1 outfit works; do not pad with mediocre alternatives. "
            "Every outfit must include: shirt, pants, socks, shoes. "
            "Exception: a one-piece garment (dress, jumpsuit, romper, bodysuit) replaces "
            "shirt + pants, so the required set becomes one-piece + socks + shoes. "
            "Any required piece missing from the wardrobe goes in supplementary_items "
            "with a specific description (e.g. 'white crew socks', 'black leather Oxford shoes'). "
            "Do NOT list wardrobe items in supplementary_items."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "item_numbers": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "1-based wardrobe item numbers to include",
                },
                "supplementary_items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Items NOT in the wardrobe but required to complete this outfit. "
                        "Be specific: 'white crew socks', 'black leather belt', 'plain white tee'. "
                        "Only list items that are genuinely necessary (e.g. socks when the look needs them), "
                        "not hypothetical upgrades."
                    ),
                },
                "occasion_match": {
                    "type": "string",
                    "description": "Formality level from the Styling Reference this meets",
                },
                "weather_rationale": {
                    "type": "string",
                    "description": "How this outfit handles the current temperature and conditions",
                },
                "rule_applied": {
                    "type": "string",
                    "description": "Exact bullet or sentence from the Styling Reference that justifies this pick",
                },
                "suitability_score": {
                    "type": "number",
                    "description": "0.0–1.0: fit for the occasion and weather",
                },
            },
            "required": ["item_numbers", "supplementary_items", "occasion_match",
                         "weather_rationale", "rule_applied", "suitability_score"],
        },
    },
    {
        "name": "report_gap",
        "description": (
            "Report that no outfit in the wardrobe properly meets this occasion. "
            "Always call this when the wardrobe falls short of the required formality. "
            "You MAY also call propose_outfit once for the single closest available "
            "combination — the best the wardrobe can do even if it undershoots the "
            "occasion. If you do, set suitability_score below 0.5 and occasion_match "
            "to the actual formality level of what you're proposing (not the target). "
            "Do not call propose_outfit if nothing in the wardrobe is even close."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "explanation": {
                    "type": "string",
                    "description": "Why the wardrobe cannot meet this occasion",
                },
                "missing_items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific garment types to buy, e.g. ['navy dress trousers', 'white Oxford shirt', 'leather Oxford shoes']",
                },
            },
            "required": ["explanation", "missing_items"],
        },
    },
]

# ── Prompt builder ─────────────────────────────────────────────────────────────

def _load_knowledge() -> str:
    files = sorted(_KNOWLEDGE_DIR.glob("*.md"))
    return "\n\n---\n\n".join(f.read_text(encoding="utf-8") for f in files)


def _format_wardrobe(garments: list["GarmentDescription"]) -> str:
    lines = [f"{i}. {g.label()}" for i, g in enumerate(garments, 1)]
    return "## Wardrobe\n\n" + "\n".join(lines)


def _build_profile_section(name: str, gender: str, age: str) -> str:
    parts = []
    if name:   parts.append(f"Name: {name}")
    if gender: parts.append(f"Gender: {gender}")
    if age:    parts.append(f"Age: {age}")
    if not parts:
        return ""
    return "## Client Profile\n\n" + "\n".join(parts) + "\n\nAddress them by name when relevant. Tailor all recommendations to their gender and age.\n\n"


def _build_system_prompt(
    knowledge: str,
    garments: list["GarmentDescription"],
    profile_name: str = "",
    profile_gender: str = "",
    profile_age: str = "",
) -> str:
    profile = _build_profile_section(profile_name, profile_gender, profile_age)
    return f"""You are a strict personal outfit advisor.

{profile}

## Rules
1. Call get_weather() first whenever the occasion is outdoors or temperature-sensitive.
2. Call lookup_dress_code(code) ONLY for dress codes not in the Styling Reference. Standard codes need no lookup.
3. Call propose_outfit() for each outfit that genuinely works:
   - Quality over quantity. 1 outfit is the right answer if only 1 works. Do not propose
     alternatives that are marginal, redundant, or stretch the occasion — the user gets
     exactly what you call, so every call must stand on its own.
   - Every outfit must be complete. Required pieces: shirt, pants, socks, shoes.
     Exception: a one-piece garment (dress, jumpsuit, romper, bodysuit) replaces
     shirt + pants, so the required set becomes: one-piece, socks, shoes.
     Use wardrobe item numbers for pieces that exist. Any required piece missing
     from the wardrobe goes in supplementary_items with a specific description
     (e.g. "white crew socks", "black leather Oxford shoes"). An outfit missing
     any required piece is incomplete and must not be proposed without listing
     the gap in supplementary_items.
   - Match the occasion's formality level from the Styling Reference.
   - Respect temperature thresholds:
     * Below 40°F: outer layer required
     * 40–55°F: outer layer strongly recommended
     * 55–70°F: light layer advisable for wind
     * Above 70°F: lightweight fabrics only, no heavy layers
   - Reference only wardrobe item numbers in item_numbers. Never invent items there.
   - Name the exact Styling Reference rule that qualifies this pick.
4. Call report_gap() whenever the wardrobe falls short of the required formality.
   After calling report_gap(), you may optionally call propose_outfit() once for the
   single closest available combination — the best the wardrobe can do, even if it
   undershoots. Set suitability_score below 0.5 and occasion_match to the actual
   formality of that combination (e.g. "casual"), not the target. Skip propose_outfit
   entirely if nothing in the wardrobe is remotely close.

{_format_wardrobe(garments)}

## Styling Reference

{knowledge}"""

# ── Web lookup (dress code only) ──────────────────────────────────────────────

def _web_lookup(dress_code: str) -> str:
    query = urllib.parse.quote(f"{dress_code} dress code what to wear")
    url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
    try:
        with urllib.request.urlopen(url, timeout=5, context=_SSL_CTX) as resp:
            data = json.loads(resp.read())
        abstract = data.get("AbstractText", "")
        if abstract:
            return abstract
        topics = [t["Text"] for t in data.get("RelatedTopics", [])
                  if isinstance(t, dict) and t.get("Text")][:3]
        if topics:
            return "\n".join(topics)
    except Exception:
        pass
    return f"No specific guidance found for '{dress_code}'. Apply best judgment from context."

# ── Public API ─────────────────────────────────────────────────────────────────

OnToolCalled = Callable[[str, dict, str], None]


def analyze(
    user_prompt: str,
    garments: list["GarmentDescription"],
    on_tool_called: Optional[OnToolCalled] = None,
    profile_name: str = "",
    profile_gender: str = "",
    profile_age: str = "",
) -> AnalysisResult:
    """Run the decider tool loop; return structured outfit proposals or a gap report.

    on_tool_called(name, args, result) fires after each tool executes so the
    caller can surface UI feedback (e.g. displaying fetched weather).
    """
    knowledge = _load_knowledge()
    system = _build_system_prompt(knowledge, garments, profile_name, profile_gender, profile_age)

    # Mutable state collected by the tool handler closure
    outfits: list[ProposedOutfit] = []
    gap_holder: list[GapReport] = []  # list avoids nonlocal for a single value

    def handle_tool(name: str, args: dict) -> str:
        if name == "get_weather":
            ctx = fetch_cached()
            result = to_string(ctx) if ctx else "Weather unavailable."

        elif name == "lookup_dress_code":
            result = _web_lookup(args["dress_code"])

        elif name == "propose_outfit":
            indices, labels = [], []
            for n in args.get("item_numbers", []):
                idx = n - 1
                if 0 <= idx < len(garments):
                    indices.append(idx)
                    labels.append(garments[idx].label())
                else:
                    labels.append(f"[invalid item #{n}]")
            outfits.append(ProposedOutfit(
                item_indices=indices,
                item_labels=labels,
                supplementary_items=args.get("supplementary_items", []),
                occasion_match=args["occasion_match"],
                weather_rationale=args["weather_rationale"],
                rule_applied=args["rule_applied"],
                suitability_score=float(args["suitability_score"]),
            ))
            result = "Outfit recorded."

        elif name == "report_gap":
            gap_holder.append(GapReport(
                explanation=args["explanation"],
                missing_items=args.get("missing_items", []),
            ))
            result = "Gap recorded."

        else:
            result = f"Unknown tool: {name}"

        if on_tool_called:
            on_tool_called(name, args, result)
        return result

    run_tool_loop(
        system=system,
        user_msg=user_prompt,
        tools=_TOOLS,
        tool_handler=handle_tool,
    )

    return AnalysisResult(
        outfits=outfits,
        gap=gap_holder[0] if gap_holder else None,
    )
