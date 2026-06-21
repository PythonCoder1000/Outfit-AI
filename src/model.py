"""
Provider abstraction — all Anthropic SDK calls live here.
Swap this file to change model backends without touching any other module.
"""
import base64
from pathlib import Path
from typing import Callable, Generator

import anthropic

MODEL = "claude-sonnet-4-6"

_MEDIA_TYPES = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp",
}


def stream_reply(system: str, user_prompt: str) -> Generator[str, None, None]:
    """Stream a text reply token by token."""
    client = anthropic.Anthropic()
    with client.messages.stream(
        model=MODEL,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user_prompt}],
    ) as stream:
        yield from stream.text_stream


def vision_describe(image_path: Path, prompt: str, tools: list[dict]) -> tuple[str, dict]:
    """Send an image + prompt with tool_choice=any; return (tool_name, tool_input).

    Using tool_choice='any' lets the model pick between multiple tools —
    e.g. describe_garment vs flag_unusable_image — rather than forcing one.
    """
    media_type = _MEDIA_TYPES.get(image_path.suffix.lower(), "image/jpeg")
    image_data = base64.standard_b64encode(image_path.read_bytes()).decode()

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        tools=tools,
        tool_choice={"type": "any"},
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )

    for block in response.content:
        if block.type == "tool_use":
            return block.name, block.input

    raise ValueError(f"Model did not call a tool for {image_path.name}")


def run_tool_loop(
    system: str,
    user_msg: str,
    tools: list[dict],
    tool_handler: Callable[[str, dict], str],
    max_turns: int = 10,
) -> None:
    """Run a tool-use conversation until the model stops calling tools.

    tool_handler(name, args) → string result returned to the model.
    Recording tools (propose_outfit, report_gap) work by side-effecting the
    caller's state inside tool_handler; this function doesn't need to know that.
    """
    client = anthropic.Anthropic()
    messages: list[dict] = [{"role": "user", "content": user_msg}]

    for _ in range(max_turns):
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=system,
            tools=tools,
            messages=messages,
        )

        tool_blocks = [b for b in response.content if b.type == "tool_use"]

        if not tool_blocks or response.stop_reason == "end_turn":
            break

        tool_results = [
            {
                "type": "tool_result",
                "tool_use_id": b.id,
                "content": tool_handler(b.name, b.input),
            }
            for b in tool_blocks
        ]

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})
