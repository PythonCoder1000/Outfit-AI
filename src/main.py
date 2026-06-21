import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from src.decider import AnalysisResult, analyze
from src.describer import describe_wardrobe

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
console = Console()


def _load_dotenv() -> None:
    dotenv = _PROJECT_ROOT / ".env"
    if not dotenv.exists():
        return
    for line in dotenv.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _scan_wardrobe() -> list:
    def on_hit(name: str) -> None:
        console.print(f"  [dim]✓ {name}[/dim]")

    def on_miss(name: str) -> None:
        console.print(f"  [yellow]⟳ {name} — describing...[/yellow]")

    def on_unusable(name: str) -> None:
        console.print(f"  [dim red]✗ {name} — unusable image, skipped[/dim red]")

    return describe_wardrobe(on_hit=on_hit, on_miss=on_miss, on_unusable=on_unusable)


def _on_tool(name: str, args: dict, result: str) -> None:
    """Display feedback when the decider calls a tool mid-reasoning."""
    if name == "get_weather":
        # result IS the weather string — show it as a status line
        lines = result.strip().splitlines()
        if lines:
            # Collapse to a single dim line: "📍 San Jose, US  •  70°F  •  clear sky"
            loc_line = next((l for l in lines if "Location:" in l), "")
            temp_line = next((l for l in lines if "Temperature:" in l), "")
            cond_line = next((l for l in lines if "Conditions:" in l), "")
            loc = loc_line.replace("Location:", "").strip()
            temp = temp_line.replace("Temperature:", "").strip()
            cond = cond_line.replace("Conditions:", "").strip()
            console.print(f"[dim]📍 {loc}  •  {temp}  •  {cond}[/dim]")
    elif name == "lookup_dress_code":
        console.print(f"[dim]🔍 Looking up dress code: {args.get('dress_code')}[/dim]")


def _display(result: AnalysisResult) -> None:
    if result.gap:
        missing = ", ".join(result.gap.missing_items) if result.gap.missing_items else "—"
        console.print(Panel(
            f"{result.gap.explanation}\n\n[dim]You'd need:[/dim] {missing}",
            title="[bold yellow]⚠ Wardrobe Gap[/bold yellow]",
            border_style="yellow",
        ))
        return

    if not result.outfits:
        console.print("[dim]No outfits returned.[/dim]")
        return

    for i, outfit in enumerate(result.outfits, 1):
        items_text = "\n".join(f"• {label}" for label in outfit.item_labels)
        score_blocks = "█" * round(outfit.suitability_score * 10)
        score_empty  = "░" * (10 - round(outfit.suitability_score * 10))
        body = (
            f"{items_text}\n\n"
            f"[dim]Formality:[/dim] {outfit.occasion_match}\n"
            f"[dim]Weather:[/dim]   {outfit.weather_rationale}\n"
            f"[dim]Rule:[/dim]      {outfit.rule_applied}\n"
            f"[dim]Score:[/dim]     [cyan]{score_blocks}[/cyan]{score_empty} "
            f"{outfit.suitability_score:.0%}"
        )
        console.print(Panel(
            body,
            title=f"[bold cyan]Outfit {i}[/bold cyan]",
            border_style="cyan",
        ))
        console.print()


def main() -> None:
    _load_dotenv()

    console.print(Rule("[bold cyan]Outfit AI — Personal Stylist[/bold cyan]"))

    wardrobe_dir = _PROJECT_ROOT / "wardrobe"
    image_files = [
        f for f in wardrobe_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".webp"}
    ]

    if image_files:
        console.print(f"\n[bold]Scanning wardrobe[/bold] ({len(image_files)} items)...")
        garments = _scan_wardrobe()
        console.print(f"[green]✓ {len(garments)} garment(s) ready.[/green]\n")
    else:
        console.print("[dim]No images in wardrobe/ — recommendations will be generic.[/dim]\n")
        garments = []

    console.print("[dim]Type 'quit' or Ctrl+C to exit.[/dim]\n")

    while True:
        try:
            prompt = console.input("[bold green]What are you dressing for?[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not prompt:
            continue
        if prompt.lower() in {"quit", "exit", "q"}:
            console.print("[dim]Goodbye![/dim]")
            break

        console.print()
        result = analyze(prompt, garments=garments, on_tool_called=_on_tool)
        console.print()
        _display(result)
