"""Terminal output with Rich: spinner, header, article panels and warnings."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from newspulse.models import Article, RunResult


def make_console() -> Console:
    """Create the console shared by logging, the spinner and the results."""
    return Console(highlight=False)


@contextmanager
def progress_spinner(console: Console, message: str = "Starting...") -> Iterator[Callable[[str], None]]:
    """Show a spinner while the pipeline runs; yields a function that updates its text."""
    with console.status(f"[bold cyan]{message}", spinner="dots") as status:
        yield lambda text: status.update(f"[bold cyan]{text}")


def _badges(article: Article) -> Text:
    """Coloured badges for text availability and missing fields."""
    badges = Text()
    if article.text_type == "full_text":
        badges.append(" full text ", style="bold black on green")
    elif article.text_type == "feed_excerpt":
        badges.append(" excerpt only ", style="bold black on yellow")
    else:
        badges.append(" no text ", style="bold white on red")
    if article.missing_fields:
        badges.append("  ")
        badges.append(f" missing: {', '.join(article.missing_fields)} ", style="bold white on red")
    if article.summary_method:
        badges.append("  ")
        badges.append(f" summary: {article.summary_method} ", style="dim")
    return badges


def _article_panel(index: int, article: Article) -> Panel:
    """Build the Rich panel for one article."""
    date = article.published_at[:16].replace("T", " ") + " UTC" if article.published_at else "date unknown"
    meta = Text(f"{article.source or 'Unknown publisher'} · {date} · via {article.discovered_via}",
                style="cyan")
    summary = Text(article.summary or "No summary available.",
                   style="" if article.summary else "italic dim")
    url = Text(article.url or "no URL", style="blue underline")
    body = Group(meta, Text(), summary, Text(), url, Text(), _badges(article))
    return Panel(body, title=f"[bold]{index}. {article.title or 'Untitled'}[/bold]",
                 title_align="left", border_style="bright_black", padding=(1, 2))


def show_results(result: RunResult, console: Console | None = None) -> None:
    """Print the header, one panel per article, then any shortfall note and source errors."""
    console = console or make_console()
    console.print()
    console.print(Rule(f"[bold]NewsPulse · {result.topic}[/bold]"))
    console.print(
        f"[bold]{result.article_count}[/bold] of {result.requested_limit} requested articles · "
        f"last {result.days} days · sources: {', '.join(result.sources_queried)} · "
        f"generated {result.generated_at}",
        justify="center",
    )
    console.print()
    for index, article in enumerate(result.articles, start=1):
        console.print(_article_panel(index, article))

    if result.shortfall_note:
        console.print(Panel(result.shortfall_note, title="Shortfall", border_style="yellow"))
    if result.source_errors:
        lines = "\n".join(f"• {e.source}: {e.error}" for e in result.source_errors)
        console.print(Panel(lines, title="Source errors", border_style="red"))


def show_saved_files(paths: list[Path], console: Console | None = None) -> None:
    """Print the paths of the files written by the exporter."""
    console = console or make_console()
    console.print("\n[bold green]Saved files:[/bold green]")
    for path in paths:
        console.print(f"  {path}")
