"""NewsPulse command-line entry point.

Usage: python main.py --topic "electric vehicles" --limit 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from rich.console import Console
from rich.logging import RichHandler

from newspulse import display, exporter, pipeline

NOISY_LOGGERS = ("trafilatura", "htmldate", "courlan", "urllib3", "charset_normalizer",
                 "httpx", "google_genai", "googlenewsdecoder")


def bounded_int(low: int, high: int) -> type:
    """Return an argparse type function that accepts integers in [low, high]."""

    def parse(value: str) -> int:
        """Parse and range-check one integer argument."""
        try:
            number = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"'{value}' is not a whole number") from None
        if not low <= number <= high:
            raise argparse.ArgumentTypeError(f"must be between {low} and {high} (got {number})")
        return number

    return parse  # type: ignore[return-value]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Define and parse the command-line options."""
    parser = argparse.ArgumentParser(
        prog="newspulse",
        description="Find recent news on a topic, extract the text and summarize each article.",
    )
    parser.add_argument("--topic", type=str, help="topic or keywords to search for (asked if missing)")
    parser.add_argument("--limit", type=bounded_int(1, 50), default=5, help="max articles, 1-50 (default 5)")
    parser.add_argument("--days", type=bounded_int(1, 30), default=7, help="only the last N days, 1-30 (default 7)")
    parser.add_argument("--summarizer", choices=("gemini", "extractive"), default="gemini",
                        help="summary method (default gemini; falls back to extractive)")
    parser.add_argument("--format", dest="fmt", choices=("json", "csv", "both"), default="both",
                        help="data file format (default both); digest.md is always written")
    parser.add_argument("--output-dir", default="output", help="where results are saved (default output)")
    parser.add_argument("--verbose", action="store_true", help="show detailed debug logs")
    return parser.parse_args(argv)


def ask_topic() -> str:
    """Prompt until the user enters a non-empty topic."""
    while True:
        topic = input("Enter a news topic: ").strip()
        if topic:
            return topic
        print("Topic cannot be empty. Please try again.")


def setup_logging(console: Console, verbose: bool) -> None:
    """Send log messages through Rich so they render cleanly above the spinner."""
    handler = RichHandler(console=console, show_path=False, show_time=verbose, markup=False)
    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING,
                        format="%(message)s", handlers=[handler], force=True)
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.INFO if verbose else logging.ERROR)


def main(argv: Sequence[str] | None = None) -> int:
    """Run NewsPulse and return a process exit code."""
    args = parse_args(argv)
    console = display.make_console()
    setup_logging(console, args.verbose)

    try:
        topic = (args.topic or "").strip() if args.topic is not None else ask_topic()
        if not topic:
            console.print("[red]Error:[/red] --topic cannot be empty.")
            return 2

        with display.progress_spinner(console, f"Searching news for '{topic}'...") as update:
            result = pipeline.run_pipeline(topic, args.limit, args.days, args.summarizer, progress=update)

        display.show_results(result, console)
        paths = exporter.export_results(result, args.output_dir, args.fmt)
        display.show_saved_files(paths, console)
        return 0
    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Cancelled.[/yellow]")
        return 130
    except OSError as exc:
        console.print(f"[red]Error:[/red] could not write output files: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001 - never show a traceback for expected problems
        if args.verbose:
            logging.getLogger("newspulse").exception("Unexpected error")
        console.print(f"[red]Error:[/red] {type(exc).__name__}: {exc}")
        console.print("Run again with --verbose for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
