"""Save a RunResult as JSON, CSV and a Markdown digest."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from newspulse.models import Article, RunResult

CSV_COLUMNS = list(Article.model_fields.keys())
MAX_SLUG_LENGTH = 50


def slugify(topic: str) -> str:
    """Turn a topic into a safe folder-name slug, e.g. 'AI Regulation!' -> 'ai-regulation'."""
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")
    return slug[:MAX_SLUG_LENGTH].rstrip("-") or "topic"


def run_folder(result: RunResult, output_dir: str | Path) -> Path:
    """Create and return output/<topic-slug>_<YYYY-MM-DDTHH-MM-SSZ>/."""
    stamp = result.generated_at.replace(":", "-")
    folder = Path(output_dir) / f"{slugify(result.topic)}_{stamp}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def write_json(result: RunResult, path: Path) -> Path:
    """Write the full RunResult as indented UTF-8 JSON."""
    path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def write_csv(result: RunResult, path: Path) -> Path:
    """Write one CSV row per article; missing_fields are joined with ';'."""
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for article in result.articles:
            row = article.model_dump(mode="json")
            row["missing_fields"] = ";".join(article.missing_fields)
            writer.writerow({k: "" if v is None else v for k, v in row.items()})
    return path


def _article_warning(article: Article) -> str | None:
    """Return a Markdown warning line for excerpt-only or incomplete articles."""
    notes: list[str] = []
    if article.text_type == "feed_excerpt":
        notes.append("summary is based on a short feed excerpt, not the full article")
    elif article.text_type == "none":
        notes.append("no article text could be retrieved")
    if article.missing_fields:
        notes.append("missing fields: " + ", ".join(article.missing_fields))
    return f"> ⚠️ {'; '.join(notes).capitalize()}." if notes else None


def _article_markdown(index: int, article: Article) -> list[str]:
    """Render one article as a block of Markdown lines."""
    date = article.published_at[:10] if article.published_at else "date unknown"
    publisher = article.source or "Unknown publisher"
    lines = [f"### {index}. {article.title or 'Untitled'}", "", f"*{publisher} · {date}*", ""]
    if article.summary:
        lines += [article.summary, ""]
    else:
        lines += ["*No summary available.*", ""]
    if article.url:
        lines += [f"[Read the original]({article.url})", ""]
    warning = _article_warning(article)
    if warning:
        lines += [warning, ""]
    return lines


def build_digest(result: RunResult) -> str:
    """Build the human-readable Markdown digest for a run."""
    lines = [
        f"# News digest: {result.topic}",
        "",
        f"*Generated {result.generated_at} · last {result.days} days · "
        f"{result.article_count} of {result.requested_limit} requested articles*",
        "",
        f"**Sources used:** {', '.join(result.sources_queried) or 'none'}",
        "",
    ]
    if result.shortfall_note:
        lines += [f"> **Note:** {result.shortfall_note}", ""]
    if result.source_errors:
        lines += ["**Source errors:**", ""]
        lines += [f"- `{e.source}`: {e.error}" for e in result.source_errors]
        lines.append("")
    lines += ["---", ""]
    if not result.articles:
        lines += ["*No articles were found for this topic.*", ""]
    for index, article in enumerate(result.articles, start=1):
        lines += _article_markdown(index, article)
    return "\n".join(lines).rstrip() + "\n"


def write_digest(result: RunResult, path: Path) -> Path:
    """Write the Markdown digest."""
    path.write_text(build_digest(result), encoding="utf-8")
    return path


def export_results(result: RunResult, output_dir: str | Path = "output", fmt: str = "both") -> list[Path]:
    """Write JSON and/or CSV (per `fmt`) plus digest.md; return the written paths."""
    folder = run_folder(result, output_dir)
    written: list[Path] = []
    if fmt in ("json", "both"):
        written.append(write_json(result, folder / "articles.json"))
    if fmt in ("csv", "both"):
        written.append(write_csv(result, folder / "articles.csv"))
    written.append(write_digest(result, folder / "digest.md"))
    return written
