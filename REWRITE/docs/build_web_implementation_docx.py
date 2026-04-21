from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

from build_project_report_docx import DocxBuilder, split_paragraphs, write_docx


def clean_inline_markdown(text: str) -> str:
    text = text.strip()
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    return text.strip()


def add_markdown_block(doc: DocxBuilder, lines: list[str]) -> None:
    if not lines:
        return

    stripped = [line.rstrip() for line in lines]

    if all(line.strip().startswith("- ") for line in stripped if line.strip()):
        for line in stripped:
            item = clean_inline_markdown(line.strip()[2:])
            if item:
                doc.add_paragraph(f"- {item}")
        return

    paragraph = "\n".join(stripped).strip()
    if not paragraph:
        return

    for part in split_paragraphs(paragraph):
        doc.add_paragraph(clean_inline_markdown(part))


def build_document_from_markdown(md_path: Path) -> DocxBuilder:
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    doc = DocxBuilder()

    today = dt.datetime.now().strftime("%d %B %Y")
    doc.add_title("Website Implementation Thought Process and Difficulties")
    doc.add_subtitle("Supporting document for the PHP/XAMPP forecasting website build")
    doc.add_subtitle(f"Prepared on {today}")
    doc.add_subtitle("Converted from the implementation notes into a standalone project document")
    doc.add_page_break()

    current_block: list[str] = []
    first_heading_seen = False

    for raw_line in lines:
        line = raw_line.rstrip()

        if line.startswith("# "):
            add_markdown_block(doc, current_block)
            current_block = []
            if not first_heading_seen:
                first_heading_seen = True
                doc.add_heading(clean_inline_markdown(line[2:]), level=1)
            else:
                doc.add_heading(clean_inline_markdown(line[2:]), level=1)
            continue

        if line.startswith("## "):
            add_markdown_block(doc, current_block)
            current_block = []
            doc.add_heading(clean_inline_markdown(line[3:]), level=1)
            continue

        if line.startswith("### "):
            add_markdown_block(doc, current_block)
            current_block = []
            doc.add_heading(clean_inline_markdown(line[4:]), level=2)
            continue

        if line.strip() == "":
            add_markdown_block(doc, current_block)
            current_block = []
            continue

        current_block.append(line)

    add_markdown_block(doc, current_block)
    return doc


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    md_path = repo_root / "web" / "IMPLEMENTATION_THOUGHT_PROCESS.md"
    output_path = repo_root / "web" / "WEBSITE_IMPLEMENTATION_THOUGHT_PROCESS.docx"

    builder = build_document_from_markdown(md_path)
    final_path = output_path

    try:
        write_docx(output_path, builder)
    except PermissionError:
        fallback = repo_root / "web" / f"WEBSITE_IMPLEMENTATION_THOUGHT_PROCESS_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
        write_docx(fallback, builder)
        final_path = fallback

    print(f"Wrote: {final_path}")
    print(f"Body words: {builder.body_word_count}")
    print(f"Estimated pages: {builder.estimate_pages()}")


if __name__ == "__main__":
    main()
