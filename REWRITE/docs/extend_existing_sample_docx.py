from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
ET.register_namespace("w", W_NS)
ET.register_namespace("r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships")
ET.register_namespace("xml", "http://www.w3.org/XML/1998/namespace")


def read_zip_text(path: Path, member: str) -> str:
    with zipfile.ZipFile(path) as zf:
        return zf.read(member).decode("utf-8")


def extract_style_block(styles_xml: str, style_id: str) -> str:
    pattern = rf'(<w:style\b[^>]*w:styleId="{re.escape(style_id)}"[\s\S]*?</w:style>)'
    match = re.search(pattern, styles_xml)
    if not match:
        raise ValueError(f"Could not find style {style_id}")
    return match.group(1)


def merge_missing_styles(target_styles_xml: str, source_styles_xml: str) -> str:
    additions: list[str] = []
    for style_id in ("BodyText", "CodeBlock"):
        if f'w:styleId="{style_id}"' not in target_styles_xml:
            additions.append(extract_style_block(source_styles_xml, style_id))
    if not additions:
        return target_styles_xml
    insert_at = target_styles_xml.rfind("</w:styles>")
    if insert_at == -1:
        raise ValueError("Invalid target styles.xml: missing </w:styles>")
    return target_styles_xml[:insert_at] + "".join(additions) + target_styles_xml[insert_at:]


def paragraph_xml(text: str, style_id: str = "BodyText") -> str:
    escaped = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return (
        f'<w:p><w:pPr><w:pStyle w:val="{style_id}"/></w:pPr>'
        f"<w:r><w:t>{escaped}</w:t></w:r></w:p>"
    )


def page_break_xml() -> str:
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


def body_children_from_heading(doc_xml: str, heading_text: str) -> list[str]:
    root = ET.fromstring(doc_xml)
    body = root.find(f"{{{W_NS}}}body")
    if body is None:
        raise ValueError("Document body not found")

    children = list(body)
    start_idx = None
    for idx, child in enumerate(children):
        texts = [t.text for t in child.findall(f".//{{{W_NS}}}t") if t.text]
        full_text = "".join(texts).strip()
        if full_text == heading_text:
            start_idx = idx
            break
    if start_idx is None:
        raise ValueError(f"Heading not found in source doc: {heading_text}")

    selected = []
    for child in children[start_idx:]:
        if child.tag == f"{{{W_NS}}}sectPr":
            break
        selected.append(ET.tostring(child, encoding="unicode"))
    return selected


def append_before_sectpr(target_doc_xml: str, fragment_xml: str) -> str:
    insert_at = target_doc_xml.rfind("<w:sectPr")
    if insert_at == -1:
        raise ValueError("Invalid document.xml: missing <w:sectPr")
    return target_doc_xml[:insert_at] + fragment_xml + target_doc_xml[insert_at:]


def build_addendum_intro() -> str:
    parts = [
        page_break_xml(),
        paragraph_xml("Project Guide Addendum and Completion", "Heading1"),
        paragraph_xml(
            "The following pages have been added to extend the existing sample document without removing or altering the material, photos, headings or layout that were already present. This addendum is designed to finish the document off against the project guide by adding deeper explanation, implementation detail, code evidence, testing discussion and evaluation while leaving the original content intact.",
            "BodyText",
        ),
        paragraph_xml(
            "Because the original document already contained its own project identification, examples, library discussion and images, those sections have been preserved exactly as part of the base file. The additional sections below act as an expansion layer rather than a replacement, which means the document keeps its original structure and media while comfortably exceeding the minimum page requirement.",
            "BodyText",
        ),
        page_break_xml(),
    ]
    return "".join(parts)


def extend_docx(base_docx: Path, source_addendum_docx: Path, output_docx: Path) -> None:
    target_doc_xml = read_zip_text(base_docx, "word/document.xml")
    target_styles_xml = read_zip_text(base_docx, "word/styles.xml")
    source_doc_xml = read_zip_text(source_addendum_docx, "word/document.xml")
    source_styles_xml = read_zip_text(source_addendum_docx, "word/styles.xml")

    merged_styles_xml = merge_missing_styles(target_styles_xml, source_styles_xml)
    selected_children = body_children_from_heading(source_doc_xml, "Computational Methods")
    addendum_fragment = build_addendum_intro() + "".join(selected_children)
    merged_doc_xml = append_before_sectpr(target_doc_xml, addendum_fragment)

    output_docx.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(base_docx) as src, zipfile.ZipFile(output_docx, "w", zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename == "word/document.xml":
                data = merged_doc_xml.encode("utf-8")
            elif info.filename == "word/styles.xml":
                data = merged_styles_xml.encode("utf-8")
            dst.writestr(info, data)


def main() -> None:
    docs_dir = Path(__file__).resolve().parent
    base_docx = Path(r"c:\Users\paron\Downloads\CS project - Copy.docx")
    source_addendum_docx = docs_dir / "CS_PROJECT_SAMPLE_CONTINUED_REPORT.docx"
    output_docx = docs_dir / "CS_project_Copy_preserved_extended.docx"

    extend_docx(base_docx, source_addendum_docx, output_docx)

    source_doc_xml = read_zip_text(source_addendum_docx, "word/document.xml")
    appended_children = body_children_from_heading(source_doc_xml, "Computational Methods")
    print(f"Wrote: {output_docx}")
    print(f"Appended body nodes: {len(appended_children)}")
    print("Source base preserved: yes")


if __name__ == "__main__":
    main()
