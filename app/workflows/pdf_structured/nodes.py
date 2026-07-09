from __future__ import annotations

import asyncio
from io import BytesIO
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pypdf import PdfReader, PdfWriter
from pydantic import BaseModel

from app.application.core.model_factory import get_chat_model
from app.application.infra.clients.paddle_client import paddle_extract_pdf_text
from app.workflows.pdf_structured.state import PdfStructuredState


TITLE_RE = re.compile(
    r"(?:^|\n)\s*##\s*"
    r"([一二三四五六七八九十]+|[（(]?[一二三四五六七八九十]+[）)])"
    r"[、.\s]+([^\n]{2,30})"
)
def _parse_page_indexes(page_range: str, page_count: int) -> list[int]:
    pages: set[int] = set()
    for part in re.split(r"[，,]", page_range):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                start, end = map(int, part.split("-", 1))
                for page in range(start, end + 1):
                    if 1 <= page <= page_count:
                        pages.add(page - 1)
            else:
                page = int(part)
                if 1 <= page <= page_count:
                    pages.add(page - 1)
        except ValueError:
            continue
    return sorted(pages)


def _crop_pdf_pages(pdf_path: str, page_range: str) -> str:
    with open(pdf_path, "rb") as f:
        original_data = f.read()

    reader = PdfReader(BytesIO(original_data))
    page_count = len(reader.pages)
    if page_count <= 0:
        return pdf_path

    page_indexes = _parse_page_indexes(page_range=page_range, page_count=page_count)
    if not page_indexes:
        return pdf_path

    writer = PdfWriter()
    for index in page_indexes:
        writer.add_page(reader.pages[index])

    output = BytesIO()
    writer.write(output)
    with open(pdf_path, "wb") as f:
        f.write(output.getvalue())
    return pdf_path


def _text_filter(text: str) -> str:
    text = re.sub(r"<img[^>]*?>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?div[^>]*?>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _normalize_title(title: str) -> str:
    title = re.sub(r"^[一二三四五六七八九十\d]+[、\.]", "", title)
    title = re.sub(r"^[（(][一二三四五六七八九十]+[）)]", "", title)
    return title.strip()


def _split_by_titles(text: str) -> list[dict[str, str]]:
    sections: list[dict[str, str]] = []
    matches = list(TITLE_RE.finditer(text))
    if not matches:
        return sections

    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        prefix = match.group(1)
        title_text = match.group(2).strip()
        content = text[start:end].strip()
        sections.append({"title": f"{prefix} {title_text}", "content": content})

    return sections


def _text_preprocess(full_text: str, target_sections: list[str] | None = None) -> str:
    if not full_text:
        return ""

    full_text = _text_filter(full_text)
    if target_sections is None:
        return full_text

    parts = full_text.split("\f", 1)
    first_page = parts[0].strip()
    rest_text = parts[1].strip() if len(parts) > 1 else ""
    kept_blocks = [first_page]

    if not rest_text or not target_sections:
        return "\n\n".join(kept_blocks)

    sections = _split_by_titles(rest_text)
    for section in sections:
        title_norm = _normalize_title(section["title"])
        for target in target_sections:
            if target in title_norm:
                kept_blocks.append(f"## {section['title']}\n{section['content'].strip()}")
                break

    return "\n\n".join(kept_blocks)


def _is_mostly_english(text: str, threshold: float = 0.9) -> bool:
    english_count = len(re.findall(r"[A-Za-z]", text))
    chinese_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    alpha_total = english_count + chinese_count
    if alpha_total < 50:
        return False
    return (english_count / alpha_total) >= threshold


async def pdf_preprocess_node(state: PdfStructuredState) -> dict[str, str]:
    pdf_process = state.get("pdf_process")
    page_range = pdf_process.get("page_range") if isinstance(pdf_process, dict) else None
    if not isinstance(page_range, str) or not page_range.strip():
        return {"pdf_path": state["pdf_path"]}

    processed_pdf_path = await asyncio.to_thread(_crop_pdf_pages, state["pdf_path"], page_range)
    return {"pdf_path": processed_pdf_path}


async def text_preprocess_node(state: PdfStructuredState) -> dict[str, str]:
    extracted_text = state.get("extracted_text", "")
    text_process = state.get("text_process")

    safe_target_sections: list[str] | None = None
    if isinstance(text_process, dict):
        raw_target_sections = text_process.get("target_sections")
        if isinstance(raw_target_sections, str) and raw_target_sections.strip():
            safe_target_sections = [raw_target_sections.strip()]
        elif isinstance(raw_target_sections, list):
            safe_target_sections = [
                item.strip() for item in raw_target_sections if isinstance(item, str) and item.strip()
            ]
            if not safe_target_sections:
                safe_target_sections = None

    return {
        "extracted_text": _text_preprocess(
            full_text=extracted_text,
            target_sections=safe_target_sections,
        )
    }


async def extract_pdf_text_node(state: PdfStructuredState) -> dict[str, Any]:
    pipeline = state.get("paddle_pipeline")
    text = await paddle_extract_pdf_text(state["pdf_path"], pipeline=pipeline)

    already_retried = bool(state.get("is_en_retry", False))
    should_retry = (not already_retried) and _is_mostly_english(text)

    if should_retry:
        return {
            "extracted_text": text,
            "retry_with_rec_en": True,
            "is_en_retry": True,
            "paddle_pipeline": "rec_en",
        }

    return {
        "extracted_text": text,
        "retry_with_rec_en": False,
    }


async def structured_output_node(state: PdfStructuredState) -> dict[str, dict[str, Any]]:
    schema_model = state["schema_model"]
    system_prompt = state.get(
        "system_prompt",
        "You are an information extraction assistant. Return only schema-conforming JSON.",
    )
    extracted_text = state.get("extracted_text", "")

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Please extract structured data from this text:\n\n{extracted_text}"),
    ]

    model = get_chat_model().with_structured_output(schema_model)
    result = await model.ainvoke(messages)

    if isinstance(result, BaseModel):
        payload = result.model_dump()
    elif isinstance(result, dict):
        payload = result
    else:
        payload = {"value": str(result)}

    return {"structured_output": payload}
