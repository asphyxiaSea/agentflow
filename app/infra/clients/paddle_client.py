from __future__ import annotations

from pathlib import Path

import httpx

from app.core.settings import PADDLE_EXTRACT_PATH_URL


async def paddle_extract_pdf_text(
    pdf_path: str,
    timeout: float = 120.0,
    pipeline: str | None = None,
) -> str:
    """Extract markdown text from a local PDF path via external Paddle service."""

    path = Path(pdf_path)
    if not path.exists() or not path.is_file():
        raise ValueError(f"PDF file does not exist: {pdf_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Only PDF files are supported: {pdf_path}")

    params: dict[str, str] = {"path": str(path)}
    if pipeline:
        params["pipeline"] = pipeline

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                PADDLE_EXTRACT_PATH_URL,
                params=params,
            )
            response.raise_for_status()
            data = response.json()
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        raise RuntimeError(f"Paddle service error: {exc}") from exc

    text = data.get("markdown_text", "")
    if not isinstance(text, str):
        raise RuntimeError("Paddle response markdown_text is invalid")

    return text
