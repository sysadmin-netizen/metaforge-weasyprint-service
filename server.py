"""
MetaForge Document Service
- PDF rendering (WeasyPrint)
- Large file download + text extraction proxy (for files too large for Edge Functions)
"""
import os
import time
import io
import httpx

from fastapi import FastAPI, Request, Response, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="MetaForge Document Service", docs_url=None, redoc_url=None)

API_KEY = os.environ.get("WEASYPRINT_API_KEY", "")
START_TIME = time.time()


def verify_api_key(request: Request):
    if not API_KEY:
        return
    key = request.headers.get("X-API-Key", "")
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "1.0.0",
        "uptime_seconds": round(time.time() - START_TIME),
    }


@app.post("/render")
async def render(request: Request):
    verify_api_key(request)

    html_content = await request.body()
    if not html_content:
        raise HTTPException(status_code=400, detail="Empty HTML body")

    if len(html_content) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=413, detail="HTML body too large (max 10MB)")

    try:
        import weasyprint
        html = weasyprint.HTML(string=html_content.decode("utf-8"))
        pdf_bytes = html.write_pdf()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Render failed: {str(e)}")

    return Response(content=pdf_bytes, media_type="application/pdf")


class ExtractRequest(BaseModel):
    url: str
    token: str
    filename: str
    max_text_chars: Optional[int] = 50000


@app.post("/extract-text")
async def extract_text(req: ExtractRequest, request: Request):
    """Download a file from URL and extract text content.
    Works for PDFs of ANY size — no timeout limits on Render."""
    verify_api_key(request)

    try:
        # Download file from Slack (no timeout limit on Render)
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.get(
                req.url,
                headers={"Authorization": f"Bearer {req.token}"},
                follow_redirects=True,
            )
            resp.raise_for_status()
            file_bytes = resp.content

        ext = req.filename.rsplit(".", 1)[-1].lower() if "." in req.filename else ""
        size_mb = len(file_bytes) / 1024 / 1024
        extracted_text = ""

        if ext == "pdf":
            try:
                import fitz  # PyMuPDF — faster and more reliable than pdf-parse
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                pages_text = []
                for page_num in range(min(doc.page_count, 100)):  # Max 100 pages
                    page = doc[page_num]
                    pages_text.append(f"--- Page {page_num + 1} ---\n{page.get_text()}")
                extracted_text = "\n".join(pages_text)
                doc.close()
            except Exception:
                # Fallback to pdfplumber
                try:
                    import pdfplumber
                    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                        pages_text = []
                        for i, page in enumerate(pdf.pages[:100]):
                            text = page.extract_text() or ""
                            pages_text.append(f"--- Page {i + 1} ---\n{text}")
                        extracted_text = "\n".join(pages_text)
                except Exception as e2:
                    extracted_text = f"[PDF text extraction failed: {str(e2)}]"

        elif ext in ("xlsx", "xls", "csv"):
            try:
                import openpyxl
                wb = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True)
                sheets_text = []
                for name in wb.sheetnames:
                    ws = wb[name]
                    sheets_text.append(f"=== Sheet: {name} ===")
                    for row in ws.iter_rows(values_only=True):
                        vals = [str(v) if v is not None else "" for v in row]
                        if any(v.strip() for v in vals):
                            sheets_text.append(" | ".join(vals))
                extracted_text = "\n".join(sheets_text)
            except Exception as e:
                extracted_text = f"[Excel extraction failed: {str(e)}]"

        else:
            extracted_text = f"[Binary file: {req.filename} ({size_mb:.1f}MB, .{ext})]"

        # Truncate if needed
        if len(extracted_text) > req.max_text_chars:
            extracted_text = extracted_text[:req.max_text_chars] + "\n\n[... truncated ...]"

        return {
            "filename": req.filename,
            "size_mb": round(size_mb, 1),
            "extension": ext,
            "text_length": len(extracted_text),
            "text": extracted_text,
        }

    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Download failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
