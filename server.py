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

    if len(html_content) > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(status_code=413, detail="HTML body too large (max 50MB)")

    try:
        import weasyprint
        html = weasyprint.HTML(string=html_content.decode("utf-8"))
        pdf_bytes = html.write_pdf()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Render failed: {str(e)}")

    return Response(content=pdf_bytes, media_type="application/pdf")


class ExtractImageRequest(BaseModel):
    url: str
    token: str
    filename: str
    page: Optional[int] = 0  # Which page to extract (0 = first/cover)
    min_width: Optional[int] = 400  # Minimum image width to consider as site plan
    min_height: Optional[int] = 300


@app.post("/extract-image")
async def extract_image(req: ExtractImageRequest, request: Request):
    """Download a PDF and extract the largest image (site plan/render) as base64 PNG."""
    verify_api_key(request)

    try:
        # Download file
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.get(
                req.url,
                headers={"Authorization": f"Bearer {req.token}"},
                follow_redirects=True,
            )
            resp.raise_for_status()
            file_bytes = resp.content

        ext = req.filename.rsplit(".", 1)[-1].lower() if "." in req.filename else ""

        if ext != "pdf":
            raise HTTPException(status_code=400, detail="Only PDF files supported for image extraction")

        import fitz  # PyMuPDF
        import base64
        import gc
        from PIL import Image as PILImage

        doc = fitz.open(stream=file_bytes, filetype="pdf")

        # Memory-efficient strategy:
        # 1. First pass: tiny thumbnails (72 DPI) to score pages by content
        # 2. Second pass: render best-scoring page at higher quality
        max_pages = min(doc.page_count, 15)  # Check max 15 pages
        best_score = -999999
        best_page_num = req.page

        for page_num in range(max_pages):
            try:
                page = doc[page_num]
                rect = page.rect

                # Low DPI thumbnail for scoring (100px wide max)
                scale = min(100 / rect.width, 100 / rect.height)
                mat = fitz.Matrix(scale, scale)
                pix = page.get_pixmap(matrix=mat, alpha=False)

                score = 0

                # Landscape bonus
                if rect.width > rect.height:
                    score += 1000

                # Skip first 2 pages (cover/logo) and last page (back/signature)
                if page_num < 2:
                    score -= 2000
                if page_num >= max_pages - 1:
                    score -= 500

                # Convert to PIL and check color variety
                thumb_bytes = pix.tobytes("ppm")
                pil_thumb = PILImage.open(io.BytesIO(thumb_bytes))
                if pil_thumb.mode != "RGB":
                    pil_thumb = pil_thumb.convert("RGB")

                colors = pil_thumb.getcolors(maxcolors=5000)
                if colors:
                    unique_colors = len(colors)
                    score += min(unique_colors * 2, 2000)

                    total = sum(c for c, _ in colors)
                    white = sum(c for c, color in colors if sum(color) > 700)
                    if total > 0:
                        white_ratio = white / total
                        if white_ratio > 0.85:
                            score -= 2000

                if score > best_score:
                    best_score = score
                    best_page_num = page_num

                # Release memory
                pix = None
                pil_thumb = None
                thumb_bytes = None
                gc.collect()

            except Exception:
                continue

        # Now render the best page at good quality (not too high — 1.5x)
        try:
            page = doc[best_page_num]
            mat = fitz.Matrix(1.5, 1.5)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            best_image = pix.tobytes("png")
            best_page = best_page_num
            pix = None
        except Exception:
            # Ultimate fallback — render page 0 at low quality
            page = doc[0]
            mat = fitz.Matrix(1, 1)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            best_image = pix.tobytes("png")
            best_page = 0

        doc.close()
        gc.collect()

        # Resize if too large (max 1200px wide) to keep HTML under size limits
        from PIL import Image
        img = Image.open(io.BytesIO(best_image))
        max_width = 1200
        if img.width > max_width:
            ratio = max_width / img.width
            new_size = (max_width, int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        # Save as JPEG for smaller size (PNG site plans can be 5MB+)
        buf = io.BytesIO()
        if img.mode == "RGBA":
            img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=85)
        optimized = buf.getvalue()

        b64 = base64.b64encode(optimized).decode("utf-8")

        return {
            "filename": req.filename,
            "page": best_page,
            "image_size": len(optimized),
            "base64": b64,
            "format": "jpeg",
        }

    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Download failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image extraction failed: {str(e)}")


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
