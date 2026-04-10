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
        from PIL import Image as PILImage

        doc = fitz.open(stream=file_bytes, filetype="pdf")

        best_image = None
        best_score = 0
        best_page = 0

        # Strategy: render each page as an image, score by:
        # - Landscape orientation (site plans are wide)
        # - Color variety (not mostly white/text)
        # - Not on first 2 pages (usually cover/logo)
        # - Not on last 2 pages (usually signature/back cover)
        max_pages = min(doc.page_count, 30)
        pages_to_check = list(range(max_pages))

        # Prioritize middle pages (site plans usually in middle of presentations)
        # Skip first 2 and last 2 pages from priority
        for page_num in pages_to_check:
            try:
                page = doc[page_num]
                rect = page.rect
                page_width = rect.width
                page_height = rect.height

                # Render at lower DPI first for scoring (fast)
                mat = fitz.Matrix(1.5, 1.5)
                pix = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")
                pil_img = PILImage.open(io.BytesIO(img_bytes))

                # Score this page
                score = 0

                # Landscape orientation bonus (site plans are landscape)
                if page_width > page_height:
                    score += 1000
                else:
                    score += 100

                # Skip very first and very last pages (usually cover/signature)
                if page_num == 0 or page_num >= max_pages - 1:
                    score -= 500

                # Color variety check — site plans have lots of colors
                # Convert to small thumbnail for fast analysis
                thumb = pil_img.copy()
                thumb.thumbnail((100, 100))
                if thumb.mode != "RGB":
                    thumb = thumb.convert("RGB")
                colors = thumb.getcolors(maxcolors=10000)
                if colors:
                    unique_colors = len(colors)
                    score += min(unique_colors * 2, 2000)

                    # Check if page is mostly white (low content = logo/text page)
                    white_pixels = sum(count for count, color in colors if sum(color) > 700)
                    total_pixels = sum(count for count, color in colors)
                    if total_pixels > 0:
                        white_ratio = white_pixels / total_pixels
                        if white_ratio > 0.8:  # >80% white = probably not site plan
                            score -= 1500

                # Size of image itself — prefer larger (more detail)
                score += min(len(img_bytes) // 10000, 500)

                if score > best_score:
                    best_score = score
                    best_image = img_bytes
                    best_page = page_num

            except Exception as e:
                continue

        doc.close()

        if best_image is None:
            # Fallback: render the first page
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            page = doc[req.page]
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            best_image = pix.tobytes("png")
            best_page = req.page
            doc.close()

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
