"""
WeasyPrint PDF Rendering Microservice
Stateless: HTML in -> PDF out. No file persistence.
"""
import os
import time

from fastapi import FastAPI, Request, Response, HTTPException

app = FastAPI(title="WeasyPrint Renderer", docs_url=None, redoc_url=None)

API_KEY = os.environ.get("WEASYPRINT_API_KEY", "")
START_TIME = time.time()


def verify_api_key(request: Request):
    if not API_KEY:
        return  # No key configured (dev mode)
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
