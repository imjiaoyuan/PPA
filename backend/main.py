import sys
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

current_file_path = os.path.abspath(__file__)
backend_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(backend_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.api import leaf_gender, branch_angle, color_scan, grain_count, spike_size

app = FastAPI(
    title="PPA - Plant Phenotyping Algorithm",
    description="A web platform for plant phenotyping analysis using CV and DL models."
)

app.mount("/static", StaticFiles(directory=os.path.join(project_root, "frontend/static")), name="static")
templates = Jinja2Templates(directory=os.path.join(project_root, "frontend/templates"))

app.include_router(leaf_gender.router)
app.include_router(branch_angle.router)
app.include_router(color_scan.router)
app.include_router(grain_count.router)
app.include_router(spike_size.router)

@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/leaf-gender", response_class=HTMLResponse)
async def serve_leaf_gender_page(request: Request):
    return templates.TemplateResponse("leaf_gender.html", {"request": request})

@app.get("/branch-angle", response_class=HTMLResponse)
async def serve_branch_angle_page(request: Request):
    return templates.TemplateResponse("branch_angle.html", {"request": request})

@app.get("/color-scan", response_class=HTMLResponse)
async def serve_color_scan_page(request: Request):
    return templates.TemplateResponse("color_scan.html", {"request": request})

@app.get("/grain-count", response_class=HTMLResponse)
async def serve_grain_count_page(request: Request):
    return templates.TemplateResponse("grain_count.html", {"request": request})

@app.get("/spike-size", response_class=HTMLResponse)
async def serve_spike_size_page(request: Request):
    return templates.TemplateResponse("spike_size.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
async def serve_about_page(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)