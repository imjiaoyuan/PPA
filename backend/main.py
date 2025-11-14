from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from api import leaf_gender

app = FastAPI(title="PPA - Plant Phenotyping Algorithm")

app.mount("/static", StaticFiles(directory="../frontend/static"), name="static")
templates = Jinja2Templates(directory="../frontend/templates")

app.include_router(leaf_gender.router)

@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/leaf-gender", response_class=HTMLResponse)
async def serve_leaf_gender_page(request: Request):
    return templates.TemplateResponse("leaf_gender.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)