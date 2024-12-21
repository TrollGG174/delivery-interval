from fastapi import FastAPI, Request, Query
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from utils import intervals

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешить все источники.
    allow_credentials=True,
    allow_methods=["*"],  # Разрешить все методы (GET, POST, и т.д.)
    allow_headers=["*"],  # Разрешить все заголовки
)

@app.get("/")
async def read_root(request: Request):
    return RedirectResponse(f"{request.url}docs")

@app.get("/intervals/")
async def return_intervals(
    distance: int = Query(...),
    traffic: str = Query(...),
    time: str = Query(...)
):
    return intervals(distance=distance, traffic=traffic, time=time)