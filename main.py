from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.classifiers import router
from schemas.models import HealthResponse

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router=router, prefix="/classifier")


@app.get("/", response_model=HealthResponse)
async def health():
    return HealthResponse(status="Ok")
