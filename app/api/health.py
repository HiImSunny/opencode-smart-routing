from fastapi import APIRouter
from app.core.settings import VERSION

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok", "version": VERSION}
