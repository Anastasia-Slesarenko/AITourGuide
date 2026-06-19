# src/api/routes/frontend.py
"""Фронтенд-роуты: рендер шаблонов и обработка формы загрузки."""

import logging
from typing import Optional
from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse

from src.api.dependencies import get_guide

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Frontend"])


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Главная страница с формой загрузки изображения."""
    return request.app.state.templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"use_internet": True},
    )


@router.post("/predict", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    image: Optional[UploadFile] = File(default=None),
    use_internet: bool = Form(default=False),
):
    """
    Обрабатывает форму: принимает изображение, вызывает AITourGuide
    и возвращает страницу с результатом или ошибкой.
    """
    result = None
    error = None

    if not image:
        error = "Пожалуйста, загрузите изображение"
    elif not (image.content_type or "").startswith("image/"):
        error = "Поддерживаются только изображения (JPEG, PNG, WebP)"
    elif image.size and image.size > 10 * 1024 * 1024:
        error = "Размер файла не должен превышать 10 МБ"
    else:
        try:
            guide = await get_guide()
            image_bytes = await image.read()
            result = await guide.predict(
                image_input=image_bytes,
                use_internet_search=use_internet,
            )
            if result.get("error"):
                error = result["error"]
        except Exception as e:
            logger.error(f"Ошибка обработки изображения: {e}")
            error = f"Внутренняя ошибка: {str(e)}"

    return request.app.state.templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "result": result,
            "error": error,
            "use_internet": use_internet,
        },
    )
