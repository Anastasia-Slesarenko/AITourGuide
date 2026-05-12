# src/api/routes/frontend_router.py
"""
Frontend routes: serves Jinja2 templates and handles form submissions.
"""

import logging
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.api.dependencies import get_guide

logger = logging.getLogger(__name__)

router = APIRouter(tags=["frontend"])


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Рендерит главную страницу с формой загрузки."""
    return request.app.state.templates.TemplateResponse(
        "index.html", 
        {"request": request}
    )


@router.post("/predict", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    image: Optional[UploadFile] = File(default=None),
    use_internet: bool = Form(default=False)
):
    """
    Обработка формы: принимает изображение, вызывает AITourGuide,
    и возвращает ту же страницу с результатом.
    """
    result = None
    error = None
    
    if not image:
        error = "Пожалуйста, загрузите изображение"
    elif not image.content_type.startswith("image/"):
        error = "Поддерживаются только изображения (JPEG, PNG, WebP)"
    elif image.size and image.size > 10 * 1024 * 1024:
        error = "Размер файла не должен превышать 10 МБ"
    else:
        try:
            guide = await get_guide()
            if not guide:
                raise RuntimeError("AITourGuide не инициализирован")
            
            image_bytes = await image.read()
            result = await guide.predict(
                image_input=image_bytes,
                use_internet_search=use_internet
            )
            
            if result.get("error"):
                error = result["error"]
                
        except Exception as e:
            logger.error(f"Ошибка обработки: {e}")
            error = f"Внутренняя ошибка: {str(e)}"
    
    # Возвращаем страницу с результатом или ошибкой
    return request.app.state.templates.TemplateResponse(
        "index.html",
        {"request": request, "result": result, "error": error, "use_internet": use_internet}
    )