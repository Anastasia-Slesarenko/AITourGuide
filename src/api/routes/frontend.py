# src/api/routes/frontend.py
"""Фронтенд-роуты: рендер шаблонов и обработка формы загрузки."""

import logging
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse

from src.api.dependencies import get_guide

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Frontend"])


def _build_sorted_candidates(result: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Строит список кандидатов (без победителя), отсортированных
    по p_yes убыванию.

    Победитель — индекс 0 в retrieved_* списках — уже показан в
    основном блоке результата. Здесь собираем остальных (индексы
    1…N) и сортируем по retrieved_p_yes убыванию, чтобы в блоке
    «Другие кандидаты из базы» наиболее уверенные шли первыми.

    Returns:
        Список словарей с ключами: image, name, p_yes (float 0–1).
    """
    images = result.get("retrieved_images") or []
    names = result.get("retrieved_names") or []
    p_yes_list = result.get("retrieved_p_yes") or []

    logger.info(
        f"_build_sorted_candidates: images={len(images)}, p_yes_list={p_yes_list}"
    )

    candidates = []
    for i in range(1, len(images)):
        p_yes = (
            float(p_yes_list[i])
            if i < len(p_yes_list) and p_yes_list[i] is not None
            else 0.0
        )
        candidates.append(
            {
                "image": images[i],
                "name": names[i] if i < len(names) else "",
                "p_yes": p_yes,
            }
        )

    candidates.sort(key=lambda c: c["p_yes"], reverse=True)
    logger.info(
        f"sorted_candidates order: "
        f"{[(c['name'], round(c['p_yes'], 3)) for c in candidates]}"
    )
    return candidates


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Главная страница с формой загрузки изображения."""
    return request.app.state.templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"use_internet": True, "sorted_candidates": []},
    )


@router.post("/predict", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    image: UploadFile | None = File(default=None),
    use_internet: bool = Form(default=False),
):
    """
    Обрабатывает форму: принимает изображение, вызывает AITourGuide
    и возвращает страницу с результатом или ошибкой.
    """
    result = None
    error = None
    sorted_candidates: list[dict[str, Any]] = []

    if not image:
        error = "Пожалуйста, загрузите изображение"
    elif not (image.content_type or "").startswith("image/"):
        error = "Поддерживаются только изображения (JPEG, PNG, WebP)"
    elif image.size and image.size > 10 * 1024 * 1024:
        error = "Размер файла не должен превышать 10 МБ"
    else:
        try:
            guide = await get_guide()
        except HTTPException as e:
            # Сервис не инициализирован — показываем понятное сообщение
            error = f"Сервис недоступен: {e.detail}"
            guide = None

        if guide is not None:
            try:
                image_bytes = await image.read()
                result = await guide.predict(
                    image_input=image_bytes,
                    use_internet_search=use_internet,
                )
                if result.get("error"):
                    error = result["error"]
                else:
                    sorted_candidates = _build_sorted_candidates(result)
            except Exception as e:
                logger.error(f"Ошибка обработки изображения: {e}")
                error = "Внутренняя ошибка сервера"

    return request.app.state.templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "result": result,
            "error": error,
            "use_internet": use_internet,
            "sorted_candidates": sorted_candidates,
        },
    )
