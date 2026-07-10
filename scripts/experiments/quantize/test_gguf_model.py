import os

import json
import time
import math
import sys
import base64
import io
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image


try:
    from llama_cpp import Llama, LlamaGrammar
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("llama-cpp-python не установлен")

# Chat handler для multimodal моделей (mmproj).
# В llama-cpp-python 0.3.22 доступен Qwen25VLChatHandler (для Qwen2-VL и Qwen2.5-VL).
# Порядок приоритета: Qwen25VLChatHandler -> Llava15ChatHandler -> Llava16ChatHandler
_ChatHandlerClass = None
_CHAT_HANDLER_NAME = None
_HANDLER_CANDIDATES = [
    ("llama_cpp.llama_chat_format", "Qwen25VLChatHandler"),
    ("llama_cpp.llama_chat_format", "Llava16ChatHandler"),
    ("llama_cpp.llama_chat_format", "Llava15ChatHandler"),
]
for _mod, _cls in _HANDLER_CANDIDATES:
    try:
        import importlib as _il
        _m = _il.import_module(_mod)
        _ChatHandlerClass = getattr(_m, _cls)
        _CHAT_HANDLER_NAME = _cls
        print(f"Chat handler: {_CHAT_HANDLER_NAME}")
        break
    except (ImportError, AttributeError):
        continue
if _ChatHandlerClass is None:
    print("Ни один VLM chat handler не найден — изображения НЕ будут обработаны!")


def _patch_chat_handler_no_gpu(handler_instance) -> None:
    """
    Monkey-patch _init_mtmd_context чтобы отключить GPU (use_gpu=False).

    В llama-cpp-python 0.3.22 Llava15ChatHandler._init_mtmd_context
    жёстко устанавливает ctx_params.use_gpu = True ПОСЛЕ вызова
    mtmd_context_params_default(), поэтому патч params_default не помогает.
    Патч заменяет весь метод целиком с use_gpu=False.
    """
    import types
    import llama_cpp
    from contextlib import suppress

    # Импортируем suppress_stdout_stderr из llama_cpp
    try:
        from llama_cpp.llama_chat_format import suppress_stdout_stderr
    except ImportError:
        from contextlib import nullcontext as suppress_stdout_stderr

    def _patched_init_mtmd_context(self_h, llama_model):
        """Полная замена _init_mtmd_context с use_gpu=False."""
        if self_h.mtmd_ctx is not None:
            return  # Already initialized

        _mtmd = self_h._mtmd_cpp

        try:
            ctx_params = _mtmd.mtmd_context_params_default()
        except Exception:
            return

        # ПАТЧ: use_gpu=False вместо True
        ctx_params.use_gpu = False
        ctx_params.print_timings = self_h.verbose
        ctx_params.n_threads = llama_model.n_threads
        ctx_params.flash_attn_type = (
            llama_cpp.LLAMA_FLASH_ATTN_TYPE_ENABLED
            if (
                llama_model.context_params.flash_attn_type
                == llama_cpp.LLAMA_FLASH_ATTN_TYPE_ENABLED
            )
            else llama_cpp.LLAMA_FLASH_ATTN_TYPE_DISABLED
        )

        self_h.mtmd_ctx = _mtmd.mtmd_init_from_file(
            self_h.clip_model_path.encode(), llama_model.model, ctx_params
        )

        if self_h.mtmd_ctx is None:
            raise ValueError(
                f"Failed to load mtmd context from: {self_h.clip_model_path}"
            )

        if not _mtmd.mtmd_support_vision(self_h.mtmd_ctx):
            raise ValueError("Vision is not supported by this model")

        def mtmd_free():
            if self_h.mtmd_ctx is not None:
                _mtmd.mtmd_free(self_h.mtmd_ctx)
                self_h.mtmd_ctx = None

        self_h._exit_stack.callback(mtmd_free)

    handler_instance._init_mtmd_context = types.MethodType(
        _patched_init_mtmd_context, handler_instance
    )

# Grammar ограничивает генерацию только токенами Yes/No.
# Это гарантирует, что оба токена всегда присутствуют в top_logprobs,
# что делает бинарный softmax корректным.
_YES_NO_GRAMMAR = None  # инициализируется лениво при первом вызове


sys.path.insert(0, str(Path(__file__).parent.parent))
from src.rag.landmark_retriever import LandmarkRetriever



INDEX_PATH = "/Users/anastasiya/Documents/AITourGuide/data/index/siglip"
IMAGE_DIR = "/Users/anastasiya/Documents/AITourGuide/images"
MODEL_BASE = "/Users/anastasiya/Documents/AITourGuide/data/models/qwen2-vl-2b-r16"
VAL_DATASET = "/Users/anastasiya/Documents/AITourGuide/data/processed/dataset_v1/test.json"

# Порог уверенности — синхронизирован с e2e_eval.py и train.py (none_threshold=0.5)
# Если max(P(yes)) < RERANK_THRESHOLD -> результат считается "unknown"
RERANK_THRESHOLD = 0.5

# Параметры retrieval — синхронизированы с e2e_eval.py
# top_k=10 как в e2e_eval.py (retrieval_top_k=10); faiss_k=50 — default в retriever
RETRIEVAL_TOP_K = 3
RETRIEVAL_FAISS_K = 50

# Параметры Qwen2VLImageProcessor (из HF transformers >= 4.45).
# smart_resize масштабирует изображение так чтобы:
#   1. H и W кратны PATCH_SIZE * MERGE_SIZE = 14 * 2 = 28
#   2. MIN_PIXELS <= H*W <= MAX_PIXELS
#   3. Соотношение сторон сохраняется
# При 224×224: H*W=50176 < MIN_PIXELS=200704 -> масштабируется до ~448×448
# Это воспроизводит поведение HF Qwen2VLImageProcessor.
QWEN2VL_PATCH_SIZE: int = 14        # patch_size в vision encoder
QWEN2VL_MERGE_SIZE: int = 2         # merge_size (spatial merge)
QWEN2VL_FACTOR: int = QWEN2VL_PATCH_SIZE * QWEN2VL_MERGE_SIZE  # = 28
QWEN2VL_MIN_PIXELS: int = 256 * 28 * 28   # = 200704 (default min_pixels)
QWEN2VL_MAX_PIXELS: int = 1280 * 28 * 28  # = 1003520 (default max_pixels)

# FIXED_IMAGE_SIZE используется как fallback если smart_resize недоступен
FIXED_IMAGE_SIZE: Tuple[int, int] = (448, 448)


def _smart_resize(
    height: int,
    width: int,
    factor: int = QWEN2VL_FACTOR,
    min_pixels: int = QWEN2VL_MIN_PIXELS,
    max_pixels: int = QWEN2VL_MAX_PIXELS,
) -> Tuple[int, int]:
    """
    Воспроизводит smart_resize из HF Qwen2VLImageProcessor.

    Масштабирует (height, width) так чтобы:
    - H и W кратны factor (=28)
    - min_pixels <= H*W <= max_pixels
    - Соотношение сторон сохраняется

    Источник: transformers/models/qwen2_vl/image_processing_qwen2_vl.py
    """
    if height < factor or width < factor:
        raise ValueError(
            f"height={height} or width={width} < factor={factor}"
        )

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar

# 2. Загрузка GGUF модели
def load_gguf_model(
    model_path: str,
    mmproj_path: str = None,
    n_ctx: int = 4096,
    n_threads: Optional[int] = None,
    n_gpu_layers: int = 0,
    verbose: bool = True,
):
    """Загружает GGUF модель с поддержкой mmproj."""
    if not LLAMA_CPP_AVAILABLE:
        print("llama-cpp-python недоступен")
        return None
    
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Модель не найдена: {model_path}")
        return None
    
    has_mmproj = False
    if mmproj_path:
        mmproj_path_obj = Path(mmproj_path)
        if mmproj_path_obj.exists():
            has_mmproj = True
            mmproj_path = str(mmproj_path_obj)
            if verbose:
                print(f"   ℹmmproj найден: {Path(mmproj_path).name} ({mmproj_path_obj.stat().st_size / 1e6:.1f} MB)")
    
    if verbose:
        print(f"\nЗагрузка: {model_path.name}")
        print(f"   Размер: {model_path.stat().st_size / 1e9:.2f} GB")
        print(f"   Контекст: {n_ctx}")
        print(f"   Потоки: {n_threads or 'все'}")
        print(f"   Vision support: {'да' if has_mmproj else 'нет'}")
    
    try:
        kwargs = {
            "model_path": str(model_path),
            "n_ctx": n_ctx,
            "verbose": verbose,
            "n_threads": n_threads or -1,
            # n_gpu_layers=0: отключаем GPU offload.
            # На Mac Metal попытка выделить ~4.4 GB вызывает segfault.
            # Модель работает на CPU — медленнее, но стабильно.
            "n_gpu_layers": n_gpu_layers,
            # logits_all=True: вычисляет логиты для ВСЕХ токенов промпта.
            # llm._scores имеет форму [n_prompt_tokens, n_vocab].
            # Сгенерированный токен НЕ включается в _scores.
            # Нам нужна строка [-1] = последний токен промпта,
            # что соответствует outputs.logits[:, -1, :] в HF transformers.
            "logits_all": True,
        }

        # КРИТИЧНО: chat_handler активирует vision pipeline (mmproj).
        # Без chat_handler create_chat_completion игнорирует image_url
        # и обрабатывает только текст — модель не видит изображения.
        # _ChatHandlerClass инициализируется с mmproj_path для кодирования
        # изображений через CLIP/SigLIP проектор.
        if has_mmproj and _ChatHandlerClass is not None:
            chat_handler = _ChatHandlerClass(
                clip_model_path=mmproj_path,
                verbose=verbose,
            )
            # Патч: отключаем GPU в mtmd_context (use_gpu=False).
            # Без патча _init_mtmd_context жёстко устанавливает use_gpu=True
            # -> Metal пытается выделить ~4.4 GB -> segfault на Mac.
            if n_gpu_layers == 0:
                _patch_chat_handler_no_gpu(chat_handler)
                if verbose:
                    print("   ℹGPU patch applied (use_gpu=False для mmproj)")
            kwargs["chat_handler"] = chat_handler
            if verbose:
                print(f"   Vision chat handler: {_CHAT_HANDLER_NAME}")
        elif has_mmproj:
            # Fallback: передаём mmproj напрямую (старый API)
            kwargs["mmproj"] = mmproj_path
            if verbose:
                print("   chat_handler недоступен, используем mmproj напрямую")
        else:
            if verbose:
                print("   mmproj не найден — модель работает БЕЗ изображений!")

        llm = Llama(**kwargs)
        if verbose:
            print("Модель загружена!")
        return llm
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        return None


# 3. Подготовка попарного промпта
def _image_to_data_uri(
    image_path: str,
    size: Optional[Tuple[int, int]] = None,
    hf_preprocess: bool = True,
) -> str:
    """
    Загружает изображение и возвращает data URI (base64 PNG, без потерь).

    Режимы ресайза:
    - hf_preprocess=True (default): воспроизводит HF Qwen2VLImageProcessor:
        1. Ресайзим до e2e_eval.py fixed_image_size=(224,224) BILINEAR
        2. Применяем _smart_resize(224,224) -> 448×448 (кратно 28, ≥ MIN_PIXELS)
        Это идентично тому что делает HF перед передачей в модель.
    - hf_preprocess=False + size задан: фиксированный размер (fallback).

    PNG без потерь — избегаем артефактов JPEG которые влияют на логиты.
    """
    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")

        if hf_preprocess:
            # Шаг 1: ресайз до 224×224 как в e2e_eval.py (fixed_image_size)
            # e2e_eval.py использует BILINEAR для этого шага
            e2e_size = (224, 224)
            img_224 = img_rgb.resize(e2e_size, Image.Resampling.BILINEAR)
            # Шаг 2: smart_resize как в HF Qwen2VLImageProcessor
            # HF использует BICUBIC для финального ресайза патчей
            target_h, target_w = _smart_resize(e2e_size[1], e2e_size[0])
            img_resized = img_224.resize(
                (target_w, target_h), Image.Resampling.BICUBIC
            )
        elif size is not None:
            target_h, target_w = size[1], size[0]
            img_resized = img_rgb.resize(
                (target_w, target_h), Image.Resampling.BILINEAR
            )
        else:
            # Fallback: smart_resize от оригинального размера
            orig_h, orig_w = img_rgb.height, img_rgb.width
            target_h, target_w = _smart_resize(orig_h, orig_w)
            img_resized = img_rgb.resize(
                (target_w, target_h), Image.Resampling.BILINEAR
            )

        buf = io.BytesIO()
        img_resized.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def prepare_pairwise_messages(
    query_image_path: str,
    candidate_image_path: str,
    candidate_name: str,
    candidate_caption: str,
    caption_max_length: int = 300,
) -> List[Dict]:
    """
    Формирует сообщения для попарного сравнения Query Image и Candidate (Image + Text).

    Промпт идентичен e2e_eval.py (_rerank_all_candidates, строки 269-289).
    Изображения ресайзятся через _smart_resize (воспроизводит HF Qwen2VLImageProcessor):
    224×224 -> 448×448, соотношение сторон сохраняется, H и W кратны 28.
    Передаются через base64 data URI чтобы GGUF runtime не применял свой ресайз.
    """
    query_uri = _image_to_data_uri(query_image_path)
    cand_uri = _image_to_data_uri(candidate_image_path)

    caption = candidate_caption[:caption_max_length]

    # Текст промпта идентичен e2e_eval.py
    prompt_text = (
        "Question: Are these photos showing"
        f" the same landmark: \"{candidate_name}\"?\n"
        f"Candidate details: {caption}\n"
        "Answer only with Yes or No."
    )

    # System message как строка (не список) — Qwen25VLChatHandler корректно
    # рендерит его в CHAT_FORMAT: <|im_start|>system\nYou are...<|im_end|>
    # Это воспроизводит HF chat template который добавляет system message автоматически.
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Query Photo:"},
                {"type": "image_url", "image_url": {"url": query_uri}},
                {"type": "text", "text": "Candidate Photo:"},
                {"type": "image_url", "image_url": {"url": cand_uri}},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]


# 4. Вычисление P(yes) через низкоуровневый API

# Кеш ID токенов Yes/No — инициализируется один раз при первом вызове
_YES_TOKEN_ID: Optional[int] = None
_NO_TOKEN_ID: Optional[int] = None

# Счётчик вызовов get_yes_probability — для диагностики первых N вызовов
_GET_YES_PROB_CALL_COUNT: int = 0
_DIAG_CALLS: int = 3  # Выводить диагностику для первых N вызовов


def _get_yes_no_ids(llm: Llama) -> tuple:
    """
    Возвращает (yes_id, no_id) — ID токенов ' Yes' и ' No' в словаре модели.

    ВАЖНО: Qwen25VLChatHandler добавляет <|im_start|>assistant без \n,
    поэтому модель генерирует ':' как первый токен, затем ' Yes'/' No' с пробелом.
    Используем токены с пробелом (' Yes', ' No') для корректного бинарного softmax.

    В e2e_eval.py (HF) промпт заканчивается на <|im_start|>assistant\n,
    поэтому там используются 'Yes'/'No' без пробела.
    Кешируется глобально после первого вызова.
    """
    global _YES_TOKEN_ID, _NO_TOKEN_ID
    if _YES_TOKEN_ID is not None and _NO_TOKEN_ID is not None:
        return _YES_TOKEN_ID, _NO_TOKEN_ID

    # Пробуем токены с пробелом (' Yes', ' No') — используются после ':'
    yes_tokens_space = llm.tokenize(b" Yes", add_bos=False, special=False)
    no_tokens_space = llm.tokenize(b" No", add_bos=False, special=False)

    # Fallback: токены без пробела ('Yes', 'No')
    yes_tokens = llm.tokenize(b"Yes", add_bos=False, special=False)
    no_tokens = llm.tokenize(b"No", add_bos=False, special=False)

    if len(yes_tokens_space) == 1 and len(no_tokens_space) == 1:
        _YES_TOKEN_ID = yes_tokens_space[0]
        _NO_TOKEN_ID = no_tokens_space[0]
        print(f"   ℹToken IDs: ' Yes'={_YES_TOKEN_ID}, ' No'={_NO_TOKEN_ID}")
    elif len(yes_tokens) == 1 and len(no_tokens) == 1:
        _YES_TOKEN_ID = yes_tokens[0]
        _NO_TOKEN_ID = no_tokens[0]
        print(f"   ℹToken IDs: Yes={_YES_TOKEN_ID}, No={_NO_TOKEN_ID}")
    else:
        raise ValueError(
            f"Yes/No tokenization failed: "
            f"Yes={yes_tokens}, No={no_tokens}"
        )

    return _YES_TOKEN_ID, _NO_TOKEN_ID


def _extract_logits_from_scores(
    llm: Llama, yes_id: int, no_id: int
) -> Tuple[bool, float, float]:
    """
    Извлекает logit_yes и logit_no из llm._scores после forward pass.

    Возвращает (ok, logit_yes, logit_no).
    ok=False если извлечение не удалось.

    llm._scores имеет форму [n_tokens_evaluated, n_vocab].
    После create_chat_completion(max_tokens=1) с logits_all=True:
      - В llama.cpp _scores содержит ТОЛЬКО токены промпта (без сгенерированного).
      - Строка [-1] = логиты ПОСЛЕДНЕГО токена промпта = нужная нам позиция.
      - Это соответствует outputs.logits[:, -1, :] в HF transformers.

    ВАЖНО: в отличие от HF (где logits[:, -1, :] = позиция перед генерацией),
    llama.cpp с logits_all=True НЕ добавляет сгенерированный токен в _scores.
    Поэтому нужен [-1], а НЕ [-2].

    Стратегии конвертации (от надёжной к запасной):
      1. numpy array  — llm._scores уже np.ndarray (llama-cpp-python >= 0.3)
      2. list/tuple   — итерируемый объект с вложенными строками
      3. ctypes flat  — LP_c_float плоский массив [n_tokens * n_vocab]
    """
    try:
        raw = llm._scores
        n_vocab = llm.n_vocab()

        # Стратегия 1: уже numpy array
        if isinstance(raw, np.ndarray):
            arr_2d = raw if raw.ndim == 2 else raw.reshape(1, -1)
            # [-1] = последний токен промпта (сгенерированный НЕ включён в _scores)
            row = arr_2d[-1]
            print(
                f"   ℹ_scores numpy {arr_2d.shape}, строка [-1]"
            )
            return True, float(row[yes_id]), float(row[no_id])

        # Стратегия 2: list/tuple вложенных строк
        if isinstance(raw, (list, tuple)) and len(raw) > 0:
            n_tokens = len(raw)
            row = np.array(raw[-1], dtype=np.float32)
            print(f"   ℹ_scores list[{n_tokens}], строка [-1]")
            return True, float(row[yes_id]), float(row[no_id])

        # Стратегия 3: ctypes LP_c_float плоский массив
        try:
            total_floats = len(raw)
        except TypeError:
            print(f"   _scores неизвестный тип: {type(raw)}")
            return False, 0.0, 0.0

        if total_floats == 0:
            print("   _scores пуст (logits_all=True не сработал?)")
            return False, 0.0, 0.0

        if total_floats % n_vocab != 0:
            print(
                f"   _scores размер {total_floats} "
                f"не кратен n_vocab={n_vocab}"
            )
            return False, 0.0, 0.0

        # Копируем через срез — работает для LP_c_float
        arr = np.array(raw[:total_floats], dtype=np.float32)
        n_tokens = total_floats // n_vocab
        arr_2d = arr.reshape(n_tokens, n_vocab)
        row = arr_2d[-1]
        print(
            f"   ℹ_scores ctypes flat [{n_tokens}, {n_vocab}], строка [-1]"
        )
        return True, float(row[yes_id]), float(row[no_id])

    except Exception as e:
        print(f"   Ошибка извлечения логитов из _scores: {e}")
        return False, 0.0, 0.0


def get_yes_probability(llm: Llama, messages: List[Dict]) -> float:
    """
    Вычисляет P(yes) через logits_processor перехват.

    Точное соответствие e2e_eval.py:
      logits = outputs.logits[:, -1, :]   # логиты последнего токена промпта
      logit_yes = logits[:, yes_id]
      logit_no  = logits[:, no_id]
      p_yes = softmax([logit_no, logit_yes])[1]

    При использовании chat_handler (Qwen25VLChatHandler) llm._scores[-1]
    содержит логиты последнего текстового токена промпта (без визуальных),
    что не соответствует HF. Вместо этого используем logits_processor:
    он вызывается с полными логитами ПЕРЕД генерацией первого токена,
    то есть после обработки всего контекста (текст + изображения через KV cache).
    """
    global _GET_YES_PROB_CALL_COUNT
    _GET_YES_PROB_CALL_COUNT += 1
    is_diag = _GET_YES_PROB_CALL_COUNT <= _DIAG_CALLS

    try:
        yes_id, no_id = _get_yes_no_ids(llm)
    except Exception as e:
        print(f"Не удалось получить ID токенов Yes/No: {e}")
        return 0.0

    # Перехватываем логиты через logits_processor.
    # LogitsProcessorList.__call__ принимает numpy arrays:
    #   input_ids: np.ndarray[np.intc]
    #   scores:    np.ndarray[np.single]
    #
    # Qwen25VLChatHandler добавляет <|im_start|>assistant без \n,
    # поэтому модель генерирует ':' как первый токен.
    # Используем логиты ПЕРВОГО вызова с токенами ' Yes'/' No' (с пробелом).
    # Это даёт p_yes≈0.73-0.80 для идентичных изображений.
    captured: Dict = {"logits": None}

    def _capture_logits(
        input_ids: np.ndarray, scores: np.ndarray
    ) -> np.ndarray:
        """Перехватывает логиты первого шага генерации."""
        if captured["logits"] is None:
            captured["logits"] = scores.copy()
        return scores

    try:
        from llama_cpp import LogitsProcessorList
        processor_list = LogitsProcessorList([_capture_logits])

        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=1,
            temperature=0.0,
            stream=False,
            logits_processor=processor_list,
        )

        if captured["logits"] is None:
            print("   logits_processor не вызван -> fallback")
            return _get_yes_probability_fallback(llm, messages)

        logit_yes = captured["logits"][yes_id]
        logit_no = captured["logits"][no_id]

        if is_diag:
            gen_token = (
                response.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "?")
            )
            raw = llm._scores
            raw_type = type(raw).__name__
            try:
                raw_len = len(raw)
            except Exception:
                raw_len = "?"
            print(
                f"   [DIAG #{_GET_YES_PROB_CALL_COUNT}] "
                f"gen_token={repr(gen_token)!r} "
                f"_scores type={raw_type} len={raw_len} "
                f"n_vocab={llm.n_vocab()}"
            )
            print(
                f"   [DIAG #{_GET_YES_PROB_CALL_COUNT}] "
                f"logit_yes={logit_yes:.4f} logit_no={logit_no:.4f}"
            )

        # Бинарный softmax — идентично e2e_eval.py
        max_logit = max(logit_yes, logit_no)
        exp_yes = math.exp(logit_yes - max_logit)
        exp_no = math.exp(logit_no - max_logit)
        p_yes = exp_yes / (exp_yes + exp_no)

        return round(p_yes, 4)

    except AttributeError as e:
        print(f"AttributeError: {e} -> fallback")
        return _get_yes_probability_fallback(llm, messages)
    except Exception as e:
        print(f"Ошибка (logits_processor): {e} -> fallback")
        return _get_yes_probability_fallback(llm, messages)


def _get_yes_probability_fallback(llm: Llama, messages: List[Dict]) -> float:
    """
    Fallback через create_chat_completion + top_logprobs + grammar.
    Используется если низкоуровневый API недоступен.
    """
    YES_TOKENS = {"yes", "Yes", "YES"}
    NO_TOKENS  = {"no",  "No",  "NO"}

    grammar = _get_yes_no_grammar()

    try:
        kwargs = dict(
            messages=messages,
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            top_logprobs=2,
            stream=False,
        )
        if grammar is not None:
            kwargs["grammar"] = grammar

        response = llm.create_chat_completion(**kwargs)
        logprobs_list = response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]

        logit_yes = None
        logit_no  = None
        for item in logprobs_list:
            token = item.get("token", "").strip()
            logprob = item.get("logprob", -100)
            if token in YES_TOKENS and logit_yes is None:
                logit_yes = logprob
            elif token in NO_TOKENS and logit_no is None:
                logit_no = logprob

        if logit_yes is not None and logit_no is not None:
            max_logit = max(logit_yes, logit_no)
            exp_yes = math.exp(logit_yes - max_logit)
            exp_no  = math.exp(logit_no  - max_logit)
            return round(exp_yes / (exp_yes + exp_no), 4)
        elif logit_yes is not None:
            return 1.0
        elif logit_no is not None:
            return 0.0
        return 0.0

    except Exception as e:
        print(f"Fallback ошибка: {e}")
        return 0.0


def _get_yes_no_grammar() -> Optional["LlamaGrammar"]:
    """Возвращает LlamaGrammar для Yes/No (кешируется)."""
    global _YES_NO_GRAMMAR
    if not LLAMA_CPP_AVAILABLE:
        return None
    if _YES_NO_GRAMMAR is None:
        try:
            _YES_NO_GRAMMAR = LlamaGrammar.from_string('root ::= "Yes" | "No"')
        except Exception as e:
            print(f"Не удалось создать grammar: {e}")
    return _YES_NO_GRAMMAR


# 5. Ранжирование кандидатов
def rank_candidates(llm: Llama, query_image_path: str, candidates: List[Dict]) -> List[Dict]:
    """
    Оценивает каждого кандидата и возвращает отсортированный список по P(yes).
    """
    print(f"\nНачало попарного ранжирования ({len(candidates)} кандидатов)...")
    results = []
    
    for i, cand in enumerate(candidates):
        cand_id = cand.landmark_id
        cand_name = cand.landmark_name

        # get_top_image() — изображение с максимальным FAISS score,
        # идентично e2e_eval.py: result.get_top_image()
        top_image = cand.get_top_image()
        if top_image is None:
            print(f"  [{i+1}/{len(candidates)}] Пропуск (нет данных gallery_images)")
            results.append({"rank": 0, "landmark_id": cand_id, "p_yes": 0.0, "status": "missing_gallery"})
            continue
        cand_caption = top_image.caption
        cand_img_path = os.path.join(IMAGE_DIR, top_image.image_path)

        print(f"  [{i+1}/{len(candidates)}] Оценка кандидата {cand_id}...", end=" ")

        if not Path(cand_img_path).exists():
            print("Пропуск (нет изображения)")
            results.append({"rank": 0, "landmark_id": cand_id, "p_yes": 0.0, "status": "missing_image"})
            continue

        messages = prepare_pairwise_messages(
            query_image_path, cand_img_path, cand_name, cand_caption
        )
        p_yes = get_yes_probability(llm, messages)
        
        results.append({
            "rank": 0, # Будет заполнено после сортировки
            "landmark_id": cand_id,
            "p_yes": p_yes,
            "status": "success"
        })
        print(f"P(yes)={p_yes:.4f}")
    
    # Сортировка по убыванию P(yes)
    results.sort(key=lambda x: x["p_yes"], reverse=True)
    
    # Присваиваем финальные ранги
    for rank, item in enumerate(results, start=1):
        item["rank"] = rank
        
    return results


# 6. Главный тестовый сценарий
def main_test(retrieved_candidates: List[Dict], query_image_path: str, true_lid: str):
    """Главная функция теста ранжирования."""
    print("=" * 70)
    print("ТЕСТ AI TOUR GUIDE: РАНЖИРОВАНИЕ ПО P(yes) (GGUF)")
    print("=" * 70)
    print(f"Query Image: {query_image_path}")
    print(f"True Landmark ID: {true_lid}")

    # 1. Выбор модели
    available_files = {
        "q5_k_m": Path(MODEL_BASE) / "model-q5_k_m.gguf",
        "f16": Path(MODEL_BASE) / "model-f16.gguf",
    }
    selected_model = None
    for key, path in available_files.items():
        if path.exists():
            selected_model = str(path)
            size_gb = round(path.stat().st_size / 1e9, 2)
            print(f"Выбор: {path.name} ({size_gb} GB)")
            break
    
    if not selected_model:
        print("Нет .gguf файлов в директории")
        return
    
    # 2. Поиск mmproj
    mmproj_path = str(Path(MODEL_BASE) / "mmproj-model-f16.gguf")
    mmproj_exists = Path(mmproj_path).exists()

    # 3. Загрузка модели
    print("\nШаг 1: Загрузка модели...")
    # n_ctx=2048: два изображения 224×224 дают ~512 визуальных токенов
    # + текст промпта (~200 токенов) = ~712 токенов на запрос.
    # n_ctx=8192 вызывает segfault на Mac: KV cache на Metal GPU
    # вырастает до ~4.4 GB (recommendedMaxWorkingSetSize = 1.6 GB).
    # n_ctx=2048 -> KV cache = 744 MB — укладывается в лимит.
    # n_gpu_layers=0: отключаем offload весов модели на GPU.
    llm = load_gguf_model(
        selected_model,
        mmproj_path=mmproj_path if mmproj_exists else None,
        n_ctx=2048,
        n_threads=None,
        n_gpu_layers=0,
        verbose=True,
    )
    if not llm:
        print("Не удалось загрузить модель")
        return

    # 4. Ранжирование
    print("\nШаг 2: Оценка и ранжирование кандидатов...")
    t0 = time.time()
    ranked_results = rank_candidates(llm, query_image_path, retrieved_candidates)
    total_time = time.time() - t0

    # 5. Вывод результатов
    print("\n" + "=" * 70)
    print("ИТОГОВЫЙ РЕЙТИНГ КАНДИДАТОВ")
    print("=" * 70)
    print(f"{'Ранг':<5} | {'Landmark ID':<20} | {'P(yes)':<10} | {'Статус'}")
    print("-" * 70)
    for res in ranked_results:
        marker = "(GT)" if str(res["landmark_id"]) == str(true_lid) else ""
        print(f"{res['rank']:<5} | {res['landmark_id']:<20} | {res['p_yes']:<10.4f} | {res['status']} {marker}")
    print("=" * 70)
    print(f"⏱Общее время оценки: {total_time:.2f} сек")

    # Unknown detection — синхронизировано с e2e_eval.py (rerank_threshold)
    # и train.py (none_threshold=0.5): если max(P(yes)) < порога -> "unknown"
    top_score = ranked_results[0]["p_yes"] if ranked_results else 0.0
    is_unknown = not ranked_results or top_score < RERANK_THRESHOLD
    if is_unknown:
        print(f"Unknown detection: UNKNOWN (top P(yes)={top_score:.4f} < {RERANK_THRESHOLD})")

    # Проверка, попал ли Ground Truth в топ-1 (только если не unknown)
    top_1_is_correct = (
        not is_unknown
        and str(ranked_results[0]["landmark_id"]) == str(true_lid)
    )
    print(f"Hit@1 (Ground Truth на 1 месте): {'ДА' if top_1_is_correct else 'НЕТ'}")

    # 6. Сохранение лога
    log_path = Path(__file__).parent / "results" / "ranking_results.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump({
            "query_image": query_image_path,
            "true_landmark_id": true_lid,
            "hit_at_1": top_1_is_correct,
            "is_unknown": is_unknown,
            "rerank_threshold": RERANK_THRESHOLD,
            "total_time_sec": round(total_time, 2),
            "model": Path(selected_model).name,
            "mmproj_used": mmproj_exists,
            "results": ranked_results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nЛог сохранён: {log_path}")
    print("\nТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")


# 7. Точка входа (Пример использования)
QUERY_ITEM_IDX = 0  # Индекс тестового примера в val.json

if __name__ == "__main__":

    with open(VAL_DATASET, mode="r", encoding="utf-8") as f:
        val_data = json.load(f)

    if QUERY_ITEM_IDX >= len(val_data):
        print(f"QUERY_ITEM_IDX={QUERY_ITEM_IDX} выходит за пределы датасета (размер: {len(val_data)})")
        sys.exit(1)

    retriever = LandmarkRetriever.from_index_dir(INDEX_PATH)

    item = val_data[QUERY_ITEM_IDX]
    image_path = os.path.join(IMAGE_DIR, item.get("query_image", ""))
    img = Image.open(image_path).convert("RGB")

    retrieved = retriever.retrieve(img, top_k=RETRIEVAL_TOP_K, faiss_k=RETRIEVAL_FAISS_K)
    true_lid = item["candidates"][item["target_idx"]]["landmark_id"]
    try:
        main_test(retrieved, image_path, true_lid)

    except Exception as e:
        print(f"\nКритическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)