"""
Патч для torchao_utils.py в SGLang v0.5.13.
Переносит float8-импорты в try/except чтобы int8wo работал со старой версией torchao.
"""
path = "/sgl-workspace/sglang/python/sglang/srt/layers/torchao_utils.py"
with open(path) as f:
    src = f.read()

old = """    from torchao.quantization import (
        float8_dynamic_activation_float8_weight,
        float8_weight_only,
        int4_weight_only,
        int8_dynamic_activation_int8_weight,
        int8_weight_only,
        quantize_,
    )"""

new = """    from torchao.quantization import (
        int4_weight_only,
        int8_dynamic_activation_int8_weight,
        int8_weight_only,
        quantize_,
    )
    try:
        from torchao.quantization import (
            float8_dynamic_activation_float8_weight,
            float8_weight_only,
        )
    except ImportError:
        float8_dynamic_activation_float8_weight = None
        float8_weight_only = None"""

assert old in src, f"Pattern not found in {path}!"
src = src.replace(old, new)
with open(path, "w") as f:
    f.write(src)
print("OK: torchao_utils.py patched")
