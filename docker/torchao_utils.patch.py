"""
Патч для torchao_utils.py в SGLang v0.5.13.
SGLang написан под старый API torchao (функции int8_weight_only и т.д.),
но образ поставляется с новой версией torchao где API изменился на классы-конфиги
(Int8WeightOnlyConfig и т.д.).

Патч заменяет вызовы старого API на новый.
"""
path = "/sgl-workspace/sglang/python/sglang/srt/layers/torchao_utils.py"
with open(path) as f:
    src = f.read()

# Заменяем старый импорт на новый API
old_import = """    from torchao.quantization import (
        float8_dynamic_activation_float8_weight,
        float8_weight_only,
        int4_weight_only,
        int8_dynamic_activation_int8_weight,
        int8_weight_only,
        quantize_,
    )
    from torchao.quantization.observer import PerRow, PerTensor"""

new_import = """    from torchao.quantization import (
        quantize_,
        Int8WeightOnlyConfig,
        Int4WeightOnlyConfig,
        Int8DynamicActivationInt8WeightConfig,
    )
    try:
        from torchao.quantization import (
            Float8WeightOnlyConfig,
            Float8DynamicActivationFloat8WeightConfig,
        )
    except ImportError:
        Float8WeightOnlyConfig = None
        Float8DynamicActivationFloat8WeightConfig = None
    try:
        from torchao.quantization.observer import PerRow, PerTensor
    except ImportError:
        PerRow = None
        PerTensor = None"""

assert old_import in src, "Import pattern not found!"
src = src.replace(old_import, new_import)

# Заменяем вызовы старого API на новый
replacements = [
    # int8wo
    ("quantize_(model, int8_weight_only(), filter_fn=proj_filter_conv3d)",
     "quantize_(model, Int8WeightOnlyConfig(), filter_fn=proj_filter_conv3d)"),
    # int8dq
    ("quantize_(model, int8_dynamic_activation_int8_weight(), filter_fn=filter_fn)",
     "quantize_(model, Int8DynamicActivationInt8WeightConfig(), filter_fn=filter_fn)"),
    # int4wo
    ("quantize_(model, int4_weight_only(group_size=group_size), filter_fn=filter_fn)",
     "quantize_(model, Int4WeightOnlyConfig(group_size=group_size), filter_fn=filter_fn)"),
    # fp8wo
    ("quantize_(model, float8_weight_only(), filter_fn=filter_fn)",
     "quantize_(model, Float8WeightOnlyConfig(), filter_fn=filter_fn)"),
    # fp8dq PerTensor
    ("quantize_(model, float8_dynamic_activation_float8_weight(granularity=PerTensor()), filter_fn=filter_fn)",
     "quantize_(model, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()), filter_fn=filter_fn)"),
    # fp8dq PerRow
    ("quantize_(model, float8_dynamic_activation_float8_weight(granularity=PerRow()), filter_fn=filter_fn)",
     "quantize_(model, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()), filter_fn=filter_fn)"),
]

for old, new in replacements:
    if old in src:
        src = src.replace(old, new)
        print(f"Replaced: {old[:50]}...")

with open(path, "w") as f:
    f.write(src)
print("OK: torchao_utils.py patched for new torchao API")
