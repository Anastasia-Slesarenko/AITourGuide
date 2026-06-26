"""
Патч для torchao_utils.py в SGLang v0.5.13.
1. Адаптирует под новый API torchao (классы-конфиги вместо функций).
2. Исключает visual encoder из int8 квантования (Conv3d несовместим).
"""
path = "/sgl-workspace/sglang/python/sglang/srt/layers/torchao_utils.py"
with open(path) as f:
    src = f.read()

# 1. Заменяем старый импорт на новый API
old_import = (
    "    from torchao.quantization import (\n"
    "        float8_dynamic_activation_float8_weight,\n"
    "        float8_weight_only,\n"
    "        int4_weight_only,\n"
    "        int8_dynamic_activation_int8_weight,\n"
    "        int8_weight_only,\n"
    "        quantize_,\n"
    "    )\n"
    "    from torchao.quantization.observer import PerRow, PerTensor"
)

new_import = (
    "    from torchao.quantization import (\n"
    "        quantize_,\n"
    "        Int8WeightOnlyConfig,\n"
    "        Int4WeightOnlyConfig,\n"
    "        Int8DynamicActivationInt8WeightConfig,\n"
    "    )\n"
    "    try:\n"
    "        from torchao.quantization import (\n"
    "            Float8WeightOnlyConfig,\n"
    "            Float8DynamicActivationFloat8WeightConfig,\n"
    "        )\n"
    "    except ImportError:\n"
    "        Float8WeightOnlyConfig = None\n"
    "        Float8DynamicActivationFloat8WeightConfig = None\n"
    "    try:\n"
    "        from torchao.quantization.observer import PerRow, PerTensor\n"
    "    except ImportError:\n"
    "        PerRow = None\n"
    "        PerTensor = None"
)

assert old_import in src, "Import pattern not found!"
src = src.replace(old_import, new_import)

# 2. Заменяем вызовы старого API на новый + исключаем visual encoder
# Фильтр: квантуем только LLM-часть (model.layers.*), не visual.*
old_int8wo = (
    "        quantize_(model, int8_weight_only(),"
    " filter_fn=proj_filter_conv3d)"
)
new_int8wo = (
    "        def llm_only_filter(mod, fqn):\n"
    "            return 'visual' not in fqn and proj_filter_conv3d(mod, fqn)\n"
    "        quantize_(model, Int8WeightOnlyConfig(),"
    " filter_fn=llm_only_filter)"
)
src = src.replace(old_int8wo, new_int8wo)

old_int8dq = (
    "        quantize_(model,"
    " int8_dynamic_activation_int8_weight(), filter_fn=filter_fn)"
)
new_int8dq = (
    "        quantize_(model,"
    " Int8DynamicActivationInt8WeightConfig(), filter_fn=filter_fn)"
)
src = src.replace(old_int8dq, new_int8dq)

old_int4wo = (
    "        quantize_(model, int4_weight_only(group_size=group_size),"
    " filter_fn=filter_fn)"
)
new_int4wo = (
    "        quantize_(model, Int4WeightOnlyConfig(group_size=group_size),"
    " filter_fn=filter_fn)"
)
src = src.replace(old_int4wo, new_int4wo)

old_fp8wo = (
    "        quantize_(model, float8_weight_only(), filter_fn=filter_fn)"
)
new_fp8wo = (
    "        quantize_(model, Float8WeightOnlyConfig(),"
    " filter_fn=filter_fn)"
)
src = src.replace(old_fp8wo, new_fp8wo)

with open(path, "w") as f:
    f.write(src)
print("OK: torchao_utils.py patched")
