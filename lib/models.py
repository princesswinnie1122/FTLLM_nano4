"""
Curated base models offered by the wizard, spanning small -> large so
GPU requirements can be estimated automatically. All are instruction-tuned
chat models with a template LLaMA-Factory already knows how to format.

GB-per-billion-params heuristics for --mem/--gpu sizing (rule of thumb,
not exact): LoRA trains bf16 weights + a small adapter, QLoRA trains a
4-bit quantized base + a small adapter, full fine-tuning needs weights +
gradients + fp32 optimizer state (AdamW, no ZeRO/offload in this wizard).
"""

MODELS = [
    {
        "key": "qwen2.5-0.5b",
        "label_en": "Qwen2.5 0.5B Instruct  (tiny, fast, good for testing the pipeline)",
        "label_zh": "Qwen2.5 0.5B Instruct（極小、快速，適合先測試整個流程）",
        "repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "template": "qwen",
        "params_billion": 0.5,
    },
    {
        "key": "llama3.2-1b",
        "label_en": "Llama 3.2 1B Instruct  (small, quick experiments)",
        "label_zh": "Llama 3.2 1B Instruct（小型，適合快速實驗）",
        "repo_id": "meta-llama/Llama-3.2-1B-Instruct",
        "template": "llama3",
        "params_billion": 1,
    },
    {
        "key": "llama3.2-3b",
        "label_en": "Llama 3.2 3B Instruct  (small-medium, decent quality)",
        "label_zh": "Llama 3.2 3B Instruct（中小型，品質不錯）",
        "repo_id": "meta-llama/Llama-3.2-3B-Instruct",
        "template": "llama3",
        "params_billion": 3,
    },
    {
        "key": "qwen2.5-7b",
        "label_en": "Qwen2.5 7B Instruct  (medium, strong general quality)",
        "label_zh": "Qwen2.5 7B Instruct（中型，整體品質佳）",
        "repo_id": "Qwen/Qwen2.5-7B-Instruct",
        "template": "qwen",
        "params_billion": 7,
    },
    {
        "key": "llama3.1-8b",
        "label_en": "Llama 3.1 8B Instruct  (medium, strong general quality)",
        "label_zh": "Llama 3.1 8B Instruct（中型，整體品質佳）",
        "repo_id": "meta-llama/Llama-3.1-8B-Instruct",
        "template": "llama3",
        "params_billion": 8,
    },
    {
        "key": "qwen2.5-14b",
        "label_en": "Qwen2.5 14B Instruct  (large, needs multiple GPUs)",
        "label_zh": "Qwen2.5 14B Instruct（大型，需要多顆 GPU）",
        "repo_id": "Qwen/Qwen2.5-14B-Instruct",
        "template": "qwen",
        "params_billion": 14,
    },
    {
        "key": "qwen2.5-32b",
        "label_en": "Qwen2.5 32B Instruct  (very large, QLoRA recommended)",
        "label_zh": "Qwen2.5 32B Instruct（超大型，建議使用 QLoRA）",
        "repo_id": "Qwen/Qwen2.5-32B-Instruct",
        "template": "qwen",
        "params_billion": 32,
    },
]

METHODS = [
    {
        "key": "lora",
        "label_en": "LoRA  (recommended default -- fast, low memory, good quality)",
        "label_zh": "LoRA（建議預設選項 —— 速度快、記憶體用量低、品質好）",
        "gb_per_billion": 3.0,
        "finetuning_type": "lora",
        "quantization_bit": None,
    },
    {
        "key": "qlora",
        "label_en": "QLoRA  (lowest memory -- best for the largest models)",
        "label_zh": "QLoRA（記憶體用量最低 —— 最適合超大型模型）",
        "gb_per_billion": 1.5,
        "finetuning_type": "lora",
        "quantization_bit": 4,
    },
    {
        "key": "full",
        "label_en": "Full fine-tune  (most GPU-hungry -- only for small models)",
        "label_zh": "Full fine-tune 完整微調（最耗 GPU —— 僅建議用於小型模型）",
        "gb_per_billion": 14.0,
        "finetuning_type": "full",
        "quantization_bit": None,
    },
]


def label(item: dict, lang: str) -> str:
    return item["label_zh"] if lang == "zh" else item["label_en"]


def short_label(item: dict, lang: str) -> str:
    """First clause of the label, without the parenthetical explanation."""
    text = label(item, lang)
    return text.split("(")[0].split("（")[0].strip()


def estimate_gpus(params_billion: float, gb_per_billion: float, gpu_mem_gb: int) -> int:
    """Rough single-node GPU count estimate. Always returns 1-8."""
    import math

    est_gb = params_billion * gb_per_billion
    gpus = max(1, math.ceil(est_gb / gpu_mem_gb))
    return min(gpus, 8)


def estimate_mem_gb(params_billion: float, gb_per_billion: float) -> float:
    return params_billion * gb_per_billion
