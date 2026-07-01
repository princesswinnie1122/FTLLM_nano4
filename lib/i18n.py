"""Bilingual (English / 繁體中文) strings for the wizard.

Terminology cross-checked against NCHC/TWCC's own SLURM documentation
(man.twcc.ai, iservice.nchc.org.tw): account/project -> 計畫帳號,
partition -> 分區, job -> 工作, submit -> 送出, node -> 節點,
pending -> 等待中, running -> 執行中.
"""

STRINGS = {
    "lang_prompt": {
        "en": "Language / 語言選擇:",
        "zh": "Language / 語言選擇:",
    },
    "banner": {
        "en": "NANO4 LLM Fine-Tuning Job Wizard",
        "zh": "NANO4 語言模型微調精靈",
    },
    "env_missing": {
        "en": (
            "Fine-tuning environment not found at {work_root}/venv.\n"
            "  Run this first:  bash setup.sh"
        ),
        "zh": (
            "在 {work_root}/venv 找不到微調環境。\n"
            "  請先執行：bash setup.sh"
        ),
    },
    "account_intro": {
        "en": (
            "Every job on this cluster must be charged to a project account "
            "(e.g. mst114031). Only accounts your PI has gotten authorized for "
            "this specific cluster will work -- the wizard will validate this "
            "for real in a moment via a dry-run submission."
        ),
        "zh": (
            "這個叢集上的每個工作都必須指定一個計畫帳號來計費，"
            "只有已申請過授權的計畫帳號才能使用！"
        ),
    },
    "account_prompt": {
        "en": "Project/account ID to charge this job to:",
        "zh": "要用來計費的計畫帳號：",
    },
    "account_required": {
        "en": "An account ID is required.",
        "zh": "必須輸入計畫帳號。",
    },
    "dataset_prompt": {
        "en": "Path to your training dataset file (JSON or JSONL):",
        "zh": "訓練資料集檔案的路徑（JSON 或 JSONL）：",
    },
    "dataset_required": {
        "en": "A dataset path is required.",
        "zh": "必須提供資料集路徑。",
    },
    "dataset_retry": {
        "en": "Try a different path?",
        "zh": "要換個路徑再試一次嗎？",
    },
    "dataset_none": {
        "en": "No valid dataset provided.",
        "zh": "沒有提供有效的資料集。",
    },
    "dataset_detected": {
        "en": "  ✓ Detected {fmt} format, {n} training examples.",
        "zh": "  ✓ 偵測到 {fmt} 格式，共 {n} 筆訓練範例。",
    },
    "model_prompt": {
        "en": "Which base model do you want to fine-tune?",
        "zh": "你想微調哪個模型？",
    },
    "method_prompt": {
        "en": "Which fine-tuning method?",
        "zh": "要使用哪一種微調方法？",
    },
    "gpu_estimate": {
        "en": "\n  Estimated GPUs needed: {n} x {gpu_model} (based on {model} + {method})",
        "zh": "\n  預估需要的 GPU 數量：{n} 顆 {gpu_model}（根據 {model} + {method} 估算）",
    },
    "gpu_use_estimate": {
        "en": "Use {n} GPU(s)?",
        "zh": "要使用 {n} 顆 GPU 嗎？",
    },
    "gpu_custom_prompt": {
        "en": "How many GPUs (1-8)?",
        "zh": "要使用幾顆 GPU（1-8）？",
    },
    "time_choice_prompt": {
        "en": "How long do you expect this run to take?",
        "zh": "你預期這個工作要跑多久？",
    },
    "time_choice_dev": {
        "en": "Quick test (≤ 4 hours) -- dev partition",
        "zh": "快速測試（4 小時內）—— dev 分區",
    },
    "time_choice_8gpus": {
        "en": "Full training run (≤ 2 days) -- 8gpus partition",
        "zh": "正式訓練（2 天內）—— 8gpus 分區",
    },
    "time_limit_prompt": {
        "en": "Wall-clock time limit (max {max_time} for this partition):",
        "zh": "時間上限（此分區最長可設 {max_time}）：",
    },
    "defaults_confirm": {
        "en": "Use recommended default training settings (3 epochs, standard learning rate)?",
        "zh": "要使用建議的預設訓練參數嗎？(3 epochs, standard learning rate)",
    },
    "epochs_prompt": {
        "en": "Number of training epochs:",
        "zh": "訓練輪數（epochs）：",
    },
    "lr_prompt": {
        "en": "Learning rate:",
        "zh": "學習率（learning rate）：",
    },
    "batch_prompt": {
        "en": "Per-device batch size:",
        "zh": "每個裝置的批次大小（batch size）：",
    },
    "cutoff_prompt": {
        "en": "Max sequence length (cutoff_len):",
        "zh": "最大序列長度（cutoff_len）：",
    },
    "job_name_prompt": {
        "en": "Job name:",
        "zh": "Job 名稱：",
    },
    "generated_files": {
        "en": "\nGenerated job files in: {job_dir}",
        "zh": "\n已產生檔案於：{job_dir}",
    },
    "validating": {
        "en": "\nValidating with the scheduler (dry run, submits nothing)...",
        "zh": "\n正在向排程系統做檢查，不會真的送出工作...",
    },
    "dry_run_fail": {
        "en": (
            "The scheduler rejected this job (see message above). Common causes: "
            "wrong/unauthorized account, requested time exceeds the partition limit, "
            "or GPU/CPU/memory request too large. Fix your answers and rerun the wizard."
        ),
        "zh": (
            "排程系統拒絕了這個工作（詳見上方訊息）。常見原因：計畫帳號錯誤或未獲授權、"
            "申請的時間超過該分區上限、或申請的 GPU/CPU/記憶體過大。"
            "請修正你的回答並重新執行精靈。"
        ),
    },
    "output_location": {
        "en": "\nModel output (LoRA adapter or full weights) will be saved to:\n  {output_dir}",
        "zh": "\n訓練結果（LoRA adapter 或完整模型權重）將會儲存於：\n  {output_dir}",
    },
    "submit_confirm": {
        "en": "Dry run passed. Submit this job for real now?",
        "zh": "檢查通過。要正式送出這個工作嗎？",
    },
    "not_submitted": {
        "en": "\nNot submitted. You can submit it later with:\n  sbatch {sbatch_path}",
        "zh": "\n尚未送出。你之後可以用這個指令送出：\n  sbatch {sbatch_path}",
    },
    "submission_failed": {
        "en": "Submission failed:\n{msg}",
        "zh": "送出失敗：\n{msg}",
    },
    "submitted": {
        "en": "\n✓ {msg}",
        "zh": "\n✓ {msg}",
    },
    "check_status": {
        "en": "\nCheck status:   squeue -u {user}",
        "zh": "\n查看狀態：      squeue -u {user}",
    },
    "watch_logs": {
        "en": "Watch logs:     tail -f {log_path}",
        "zh": "查看訓練紀錄：  tail -f {log_path}",
    },
    "cancelled": {
        "en": "\n\nCancelled.",
        "zh": "\n\n已取消。",
    },
}


def translator(lang: str):
    def t(key: str, **kwargs) -> str:
        text = STRINGS[key][lang]
        return text.format(**kwargs) if kwargs else text
    return t
