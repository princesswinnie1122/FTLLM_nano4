# NANO4 語言模型微調精靈

只要回答幾個問題，就能在 NANO4 GPU 叢集上微調語言模型 ！

> English version: [README.md](README.md)

## 使用前安裝

```bash
git clone https://github.com/princesswinnie1122/FTLLM_nano4.git
cd FTLLM_nano4 && bash setup.sh
```

## 立即試用

```bash
/work/$USER/hpc-llm-finetune/venv/bin/python finetune_wizard.py
```

快速測試時可以這樣回答：

```
語言選擇             -> 繁體中文
計畫帳號             -> 你的計畫帳號（不確定可問指導教授，例如 mst114031）
資料集檔案路徑       -> examples/sample_dataset.json
基礎模型             -> Qwen2.5 0.5B Instruct
微調方法             -> LoRA
使用建議的 GPU 數量？ -> Yes
預期執行時間         -> 快速測試（4 小時內）
使用預設訓練參數？   -> Yes
工作名稱             -> 直接按 Enter
要正式送出嗎？       -> Yes
```

接著查看狀態：

```bash
squeue -u $USER
tail -f /work/$USER/hpc-llm-finetune/jobs/<job>/logs/*.out   # 按 Ctrl+C 停止觀看
```

訓練過程中會印出一個叫「loss」的數字，應該會隨時間慢慢變小。

## 使用自己的資料

存一個 JSON 檔，裡面放你的問答範例：

```json
[
  {"instruction": "What is the capital of Taiwan?", "input": "", "output": "Taipei."}
]
```

執行精靈時輸入檔案的完整路徑即可。範例數量越多通常越好 —— 50–100
筆是不錯的起點。（也支援多輪對話的「ShareGPT」格式；如果檔案格式無法辨識，精靈會告訴你。）

### 從 MMLU 資料集系統取得資料

如果資料在 [MMLU 資料集系統網站](http://103.124.75.139/)，不需要手動寫 JSON。開啟
**題目資料庫** 分頁，點資料集的下載選單，選 **匯出訓練格式
(Alpaca)**，就會下載一個已符合上述格式的 `<資料集>_alpaca.json`。
把它 `scp` 上 nano4，再把路徑交給精靈即可：

```bash
scp <資料集>_alpaca.json <帳號>@nano4.nchc.org.tw:/work/<帳號>/
```

## 如何選擇方法

- **LoRA** —— 預設選項，速度快、記憶體用量低，沒有特別理由就選這個。
- **QLoRA** —— 適合跑最大的模型／記憶體最吃緊的情況。
- **Full fine-tune（完整微調）** —— 品質最好，但只適合小模型。

不需要自己計算 GPU 數量 —— 精靈會幫你估算，並讓你確認或修改。

## 結果在哪裡

```
/work/$USER/hpc-llm-finetune/jobs/<日期時間>_<工作名稱>/output/
```

## 常見問題

- **"Project X is not allowed to use NANO4"** —— 這個計畫帳號未獲授權使用此叢集，請你的指導教授確認。
- **資料集被拒絕** —— 確認是合法 JSON，且有 `instruction`/`output` 欄位（ShareGPT 格式則需要 `conversations`）。
- **乾跑檢查（dry-run）階段被拒絕** —— 精靈會顯示排程系統的真實錯誤訊息（例如時間申請過長）。在通過乾跑檢查之前不會真的送出工作，調整回答後重新執行即可。
- **工作一直卡在 "PENDING"** —— 叢集目前很忙，等有 GPU 空出來就會開始執行。

## 專案檔案

- `finetune_wizard.py` —— 精靈主程式
- `setup.sh` —— 第一次使用前的安裝腳本
- `examples/sample_dataset.json` —— 上面用到的範例資料集
- `lib/` —— 叢集規則、模型清單、資料集驗證、設定檔產生（細節見各檔案內的註解）
