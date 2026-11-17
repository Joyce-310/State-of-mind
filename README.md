# Qwen2.5-0.5B-Instruct 中文六情緒 SLM（LoRA 微調）

## 專案內容
- `data/train.jsonl`：120 筆示範訓練資料，每種情緒 20 筆
- `train.py`：分層切分、LoRA 微調、測試集評估與儲存 adapter
- `predict.py`：載入微調後的模型，互動輸入句子並辨識情緒
- `requirements.txt`：Python 套件

## Mac 安裝
建議 Python 3.10 或 3.11、Apple Silicon Mac、macOS 14+。
在終端機切換到本專案資料夾後執行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

安裝 Xcode command line tools（若尚未安裝）：
```bash
xcode-select --install
```

確認 MPS：
```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```
若輸出 True，代表 PyTorch 可使用 Mac GPU。若為 False，程式會退回 CPU（較慢）。

## 訓練與測試
```bash
python train.py
```
首次執行會從 Hugging Face 下載 `Qwen/Qwen2.5-0.5B-Instruct`。訓練完成後：
- LoRA adapter：`outputs/qwen-emotion-lora/`
- 分層保留測試資料：`data/test.jsonl`
- 終端機會列出 Accuracy、Precision、Recall、F1-score 和混淆矩陣。

## 使用模型
```bash
python predict.py
```
輸入中文句子；輸入 `q` 離開。

## JSONL 資料格式
每一行是一筆 JSON，欄位包含：
- `text`：原始句子
- `label`：人工標註的情緒
- `messages`：Qwen chat 格式，assistant 的回答是正確標籤

標籤固定為：開心、悲傷、憤怒、恐懼、平靜、焦慮。
新增資料時，請確保 `label` 與 `messages` 最後一則 assistant 內容一致。

## 重要研究限制
附檔只有 120 筆示範資料（每類 20 筆），用途是確認程式流程，不足以訓練出可靠的情緒理解能力。正式專題應擴充多來源、自然且多樣的中文句子，讓標註者複核；建議每類至少數百筆起，並保留不曾參與訓練的獨立測試集。
六種情緒是簡化分類，句子可能同時包含多種情緒、反諷或需要上下文。模型輸出不代表心理診斷。
