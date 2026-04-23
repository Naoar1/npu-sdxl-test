# SDXL → QNN NPU 轉換（Kaggle TPU VM + GitHub Actions）

把 Civitai 上的 SDXL 模型轉成 Qualcomm QNN 二進位檔（給 `local-dream` Android App 用），利用免費的 Kaggle TPU v5e-8 VM（377GB RAM、96 vCPU）當作算力，由 GitHub Actions 負責編排。

## 三個 GitHub Secrets（缺一不可）

在 repo Settings → Secrets and variables → Actions 新增：

| Secret 名稱        | 用途                                                                 |
| ------------------ | -------------------------------------------------------------------- |
| `KAGGLE_JSON`      | Kaggle API 憑證 JSON 全文（從 kaggle.com 下載的 `kaggle.json` 內容） |
| `CIVITAI_API_KEY`  | Civitai API key，Notebook A 下載 SDXL safetensors 時用              |
| `HF_TOKEN`         | HuggingFace token，Notebook B 的 TPU burner 下載 gated Llama-3.2-1B 時用（必須先去 HF 網站同意 Meta Llama License） |

## Code Flow

```
GitHub Actions (workflow.yml)
    │
    ├── 計算 slug：sdxl-convert-phase{1,25}-*-{model_name}-{timestamp}
    ├── 把 __CIVITAI_VERSION_ID__ / __MODEL_NAME__ / __CIVITAI_API_KEY__ 等替換進 notebook
    │
    ├── Notebook A (Kaggle T4 GPU) — Phase 1
    │   ├── curl civitai → model.safetensors
    │   ├── uv venv 3.10.17 + torch 2.5.1+cu121
    │   └── prepare_data_sdxl.py → data_sdxl.pkl + images_sdxl/
    │   輸出留在 /kaggle/working/（B 透過 kernel_sources 掛載讀取）
    │
    └── Notebook B (Kaggle TPU v5e-8) — Phase 2-5
        ├── apt libc++1-19 / libc++abi1-19 / libunwind-19（QNN 原生依賴）
        ├── 下載 convertsdxl.zip + QNN SDK 2.28
        ├── uv venv 3.10.17 + uv sync
        ├── 從 A 的 /kaggle/input 複製 data_sdxl.pkl + images_sdxl
        │
        ├── [cell 10] 啟動 TPU burner（防 Kaggle 2h idle 自動取消）
        │   ├── uninstall jax / libtpu → install torch==2.6.0 + torch_xla[tpu]==2.6.0
        │   ├── 寫出 burner.py：Llama-3.2-1B + LoRA，GSM8K fine-tune + inference 無限迴圈
        │   ├── subprocess.Popen 一次性啟動 burner（整個 notebook 生命週期只此一次）
        │   └── 每輪結束寫 /kaggle/working/burner/rounds/round_NNNN.txt（= HW5 cell 32 的 signal）
        │
        ├── [cell 11-14] Phase 2-3：gen_quant_data_sdxl → export_onnx_sdxl → ONNX 備份 → patch 腳本
        │
        ├── [cell 15] Phase 4-5：convert_all_sdxl.sh (qairt-converter + quantizer + context-bin-gen)
        │   try:
        │       跑 convert_all_sdxl.sh（UNet quantizer 約 5h 44m 是主要耗時）
        │       stdout 只放行 [*] 里程碑 + QNN 成功/錯誤關鍵字；完整 log 寫 phase45.log
        │   finally:
        │       SIGTERM / SIGKILL burner subprocess + mem watcher
        │       打包 /kaggle/working/{model_name}_qnn2.28_{soc}.zip（成功路徑）
        │       失敗則保留 phase3_backup 供下次重跑
        │
        └── [cell 16] 統計摘要（mem profile、burner 完成輪數、output listing）

GitHub Actions 最後
    ├── kaggle kernels output 下載 B 的 /kaggle/working/
    └── upload-artifact（即使失敗也跑，if: always()）
```

## 主要執行參數（workflow_dispatch 輸入）

- `civitai_version_id`：Civitai modelVersionId（預設 `1408658`）
- `model_name`：輸出 zip 檔名前綴（預設 `customxl`）
- `realistic`：寫實風 SDXL Base / 動漫風二選一
- `min_soc`：目標 SoC，`8gen3` 或 `8gen4`（upstream 2026-04-20 之後預設改 8gen3）
- `skip_phase1` / `phase1_slug`：若要重用前一次的 A 輸出，避免重跑 GPU phase

## 設計要點

- **A+B 兩個 notebook**：A 在 T4 GPU 跑 prepare_data；B 在 TPU VM 跑主要 convert pipeline（純用它的 CPU 和 RAM，TPU 只是用來當燒機以避免 idle 被砍）
- **burner 只開一次 subprocess**：整個 notebook 跑完前都活著；convert 結束後在 cell 15 的 finally 區塊 SIGTERM → SIGKILL
- **訊號以檔案為單位**：burner 每輪寫一個 round_NNNN.txt，主 cell 15 不等它、不輪詢；burner 和 convert 是獨立平行跑
- **log hygiene**：stdout 只有 `[*] HH:MM:SS ...` 里程碑等級，Firefox 看不會掛；完整 log 全部進 `/kaggle/working/logs/*.log`
- **torch_xla 版本**：必須 2.6.0 以上（Kaggle TPU base 是 Python 3.12，2.5.0 沒有 cp312 wheel）
- **QNN 原生依賴**：Debian 13 trixie 沒有 `libc++.so.1` / `libunwind.so.1`，必須裝 LLVM 19 runtime（`libunwind-dev` 是 GNU 版的 .so.8，不能用）

## 路徑說明

```
kaggle-next/
├── github/workflows/workflow.yml          （上傳前記得改回 .github/workflows/）
├── kaggle/
│   ├── notebook_a_gpu/
│   │   ├── kernel-metadata.json
│   │   └── notebook.ipynb
│   └── notebook_b_tpu/
│       ├── kernel-metadata.json
│       └── notebook.ipynb
└── README.md
```
