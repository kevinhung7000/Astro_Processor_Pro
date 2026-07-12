# 🛠️ Optimization Plan — v1.3.0

[English](#english) | [繁體中文](#繁體中文)

> 📝 Scope: unlike the v1.2.x series, none of the items below come from a prior plan doc or release note — they were found by reading `Astro_Processor_Pro.py` directly (function bodies, not just docstrings/comments). This plan has **two tracks**: Track A is one focused item (true bit-depth pipeline) that's substantial enough to warrant its own investigation pass before any code is written. Track B is a maintenance grab-bag of smaller, independently-shippable items — explicitly separate from Track A so a fast-follow on Track B doesn't need to wait on Track A's investigation.
>
> Nothing in this plan has been investigated in code yet — all items are at the "confirmed as a real gap by reading the source" stage, same as where the v1.2.1 plan started for StarNet2/Cosmic Clarity. No effort estimates or status markers below should be read as more than a starting guess.

---

## English

Candidate improvements for v1.3.0, in suggested priority order. Not a strict commitment — same as `ROADMAP.md`.

### Track A — Pipeline Bit Depth

#### High priority

- **Final output is 8-bit precision even when exported as 16-bit TIFF** — `finish_pipeline()` converts to `uint8` at multiple internal stages: the denoise step (`img8_tmp = (np.clip(img_work, 0, 1) * 255).astype(np.uint8)`), the clarity/sharpen step (`img8_cs = (np.clip(img_work, 0, 1) * 255).astype(np.uint8)`), and `boost_saturation()` itself returns `uint8`. `save_image_files()` then writes the "16-bit TIFF" by upscaling that 8-bit result (`img8.astype(np.uint16) * 257`) rather than working with genuine 16-bit data — the TIFF container is 16-bit, but the tonal information inside it is not.
  - *Why it matters here specifically*: this app's core operations — background gradient subtraction and dynamic-range stretch — are both aggressive tonal manipulations applied to faint, low-contrast source data (nebulosity, sky background). That's exactly the scenario where 8-bit quantization shows up as visible banding after stretching, more so than in general photo editing.
  - *Effort*: unknown until investigated — likely medium-to-large. This isn't a single-line fix; it touches every stage of `finish_pipeline()` (denoise, clarity/sharpen, saturation/gain) plus whatever OpenCV calls inside those stages currently assume 8-bit input (e.g. `cv2.bilateralFilter`, `cv2.fastNlMeansDenoisingColored`, and the clarity/sharpen kernels all currently operate on `uint8`).
  - *Suggested first step*: a scoped investigation (not a code change) into which of `denoise()`, `apply_clarity_and_sharpen()`, and `boost_saturation()` can run in float32 or uint16 without a full rewrite of their underlying OpenCV calls — some OpenCV ops have native 16-bit support, others don't and would need a different algorithm or a custom implementation. Output of that investigation should be a per-function feasibility note, similar in format to what the v1.2.2 plan did for Cosmic Clarity before code was written.
  - *Open question*: whether this should be an always-on pipeline change or an opt-in "high precision" mode, given the unknown performance cost of running heavier operations (e.g. NLM denoise) at higher bit depth.

### Track B — Maintenance (independent, carried findings)

#### Medium priority

- **Status bar GPU readout still doesn't have the DirectML fallback that the monitor panel got in v1.2.1** — `get_status_bar_html()` and `get_system_stats_html()` have near-identical GPU-detection logic, but only `get_system_stats_html()` received the v1.2.1 fix that shows "DirectML acceleration active" instead of going blank/`N/A` when `GPUtil` can't find an NVIDIA card. `get_status_bar_html()` still falls straight to `gpu_str = "N/A"` for AMD/Intel users. This has been listed as an unaddressed known limitation in both the v1.2.1 and v1.2.2 release notes; reading the code confirms the specific reason — the fix was applied to one function and not the other structurally-identical one.
  - *Effort*: low — same fallback logic already exists in `get_system_stats_html()` and can likely be adapted directly.

- **Cosmic Clarity Sharpen support** — carried over from the v1.2.2 plan's own follow-up note. `SetiAstroCosmicClarity.py` (Sharpen) uses the same fixed `input/`/`output/` folder convention as the already-supported Denoise tool, but with a different output suffix (`_sharpened` instead of `_denoised`). A near-duplicate of `run_cosmic_clarity_denoise()` would be needed.
  - *Effort*: low — mechanically similar to already-shipped code.

#### Low priority

- **Batch processing has no cancel/stop control** — `batch_process_fn()` is a generator that yields progress per file, which is good for showing live progress, but there's no interrupt check inside the per-file loop. For a large batch — especially one using an external tool per image — there's currently no way to stop a batch run once started short of closing the app.
  - *Effort*: low-to-medium, depends on how cleanly a stop flag can be threaded through Gradio's generator-based progress updates.

- **No control over which export formats get written** — `save_image_files()` unconditionally writes both a JPG and a 16-bit TIFF for every export; `export_fn()` and `batch_process_fn()` both call it without a format choice. For batch runs where only one format is actually needed, this doubles both export time and disk usage with no way to opt out.
  - *Effort*: low — likely just a checkbox/dropdown feeding into an existing function signature.

### 💬 Feedback

Direction, not commitment, same as always. Track A's investigation step should happen before committing to any specific approach — it may turn out the performance cost of higher bit depth is too high for live preview and this needs a "preview vs. export precision" split rather than a single across-the-board change. Track B items are independent of Track A and of each other; any could ship on its own.

---

## 繁體中文

v1.3.0 候選優化項目，依建議優先度排序。跟 `ROADMAP.md` 一樣，不是硬性承諾。

> 📝 範圍說明：跟 v1.2.x 系列不同，以下項目都不是延續自之前的計畫文件或發版說明——是直接讀 `Astro_Processor_Pro.py` 原始碼（讀函式內容，不只是看註解/docstring）找出來的。這份計畫分成**兩條軌**：A 軌是單一項目（真實位元深度 pipeline），份量大到需要先做一次獨立調查再決定怎麼動手；B 軌是可以各自獨立出貨的維護項目，刻意跟 A 軌分開，這樣 B 軌要先出的話不用等 A 軌調查完。
>
> 以下所有項目目前都還沒有實際去程式碼裡動手調查——都停在「讀原始碼確認是真的缺口」這個階段，跟 v1.2.1 計畫當初對 StarNet2/Cosmic Clarity 的起點一樣。下面的工作量估計與任何標記都只是起始猜測，不代表已查證。

### A 軌 — Pipeline 位元深度

#### 高優先

- **就算匯出成 16-bit TIFF，實際輸出仍然只有 8-bit 精度** — `finish_pipeline()` 在多個內部階段都轉成了 `uint8`：降噪那一步（`img8_tmp = (np.clip(img_work, 0, 1) * 255).astype(np.uint8)`）、Clarity/銳化那一步（`img8_cs = (np.clip(img_work, 0, 1) * 255).astype(np.uint8)`），以及 `boost_saturation()` 本身回傳的也是 `uint8`。`save_image_files()` 寫「16-bit TIFF」時，其實是把這個 8-bit 結果直接撐大（`img8.astype(np.uint16) * 257`），而不是真的用 16-bit 精度算出來的資料——TIFF 容器是 16-bit，但裡面的色調資訊不是。
  - *為什麼這對這套程式特別重要*：這套程式的核心操作——背景漸層扣除、動態範圍拉伸——都是對本來就很暗、對比很低的原始資料（星雲、天空背景）做很激進的色調處理。這正是 8-bit 量化在拉伸後最容易出現色階斷層（banding）的情境，比一般修圖更明顯。
  - *工作量*：還沒調查，先假設是中～大。這不是一行就能改完的修正，牽涉 `finish_pipeline()` 裡的每一步（降噪、Clarity/銳化、飽和度/增益），還有這些步驟裡目前假設輸入是 8-bit 的 OpenCV 呼叫（例如 `cv2.bilateralFilter`、`cv2.fastNlMeansDenoisingColored`，以及 Clarity/銳化用的 kernel 目前都是對 `uint8` 操作）。
  - *建議第一步*：先做一次範圍明確的可行性調查（不是直接改程式），確認 `denoise()`、`apply_clarity_and_sharpen()`、`boost_saturation()` 裡哪些可以在不整個重寫底層 OpenCV 呼叫的前提下改用 float32 或 uint16——有些 OpenCV 操作原生支援 16-bit，有些不支援，需要換演算法或自己實作。調查結果應該產出一份逐函式的可行性筆記，格式比照 v1.2.2 計畫處理 Cosmic Clarity 時「先查證、再動手」的做法。
  - *待確認的問題*：這應該做成永遠開啟的 pipeline 變更，還是一個可選的「高精度」模式——因為目前不知道用更高位元深度跑比較重的運算（例如 NLM 降噪）效能成本會有多大。

### B 軌 — 維護項目（各自獨立，屬於延續發現）

#### 中優先

- **狀態列的 GPU 顯示，還沒補上監控面板在 v1.2.1 拿到的 DirectML fallback** — `get_status_bar_html()` 跟 `get_system_stats_html()` 的 GPU 偵測邏輯幾乎一模一樣，但只有 `get_system_stats_html()` 拿到 v1.2.1 的修正（`GPUtil` 找不到 NVIDIA 顯卡時顯示「DirectML 加速中」，而不是空白/`N/A`）。`get_status_bar_html()` 目前對 AMD/Intel 使用者還是直接落到 `gpu_str = "N/A"`。這件事在 v1.2.1 跟 v1.2.2 的發版說明裡都被列為「未處理的已知限制」；讀了程式碼後可以確認具體原因——修正只套用到其中一個函式，另一個結構幾乎一樣的函式沒有跟著改。
  - *工作量*：低——`get_system_stats_html()` 裡已經有現成的 fallback 邏輯，應該可以直接搬過去用。

- **Cosmic Clarity Sharpen 支援** — 延續自 v1.2.2 計畫自己點名的後續項目。`SetiAstroCosmicClarity.py`（Sharpen）用的是跟已支援的 Denoise 工具一樣的固定 `input/`/`output/` 資料夾慣例，只是輸出檔名字尾不同（`_sharpened` 而不是 `_denoised`）。需要一支跟 `run_cosmic_clarity_denoise()` 幾乎一樣的函式。
  - *工作量*：低——跟已經上線的程式碼機制上很類似。

#### 低優先

- **批次處理沒有中止/停止機制** — `batch_process_fn()` 是一個逐檔案 yield 進度的 generator，這點對即時顯示進度很好，但整個逐檔迴圈裡沒有任何中斷檢查點。批次量大的時候——尤其是每張圖都要呼叫外部工具的情境——目前除了關掉整個程式，沒有辦法在批次跑到一半時停下來。
  - *工作量*：低～中，取決於能不能乾淨地把一個「停止旗標」接進 Gradio 那套 generator 式進度更新機制裡。

- **匯出格式沒有選擇權** — `save_image_files()` 不論如何都會同時寫出 JPG 跟 16-bit TIFF 兩個檔案，`export_fn()` 跟 `batch_process_fn()` 呼叫它時都沒有格式選項。如果批次處理其實只需要其中一種格式，匯出時間跟硬碟用量都會白白變成兩倍，而且沒有辦法關掉。
  - *工作量*：低——大概只需要一個勾選框/下拉選單，接進既有函式的參數即可。

### 💬 意見回饋

跟以往一樣，這只是方向，不是承諾。A 軌建議先完成調查步驟，再決定要用哪種做法——調查完也有可能發現高位元深度的效能成本太高，不適合套用在即時預覽上，屆時可能需要「預覽精度」跟「匯出精度」分開處理，而不是整條 pipeline 一次到位改掉。B 軌的每一項都跟 A 軌、也跟彼此互相獨立，任何一項都可以單獨先出。
