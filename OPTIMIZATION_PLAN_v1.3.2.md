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

- ✅ **DONE (v1.3.0)** — ~~Final output is 8-bit precision even when exported as 16-bit TIFF~~ — `finish_pipeline()` converts to `uint8` at multiple internal stages: the denoise step (`img8_tmp = (np.clip(img_work, 0, 1) * 255).astype(np.uint8)`), the clarity/sharpen step (`img8_cs = (np.clip(img_work, 0, 1) * 255).astype(np.uint8)`), and `boost_saturation()` itself returns `uint8`. `save_image_files()` then writes the "16-bit TIFF" by upscaling that 8-bit result (`img8.astype(np.uint16) * 257`) rather than working with genuine 16-bit data — the TIFF container is 16-bit, but the tonal information inside it is not.
  - *Why it matters here specifically*: this app's core operations — background gradient subtraction and dynamic-range stretch — are both aggressive tonal manipulations applied to faint, low-contrast source data (nebulosity, sky background). That's exactly the scenario where 8-bit quantization shows up as visible banding after stretching, more so than in general photo editing.
  - *Effort*: unknown until investigated — likely medium-to-large. This isn't a single-line fix; it touches every stage of `finish_pipeline()` (denoise, clarity/sharpen, saturation/gain) plus whatever OpenCV calls inside those stages currently assume 8-bit input (e.g. `cv2.bilateralFilter`, `cv2.fastNlMeansDenoisingColored`, and the clarity/sharpen kernels all currently operate on `uint8`).
  - *Suggested first step*: a scoped investigation (not a code change) into which of `denoise()`, `apply_clarity_and_sharpen()`, and `boost_saturation()` can run in float32 or uint16 without a full rewrite of their underlying OpenCV calls — some OpenCV ops have native 16-bit support, others don't and would need a different algorithm or a custom implementation. Output of that investigation should be a per-function feasibility note, similar in format to what the v1.2.2 plan did for Cosmic Clarity before code was written.
  - *Open question*: whether this should be an always-on pipeline change or an opt-in "high precision" mode, given the unknown performance cost of running heavier operations (e.g. NLM denoise) at higher bit depth.

  **✅ Resolution (v1.3.0)**: `finish_pipeline()` now runs end-to-end in float32 `[0,1]`, instead of quantizing to `uint8` three separate times (before denoise, mid clarity/sharpen, and at function exit). Changes span `denoise()` (the `fast`/bilateral-filter path now runs natively in float32, no mid-pipeline conversion), `apply_clarity_and_sharpen()` (removed the redundant 8-bit quantization between the clarity and sharpen steps), `boost_saturation()` (no longer forces a `uint8` return), and `save_image_files()` (now writes genuine full-precision `uint16` TIFFs via `tifffile`, instead of upscaling an 8-bit result by `×257`; JPEG export is the only place that still converts to 8-bit, which is correct since JPEG has no higher-precision mode). Preview functions (`update_preview_fn`, `layer_preview_fn`, `local_preview_fn`) convert to `uint8` at their own call sites for on-screen display only — this doesn't affect export precision, and `export_fn()` / `batch_process_fn()` required no changes since they already passed `result['main']` straight to `save_image_files()`.
  - *Verified*: tested against a synthetic gradient/noise image through `denoise()`, full `run_pipeline()`, `export_fn()`, and `batch_process_fn()`. Confirmed output TIFFs are genuine `uint16` with far more than 256 distinct tonal levels, and that values are *not* all evenly divisible by 257 (divisibility by 257 would indicate the old pseudo-16-bit behavior — an 8-bit result merely upscaled). Also confirmed RGB channel order is correct in both JPG and TIFF outputs, and that preview-path dtype/shape weren't broken by the change.
  - **✅ Follow-up fix (v1.3.1)**: the `"quality"` mode limitation above has now been closed. `denoise()` switched to `skimage.restoration.denoise_nl_means`, which supports float32 natively — the internal 8-bit quantization step is gone. Implementation converts to CIELAB (mirroring what `cv2.fastNlMeansDenoisingColored` did internally — separate strength for luminance vs. color), denoises the L/a/b planes independently in full float32 via `denoise_nl_means`, then converts back to RGB; `patch_size=7` / `patch_distance=10` mirror the old `templateWindowSize=7` / `searchWindowSize=21`. The `nlm_h`/`nlm_h_color` sliders (originally calibrated for OpenCV's 0–255-scale internals) are re-scaled by best-effort approximation rather than a verified pixel-exact equivalence, so perceptual strength may shift slightly from pre-v1.3.1 behavior — worth re-tuning by eye after upgrading.
    - *New dependency*: `scikit-image` is now an **optional** dependency (`pip install scikit-image`). If it's not installed, `"quality"` mode automatically falls back to the old pre-v1.3.1 behavior (internal 8-bit quantization via OpenCV) — nothing breaks, a console message notes the fallback, and the precision bottleneck simply returns for that one mode.
    - *New trade-off, disclosed*: denoising L/a/b as three separate full-resolution passes is slower than OpenCV's single combined-channel call. Local benchmark on a 256×256 synthetic image: ~0.48s (skimage/float32 path) vs. ~0.22s (old 8-bit fallback path) — roughly 2x slower at this size; the gap will scale with image resolution. This is expected given `"quality"` was already documented as being for final export rather than live preview.
    - *Verified*: re-ran the synthetic gradient/noise test through the new path — output stayed `float32` with far more than 256 unique tonal values (no 8-bit quantization detected), noise standard deviation dropped relative to the input (denoising is actually happening, not a no-op), and toggling `HAS_SKIMAGE` off reproduced the old capped-at-256-values / all-divisible-by-257 fallback behavior correctly.
    - *Packaging note (PyInstaller/.exe builds)*: shipping this fix in a bundled .exe needs extra PyInstaller flags beyond the base build — `scikit-image` uses `lazy_loader` for dynamic submodule imports (missed by PyInstaller's static analysis) and checks dependency versions via `importlib.metadata` at import time, so both need to be explicitly collected: `--collect-all skimage`, plus `--copy-metadata scikit-image`, `--copy-metadata imageio`, `--copy-metadata networkx`, `--copy-metadata lazy-loader` (its `scipy`/`numpy`/`pillow` metadata was already covered by the existing build command). This is only needed if the .exe should carry the full-precision `"quality"` path; skipping it is fine too — the app already falls back to the pre-v1.3.1 behavior gracefully if `scikit-image` isn't present in the built environment. Also note the build machine needs `scikit-image` actually `pip install`-ed beforehand, since `--collect-all` can't bundle a package it can't find.

### Track B — Maintenance (independent, carried findings)

#### Medium priority

- **Status bar GPU readout still doesn't have the DirectML fallback that the monitor panel got in v1.2.1** — `get_status_bar_html()` and `get_system_stats_html()` have near-identical GPU-detection logic, but only `get_system_stats_html()` received the v1.2.1 fix that shows "DirectML acceleration active" instead of going blank/`N/A` when `GPUtil` can't find an NVIDIA card. `get_status_bar_html()` still falls straight to `gpu_str = "N/A"` for AMD/Intel users. This has been listed as an unaddressed known limitation in both the v1.2.1 and v1.2.2 release notes; reading the code confirms the specific reason — the fix was applied to one function and not the other structurally-identical one.
  - *Effort*: low — same fallback logic already exists in `get_system_stats_html()` and can likely be adapted directly.

  **✅ Resolution (v1.3.2)**: `get_status_bar_html()` now runs the same three-tier GPU lookup as `get_system_stats_html()` — real `GPUtil` reading first, then (if `USE_GPU` and running on the DirectML backend) a "DirectML active" hint instead of blank/`N/A`, then a CUDA `torch.cuda.memory_allocated()` fallback. AMD/Intel users now see a status-bar hint instead of a silent `N/A`.

- **Cosmic Clarity Sharpen support** — carried over from the v1.2.2 plan's own follow-up note. `SetiAstroCosmicClarity.py` (Sharpen) uses the same fixed `input/`/`output/` folder convention as the already-supported Denoise tool, but with a different output suffix (`_sharpened` instead of `_denoised`). A near-duplicate of `run_cosmic_clarity_denoise()` would be needed.
  - *Effort*: low — mechanically similar to already-shipped code.

  **✅ Resolution (v1.3.2)**: `run_cosmic_clarity_sharpen()` added, mirroring `run_cosmic_clarity_denoise()`'s structure (same `exe_dir/input`, `exe_dir/output` protocol, same temp-file lifecycle and error handling), targeting the `_sharpened` output suffix.
  - *Verification caveat, disclosed*: unlike the Denoise function (which was verified against the official Cosmic Clarity source), the exact headless CLI flag names for the Sharpen tool have not been confirmed against its source — only the documented parameter *semantics* (Stellar/Non-Stellar/Both, strength, linear-image flag) are known. The function therefore doesn't gate on one specific flag name (as the Denoise version does with `--denoise_strength`) and instead just requires a non-empty `extra_args` string, to reduce (not eliminate) the risk of hanging on a GUI popup. Users should confirm actual flag names via `--help` or the tool's source before relying on this in production.
  - *Not yet done*: this function isn't wired into the UI/pipeline yet — `apply_clarity_and_sharpen()` has no `ext_path`/`ext_args` parameters or dropdown for it. Wiring it up can follow the same pattern `denoise()` uses to call `run_cosmic_clarity_denoise()`.

#### Low priority

- **Batch processing has no cancel/stop control** — `batch_process_fn()` is a generator that yields progress per file, which is good for showing live progress, but there's no interrupt check inside the per-file loop. For a large batch — especially one using an external tool per image — there's currently no way to stop a batch run once started short of closing the app.
  - *Effort*: low-to-medium, depends on how cleanly a stop flag can be threaded through Gradio's generator-based progress updates.

  **✅ Resolution (v1.3.2)**: added a "Stop Batch" button backed by a `gr.State({"stop": False})` flag (`state_batch_stop`), mutated in place rather than reassigned so the already-running generator (which captured the same dict object by reference at call time) can observe the change. `batch_process_fn()` checks the flag at the start of each file's loop iteration — a per-file boundary, not mid-file — so a stop request never leaves a half-written output file; the currently-processing file always finishes normally, and the final summary reports success/failed/not-processed counts distinctly for a stopped-early run. As defense in depth, the stop button also calls Gradio's built-in `cancels=[...]` on the batch click event.
  - *Verified*: ran a synthetic 4-file batch, set the stop flag after the first file completed, confirmed the generator halted with an accurate "1 succeeded, 0 failed, 3 not processed" summary and only the first file's output was written.

- **No control over which export formats get written** — `save_image_files()` unconditionally writes both a JPG and a 16-bit TIFF for every export; `export_fn()` and `batch_process_fn()` both call it without a format choice. For batch runs where only one format is actually needed, this doubles both export time and disk usage with no way to opt out.
  - *Effort*: low — likely just a checkbox/dropdown feeding into an existing function signature.

  **✅ Resolution (v1.3.2)**: `save_image_files()` now takes a `formats` argument (accepts `"jpg"`/`"jpeg"` and `"tif"`/`"tiff"`, case-insensitive; falls back to writing both if the set ends up empty). Both the Export tab and the Batch tab now have a "JPG / TIFF" checkbox group wired through `export_fn()` / `batch_process_fn()` into `save_image_files()`. Skipped formats return `None` in the `(out_jpg, out_tif)` tuple; call sites filter `None`s out of the returned file list.
  - *Verified*: tested `save_image_files()` with JPG-only, TIFF-only, and empty-format-set inputs — each produced exactly the expected file(s) on disk.

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

- ✅ **已完成（v1.3.0）** — ~~就算匯出成 16-bit TIFF，實際輸出仍然只有 8-bit 精度~~ — `finish_pipeline()` 在多個內部階段都轉成了 `uint8`：降噪那一步（`img8_tmp = (np.clip(img_work, 0, 1) * 255).astype(np.uint8)`）、Clarity/銳化那一步（`img8_cs = (np.clip(img_work, 0, 1) * 255).astype(np.uint8)`），以及 `boost_saturation()` 本身回傳的也是 `uint8`。`save_image_files()` 寫「16-bit TIFF」時，其實是把這個 8-bit 結果直接撐大（`img8.astype(np.uint16) * 257`），而不是真的用 16-bit 精度算出來的資料——TIFF 容器是 16-bit，但裡面的色調資訊不是。
  - *為什麼這對這套程式特別重要*：這套程式的核心操作——背景漸層扣除、動態範圍拉伸——都是對本來就很暗、對比很低的原始資料（星雲、天空背景）做很激進的色調處理。這正是 8-bit 量化在拉伸後最容易出現色階斷層（banding）的情境，比一般修圖更明顯。
  - *工作量*：還沒調查，先假設是中～大。這不是一行就能改完的修正，牽涉 `finish_pipeline()` 裡的每一步（降噪、Clarity/銳化、飽和度/增益），還有這些步驟裡目前假設輸入是 8-bit 的 OpenCV 呼叫（例如 `cv2.bilateralFilter`、`cv2.fastNlMeansDenoisingColored`，以及 Clarity/銳化用的 kernel 目前都是對 `uint8` 操作）。
  - *建議第一步*：先做一次範圍明確的可行性調查（不是直接改程式），確認 `denoise()`、`apply_clarity_and_sharpen()`、`boost_saturation()` 裡哪些可以在不整個重寫底層 OpenCV 呼叫的前提下改用 float32 或 uint16——有些 OpenCV 操作原生支援 16-bit，有些不支援，需要換演算法或自己實作。調查結果應該產出一份逐函式的可行性筆記，格式比照 v1.2.2 計畫處理 Cosmic Clarity 時「先查證、再動手」的做法。
  - *待確認的問題*：這應該做成永遠開啟的 pipeline 變更，還是一個可選的「高精度」模式——因為目前不知道用更高位元深度跑比較重的運算（例如 NLM 降噪）效能成本會有多大。

  **✅ 解決方案（v1.3.0）**：`finish_pipeline()` 現在全程用 float32 `[0,1]` 運算，不再像以前那樣分三次獨立量化成 `uint8`（降噪前、Clarity/銳化中間、函式結尾各一次）。牽動的函式包括：`denoise()`（`fast`／雙邊濾波模式改用 float32 原生跑，不再中途轉 8-bit）、`apply_clarity_and_sharpen()`（拿掉了 Clarity 算完、銳化前那次多餘的 8-bit 量化）、`boost_saturation()`（回傳值不再強制轉 uint8）、以及 `save_image_files()`（改用 `tifffile`，真正吃 float32 全精度資料寫出 `uint16` TIFF，不再是把 8-bit 結果 `×257` 撐大；JPEG 匯出才在這裡轉 8-bit，這是合理的，因為 JPEG 本身就沒有更高精度模式）。預覽函式（`update_preview_fn`、`layer_preview_fn`、`local_preview_fn`）只在各自呼叫端轉成 uint8 供畫面顯示，不影響匯出精度；`export_fn()`／`batch_process_fn()` 完全不用改，因為它們本來就是直接把 `result['main']` 丟給 `save_image_files()`。
  - *驗證方式*：用合成漸層/雜訊圖跑過 `denoise()`、完整 `run_pipeline()`、`export_fn()`、`batch_process_fn()`，確認輸出的 TIFF 是真正的 `uint16`、色階數遠超過 256 階，且數值不是每個都能被 257 整除（能整除代表還是舊的偽 16-bit，也就是把 8-bit 結果單純撐大）。也確認了 RGB 色彩順序在 JPG／TIFF 兩種輸出裡都正確，且預覽路徑的 dtype/shape 沒有被這次改動破壞。
  - **✅ 後續修復（v1.3.1）**：上面提到的 `"quality"` 模式限制，現在已經補上了。`denoise()` 改用 `skimage.restoration.denoise_nl_means`，原生支援 float32，內部那次 8-bit 量化已經拿掉。做法是先轉到 CIELAB（跟 `cv2.fastNlMeansDenoisingColored` 原本內部做的事一樣——亮度跟色度分開套用不同強度），L/a/b 三個通道各自在全 float32 精度下用 `denoise_nl_means` 降噪，再轉回 RGB；`patch_size=7`／`patch_distance=10` 對應原本的 `templateWindowSize=7`／`searchWindowSize=21`。`nlm_h`／`nlm_h_color` 滑桿（原本是針對 OpenCV 內部 0–255 尺度校準的）這次是用「盡力而為」的近似換算，不是逐像素驗證過的精確等價，升級後降噪的感覺可能跟以前有些微落差，建議肉眼微調。
    - *新增依賴*：`scikit-image` 現在是**選用**依賴（`pip install scikit-image`）。沒裝的話，`"quality"` 模式會自動退回 v1.3.1 之前的行為（內部量化成 8-bit 再跑 OpenCV）——不會壞掉，主控台會印出退回訊息，只是那一步的精度瓶頸會回來。
    - *新的取捨，主動揭露*：L/a/b 三個通道分開跑、各自都是全解析度，會比 OpenCV 原本單次合併通道處理慢。在本機用 256×256 合成圖測試：skimage/float32 路徑約 0.48 秒，舊版 8-bit fallback 路徑約 0.22 秒——這個尺寸大概慢了兩倍，圖越大差距會越明顯。這算是預期中的取捨，畢竟 `"quality"` 模式本來就被標註為「適合最終匯出而非即時預覽」。
    - *驗證方式*：用同一組合成漸層/雜訊圖重新跑過新路徑——輸出維持 `float32`、色階數遠超過 256（沒有偵測到 8-bit 量化）、降噪後的標準差確實比輸入低（代表真的有在降噪，不是空跑），並且手動關掉 `HAS_SKIMAGE` 後，確認能正確重現舊版「色階數封頂在 256 以內、且都能被 257 整除」的 fallback 行為。
    - *打包備註（PyInstaller／.exe）*：要讓打包出來的 .exe 也帶有這個修正，除了原本的打包指令外還要額外加參數——`scikit-image` 用 `lazy_loader` 做動態子模組載入（PyInstaller 靜態分析抓不到），import 時又會透過 `importlib.metadata` 檢查相依套件版本，所以兩邊都要明確收進去：`--collect-all skimage`，加上 `--copy-metadata scikit-image`、`--copy-metadata imageio`、`--copy-metadata networkx`、`--copy-metadata lazy-loader`（`scipy`/`numpy`/`pillow` 的 metadata 原本的打包指令就有涵蓋了，不用重複加）。這只有在希望 .exe 版本也保留全精度 `"quality"` 路徑時才需要；不加也沒關係，只要打包環境裡沒有 `scikit-image`，程式會照樣優雅退回 v1.3.1 之前的行為。另外打包機器上要先實際 `pip install scikit-image` 過，`--collect-all` 才抓得到東西可以打包。

### B 軌 — 維護項目（各自獨立，屬於延續發現）

#### 中優先

- **狀態列的 GPU 顯示，還沒補上監控面板在 v1.2.1 拿到的 DirectML fallback** — `get_status_bar_html()` 跟 `get_system_stats_html()` 的 GPU 偵測邏輯幾乎一模一樣，但只有 `get_system_stats_html()` 拿到 v1.2.1 的修正（`GPUtil` 找不到 NVIDIA 顯卡時顯示「DirectML 加速中」，而不是空白/`N/A`）。`get_status_bar_html()` 目前對 AMD/Intel 使用者還是直接落到 `gpu_str = "N/A"`。這件事在 v1.2.1 跟 v1.2.2 的發版說明裡都被列為「未處理的已知限制」；讀了程式碼後可以確認具體原因——修正只套用到其中一個函式，另一個結構幾乎一樣的函式沒有跟著改。
  - *工作量*：低——`get_system_stats_html()` 裡已經有現成的 fallback 邏輯，應該可以直接搬過去用。

  **✅ 解決方案（v1.3.2）**：`get_status_bar_html()` 現在跟 `get_system_stats_html()` 走同一套三層判斷順序——先試真的 `GPUtil` 讀值，找不到的話（`USE_GPU` 且走 DirectML 後端時）顯示「DirectML 加速中」而不是空白/`N/A`，再不行才退回 CUDA 的 `torch.cuda.memory_allocated()`。AMD/Intel 使用者現在狀態列會看到提示，而不是靜默顯示 `N/A`。

- **Cosmic Clarity Sharpen 支援** — 延續自 v1.2.2 計畫自己點名的後續項目。`SetiAstroCosmicClarity.py`（Sharpen）用的是跟已支援的 Denoise 工具一樣的固定 `input/`/`output/` 資料夾慣例，只是輸出檔名字尾不同（`_sharpened` 而不是 `_denoised`）。需要一支跟 `run_cosmic_clarity_denoise()` 幾乎一樣的函式。
  - *工作量*：低——跟已經上線的程式碼機制上很類似。

  **✅ 解決方案（v1.3.2）**：新增 `run_cosmic_clarity_sharpen()`，結構比照 `run_cosmic_clarity_denoise()`（同一套 `exe_dir/input`、`exe_dir/output` 協定，同樣的暫存檔生命週期與錯誤處理），鎖定 `_sharpened` 輸出檔名字尾。
  - *查證程度落差，主動揭露*：跟當初直接讀過官方原始碼驗證的 Denoise 版本不同，Sharpen 工具實際的 headless CLI 旗標名稱還沒有逐行對照原始碼確認——只確認了官方文件描述的參數語意（Stellar/Non-Stellar/Both、強度、是否為線性影像）。因此這支函式不像 Denoise 版本那樣鎖定檢查某個特定旗標（例如 `--denoise_strength`）是否存在，只要求 `extra_args` 不可留空，藉此降低（但無法完全排除）跳出 GUI 視窗卡住的風險。實際使用前建議先用 `--help` 或讀原始碼核對正確旗標名稱。
  - *尚未完成*：這支函式目前還沒接進 UI／pipeline——`apply_clarity_and_sharpen()` 還沒有對應的 `ext_path`/`ext_args` 參數或下拉選單。要串接的話可以比照 `denoise()` 呼叫 `run_cosmic_clarity_denoise()` 的做法。

#### 低優先

- **批次處理沒有中止/停止機制** — `batch_process_fn()` 是一個逐檔案 yield 進度的 generator，這點對即時顯示進度很好，但整個逐檔迴圈裡沒有任何中斷檢查點。批次量大的時候——尤其是每張圖都要呼叫外部工具的情境——目前除了關掉整個程式，沒有辦法在批次跑到一半時停下來。
  - *工作量*：低～中，取決於能不能乾淨地把一個「停止旗標」接進 Gradio 那套 generator 式進度更新機制裡。

  **✅ 解決方案（v1.3.2）**：新增「停止批次」按鈕，背後是一個 `gr.State({"stop": False})` 旗標（`state_batch_stop`）——按鈕原地修改這個 dict 的內容，而不是整個重新賦值，這樣正在跑的 generator（呼叫當下就已經收下同一個 dict 物件的參照）才看得到這次修改。`batch_process_fn()` 在每張圖片開始處理「前」檢查一次旗標——檢查點刻意放在單張圖片的邊界而非跑到一半強制中斷，確保不會留下寫到一半的殘缺輸出檔，已經在跑的那張圖片一定會正常跑完；最終摘要也會分別列出「已停止批次」情境下的成功/失敗/尚未處理張數。作為保底，停止按鈕同時也呼叫了 Gradio 內建的 `cancels=[...]` 機制。
  - *驗證方式*：用 4 張合成圖跑一次批次，在第 1 張處理完後設定停止旗標，確認 generator 正確停止並印出「成功 1 張、失敗 0 張、尚未處理 3 張」的摘要，且硬碟上只有第 1 張的輸出檔。

- **匯出格式沒有選擇權** — `save_image_files()` 不論如何都會同時寫出 JPG 跟 16-bit TIFF 兩個檔案，`export_fn()` 跟 `batch_process_fn()` 呼叫它時都沒有格式選項。如果批次處理其實只需要其中一種格式，匯出時間跟硬碟用量都會白白變成兩倍，而且沒有辦法關掉。
  - *工作量*：低——大概只需要一個勾選框/下拉選單，接進既有函式的參數即可。

  **✅ 解決方案（v1.3.2）**：`save_image_files()` 現在接受 `formats` 參數（支援 `"jpg"`/`"jpeg"`、`"tif"`/`"tiff"`，大小寫不拘；若最後篩選出的集合是空的，退回「兩種都寫」的舊行為保底）。匯出分頁跟批次分頁都新增了「JPG / TIFF」勾選框，經 `export_fn()`／`batch_process_fn()` 串進 `save_image_files()`。沒被選到的格式在回傳的 `(out_jpg, out_tif)` tuple 裡會是 `None`，呼叫端組檔案清單時會過濾掉 `None`。
  - *驗證方式*：分別用「只選 JPG」「只選 TIFF」「格式集合為空」三種情況測試 `save_image_files()`，確認硬碟上產生的檔案跟預期完全一致。

### 💬 意見回饋

跟以往一樣，這只是方向，不是承諾。A 軌建議先完成調查步驟，再決定要用哪種做法——調查完也有可能發現高位元深度的效能成本太高，不適合套用在即時預覽上，屆時可能需要「預覽精度」跟「匯出精度」分開處理，而不是整條 pipeline 一次到位改掉。B 軌的每一項都跟 A 軌、也跟彼此互相獨立，任何一項都可以單獨先出。
