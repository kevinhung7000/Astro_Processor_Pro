# 🛠️ Optimization Plan — v1.2.0

[English](#english) | [繁體中文](#繁體中文)

> 📝 本版已對照實際程式碼（`Astro_Processor_Pro.py` / `AstroProcessorPro_Setup.iss`）複核，於各項目下方以「> 複核備註」標註修正處。

---

## English

Candidate improvements for the next release, in suggested priority order. Not a strict commitment — just current direction, same as `ROADMAP.md`.

### High priority

- **~~Batch / export timing~~ ✅ Done (v1.2.0)** — the live preview already shows elapsed time per run (added in v1.1.0), but batch processing and full-resolution export still give no time feedback at all. Someone running batch on 40 images has no idea whether it'll take 2 minutes or 20. Add per-file timing to the batch progress log, and a total elapsed time to the export success message.
  - *Effort*: low — same `time.perf_counter()` pattern already used for preview, just applied to `batch_process_fn` and `export_fn`.
  - > **Code review note**: Confirmed straightforward. `batch_process_fn` is already a generator with `yield` per file, and `export_fn` has a single clear success-message return point — both are natural insertion points for `time.perf_counter()`. Effort estimate holds as-is; this is the best cost/value item on the list.
  - > **Implementation note (v1.2.0)**: Shipped. `export_fn` now reports total elapsed time on both success and failure. `batch_process_fn` now logs per-file elapsed time in each progress line, and — as a natural small extension beyond the original scope — also shows a running ETA (elapsed so far / estimated time remaining, based on the average per-file time observed so far). Flagging the ETA line as a scope addition in case it's more than wanted on screen; easy to drop if it makes the progress log too busy.

- **~~Config export/import — parameterized filename~~ ✅ Done (v1.2.0)** *(new, split out from the snapshot item below)*
  - `export_config_fn` currently always writes to a fixed filename `astro_config.json`, overwriting any previous export. This needs to change to a parameterized filename before named snapshot-to-file saving (below) can work without one snapshot silently overwriting another.
  - *Effort*: low — small, self-contained change; worth doing together with the snapshot work since it's a prerequisite for it.
  - > **Implementation note (v1.2.0)**: Shipped together with the snapshot work below. Added an optional "Export Filename" textbox; if left blank, falls back to the original `astro_config.json` default (unsanitized user input is filtered to safe characters — alphanumeric, underscore, hyphen, and CJK — to avoid path-traversal or invalid-filename issues).

### Medium priority

- **~~Named, savable snapshots~~ ✅ Done (v1.2.0)** — Snapshot A/B/C (v1.1.0) are session-only and unlabeled. Add an optional name field per snapshot, and a "save snapshot to file" / "load snapshot from file" pair (reusing the existing JSON config export/import plumbing) so a favorite setup like "backyard light pollution" survives a page refresh.
  - *Effort*: medium — mostly UI wiring, minimal new logic since JSON export/import already exists.
  - > **Code review note**: Current `save_snapshot_fn` / `load_snapshot_fn` are indeed pure `gr.State`, session-only — matches the description exactly. But this item depends on the fixed-filename issue above being fixed first, otherwise saving a second named snapshot will overwrite the first one on disk. Recommend doing the filename parameterization item together with this one in the same change.
  - > **Implementation note (v1.2.0)**: Shipped. Added a "Target Slot" (A/B/C) selector, an optional "Snapshot Name" field, a "Save Snapshot to File" button, and a "Load Snapshot from File" upload. Save writes a JSON payload (`snapshot_name`, `slot`, `saved_at`, `params`) to a name-derived filename (e.g. `astro_snapshot_backyard.json`); load reads it back into the chosen A/B/C slot's `gr.State` and status label — the user still needs to click "Apply A/B/C" afterward to push it onto the sliders, matching how loading already worked for in-session snapshots. Also made `import_config_fn` and the new snapshot loader tolerant of each other's file format (both a bare params dict and a `{"params": {...}}` wrapper are accepted), so a snapshot file and a plain config export are interchangeable.

- **~~Collapse Presets / Snapshots into accordions~~ ✅ Done (v1.2.0)** *(new, small UI follow-up to the two items above)* — Beginner Presets and Parameter Snapshots were plain, always-open `gr.Group` blocks sitting above the numbered parameter accordions (`1️⃣ Background Gradient Removal`, etc.), taking up permanent vertical space even for users who never touch them after their first session.
  - *Effort*: low — pure UI change, no pipeline logic involved; same `gr.Accordion(..., open=False)` pattern already used everywhere else in the parameter panel.
  - > **Implementation note (v1.2.0)**: Shipped. Both sections are now collapsible accordions — `0️⃣a 🎛️ Beginner Presets` and `0️⃣b 📌 Parameter Snapshots` — collapsed by default (`open=False`), matching the existing numbered accordions rather than sitting permanently expanded. Numbered `0️⃣a`/`0️⃣b` to signal they run before the `1️⃣`+ processing steps. Language switching (zh/en) for both accordion titles and their contents continues to work as before.

- **~~Manual region selection for local stretch~~ ✅ Done (v1.2.0)** — Auto Localized Stretch (v1.1.0) detects textured regions automatically, but if it misses a faint target or over-boosts a noisy patch, there's currently no way to override it. Add an optional manual mask: let the user draw/select a region (e.g. via the crop-preview coordinates already used for local preview) that's always boosted regardless of what auto-detection finds.
  - *Effort*: ~~medium-high~~ **high** — needs a way to capture a user-drawn region and merge it with the existing per-pixel mask; the local-preview crop selector is a natural starting point.
  - > **Code review note**: `local_x_pct` / `local_y_pct` / `local_crop_px` do exist and are a reasonable starting point, but today they only drive a **preview crop overlay** (`_update_overview_only`), not the pipeline's actual pixel mask — these are two separate data paths. Turning a user-selected rectangle into something that merges with the existing auto-detected per-pixel mask requires: (1) converting percentage-based preview coordinates to full-resolution pixel coordinates, (2) feathering the edges of the manual region so it doesn't create a hard, visible boundary, and (3) deciding the merge policy with the auto mask (union? user override wins? adjustable blend weight?). The UI part is simple; the masking/merge logic is the real work. Bumping effort from medium-high to **high**.
  - > **Implementation note (v1.2.0)**: Shipped, in `3️⃣b Auto Localized Stretch`, below the auto-detection controls. Ended up using dedicated percentage-based sliders (X/Y center, width/height, all as % of image dimensions) rather than reusing `local_x_pct`/`local_y_pct`/`local_crop_px` directly — those drive the separate ROI-preview crop tool (a different data path, per the code review note above), and percentage-based sizing (rather than the ROI tool's absolute `crop_px`) means the region works correctly at any resolution — full export, thumbnail preview, or the ROI-preview crop — without needing `preview_scale` conversion. Resolved the three open design questions as: (1) percent-to-pixel conversion happens fresh against whatever image the pipeline is currently running on; (2) edge feathering via `gaussian_filter`, with a "Edge Feather (%)" slider sized relative to the region's shorter side; (3) merge policy is **union with adjustable weight** — `final_mask = max(auto_mask, manual_mask)`, where the manual region's mask value is set by a "Region Boost Weight" slider (0–1) rather than always forcing 1.0, so it can add a gentle nudge or a full override. The manual region also works independently of the auto-detection toggle (auto can stay off and the manual region alone still boosts). No dedicated visual overlay was added for the manual rectangle specifically — the existing "Auto Localized Stretch mask" preview already shows the *merged* mask (auto + manual) as a red highlight with a yellow contour, so the manual region's actual coverage is visible there without extra UI.
  - > **Follow-up (v1.2.0)**: Added a "Region Shape" toggle (rectangle / ellipse) next to the enable checkbox. Rectangle keeps the original hard-edged-box-then-feather behavior; ellipse computes the region as a normalized-distance mask (`((x-x0)/(rw/2))² + ((y-y0)/(rh/2))² ≤ 1`) before the same `gaussian_filter` feathering is applied, so both shapes share identical edge-softening behavior. No separate "circle" option was added — setting width % = height % on the ellipse shape already gives a perfect circle, so a third choice would have been redundant. `build_manual_target_mask()` and `apply_local_target_boost()` both take the new `shape` parameter (default `"rectangle"`, so existing configs/snapshots without the field keep working unchanged); the merged-mask preview overlay needed no changes since it already visualizes whatever the merged per-pixel mask happens to contain.

### Low priority

- **~~ML-based denoising~~ ✅ Done（v1.2.0） (implemented)** — quality mode (v1.1.0) was Non-local Means, not a learned denoiser. A real ML option (e.g. a small pretrained model) would help with noisier subs, but has real costs: a noticeably larger installer, a new inference dependency, and slow CPU-only performance for users without a GPU. Worth scoping only after the above are done.
  - *Effort*: high (originally) — model selection, packaging size/dependency tradeoffs, and a CPU fallback story all need deciding before writing any code.
  - > **Code review note**: `AstroProcessorPro_Setup.iss` already notes the installer bundles `torch` for GPU-accelerated background estimation. If the shipped build already includes torch, the incremental dependency cost for an ML denoiser is mainly model weight size, not an additional runtime dependency. Confirm the distributed build variant before re-scoping priority.
  - > **Implementation note (implemented)**: Implemented the "方案 A: 外部工具介接" path and integrated it into the pipeline and UI. Core changes:
    - Added run_external_image_tool(): a shared subprocess wrapper that handles img01 → 16-bit TIFF temp file → call external CLI → read result → convert back to img01. Supports `{input}`/`{output}` placeholders or the common "executable input output" convention, with timeout, error catching, and temp-file cleanup.
    - denoise() now supports mode="external"; process_stars() now supports mode="external". On failure the steps are skipped and the original image is returned (pipeline continues without aborting and without silently falling back to fast/quality).
    - denoise_mode and star_mode each gained an "external" option.
    - The denoise/star accordions each have two new fields: external tool executable path and extra CLI args; the field info text clearly states licensing/activation responsibility remains with the user.
    - New params (`denoise_ext_path/args`, `star_ext_path/args`) appended at the end of `PARAM_NAMES`/`DEFAULTS`/`PARAM_COMPONENTS`, avoiding changes to existing positional indexes like `DEFAULTS[24]`/`DEFAULTS[43]`.
    - Chinese/English labels completed; config export/import, snapshot save/load, Reset and Preset all remain compatible (they iterate `PARAM_NAMES`, so no extra handling was needed).
  - > **Testing note**: Loaded the full Gradio UI and exercised scenarios: not-set path, executable-not-found, tool produced no output, successful call and read-back — all behaved as expected per design.
  - > **Known limitations**: When preview switches to external mode it will call the external tool and performance depends on that external program. Current verification covered "call failure" and "successful round-trip file copy" paths; actual compatibility with real tools' CLI (e.g., NoiseXTerminator / StarXTerminator) has not yet been validated on-device and may require per-tool argument tweaks. Also beware GPU contention if both background estimation and external tool use the GPU — serialize those steps in batch if needed.

### 💬 Feedback

As with the main roadmap, this is a direction, not a commitment — open to reordering based on what actually turns out to matter for real usage.

---

## 繁體中文

下一版候選優化項目，依建議優先度排序。跟 `ROADMAP.md` 一樣，不是硬性承諾，只是目前的方向。

### 高優先

- **~~批次 / 匯出顯示耗時~~ ✅ 已完成（v1.2.0）** — 即時預覽（v1.1.0）已經會顯示每次運算花多久，但批次處理和全解析度匯出目前完全沒有時間回饋。跑 40 張批次的人根本不知道要等 2 分鐘還是 20 分鐘。要在批次進度訊息中加上每張圖的耗時，匯出成功訊息也加上總耗時。
  - *工作量*：低 — 沿用 v1.1.0 即時預覽已經在用的 `time.perf_counter()` 模式，套用到 `batch_process_fn` 和 `export_fn` 即可。
  - > **複核備註**：確認可行且估計準確。`batch_process_fn` 本來就是逐張 `yield` 的 generator，`export_fn` 也只有單一個明確的成功訊息回傳點，兩者都是插入 `time.perf_counter()` 的天然位置。工作量估計維持不變，是清單裡 CP 值最高的一項。
  - > **實作備註（v1.2.0）**：已完成。`export_fn` 在成功與失敗兩種情況下都會回報總耗時。`batch_process_fn` 每一行進度都加上該張的耗時，並且多做了一點原計畫沒明講的延伸：加了一行即時 ETA（已耗時／預估剩餘時間，依目前已處理張數的平均耗時推算）。這個 ETA 屬於範圍外的小擴充，先在這裡註記一下；如果畫面覺得太擠不需要，之後可以直接拿掉。

- **~~參數配置匯出/匯入 — 檔名參數化~~ ✅ 已完成（v1.2.0）** *（新增，從下方快照項目拆分出來）*
  - `export_config_fn` 目前永遠寫入固定檔名 `astro_config.json`，每次匯出都會覆蓋前一次的結果。這個問題必須先解決，下面的「具名快照存檔」功能才不會發生「存第二個快照時把第一個覆蓋掉」的狀況。
  - *工作量*：低 — 改動範圍小且獨立，建議跟下面的快照項目一起做，因為它是快照存檔功能的前置條件。
  - > **實作備註（v1.2.0）**：已與下面的快照功能一併完成。新增了一個選填的「匯出檔名」欄位；留空則沿用原本的 `astro_config.json` 預設檔名（使用者輸入的文字會先過濾成安全字元——英數字、底線、連字號、中日韓文字——避免路徑穿越或不合法檔名的問題）。

### 中優先

- **~~快照可命名並存檔~~ ✅ 已完成（v1.2.0）** — 快照 A/B/C（v1.1.0）目前僅暫存於單次瀏覽器工作階段，也沒有名稱。可以加上選填的命名欄位，並提供「快照存成檔案 / 從檔案載入快照」（沿用現有的 JSON 參數配置匯出/匯入機制），這樣像「我家後院光害設定」這類常用組合就不會因為重新整理頁面而消失。
  - *工作量*：中 — 主要是介面串接，邏輯大部分可重用既有的 JSON 匯出/匯入。
  - > **複核備註**：現有 `save_snapshot_fn` / `load_snapshot_fn` 確實純粹用 `gr.State`，只存在單次工作階段，跟描述完全吻合。但這項功能必須先解決上面提到的固定檔名問題，否則存第二個具名快照就會覆蓋磁碟上的第一個。建議跟「檔名參數化」這項合併在同一次改動裡一起做。
  - > **實作備註（v1.2.0）**：已完成。新增「目標快照」（A/B/C）選擇器、選填的「快照名稱」欄位、「快照存成檔案」按鈕，以及「從檔案載入快照」上傳欄位。存檔會把 JSON（含 `snapshot_name`、`slot`、`saved_at`、`params`）寫到依名稱產生的檔名（例如 `astro_snapshot_後院光害.json`）；載入則讀回指定 A/B/C 那一格的 `gr.State` 與狀態文字——使用者仍需按「套用 A/B/C」才會實際套用到滑桿，跟原本工作階段內的快照套用方式一致。另外也讓 `import_config_fn` 跟新的快照載入功能能互相相容彼此的檔案格式（純參數字典、或包了一層 `{"params": {...}}` 的格式都能讀），所以快照檔跟一般的參數匯出檔可以互通使用。

- **~~新手預設集 / 參數快照收合成 Accordion~~ ✅ 已完成（v1.2.0）** *（新增，上面兩項的小型 UI 後續調整）* — 「新手預設集」和「參數快照」原本是固定展開、不可收合的 `gr.Group` 區塊，放在編號的參數 accordion（`1️⃣ 背景漸層去除`⋯等）上方，即使使用者第一次設定完就不再需要，也一直佔用畫面空間。
  - *工作量*：低 — 純 UI 調整，不涉及任何運算邏輯；沿用參數面板其他地方已經在用的 `gr.Accordion(..., open=False)` 模式即可。
  - > **實作備註（v1.2.0）**：已完成。兩個區塊都改成可收合的 accordion——`0️⃣a 🎛️ 新手預設集`與`0️⃣b 📌 參數快照`——預設為收合狀態（`open=False`），跟其他編號 accordion 的預設行為一致，不再永遠佔著展開。編號用 `0️⃣a`／`0️⃣b`，表示這兩項是在 `1️⃣`⋯等處理步驟「之前」執行的設定動作。中英文切換對這兩個 accordion 的標題與內容依然正常運作。

- **~~手動框選局部拉伸區域~~ ✅ 已完成（v1.2.0）** — 自動局部拉伸（v1.1.0）會自動偵測有結構的區域，但如果漏掉較弱的目標、或誤把雜訊區加強，目前沒有辦法手動覆蓋。可以加上選填的手動遮罩：讓使用者框選（例如沿用局部預覽已經有的裁切座標機制）一塊區域，不論自動偵測結果如何，該區域一律加強。
  - *工作量*：~~中高~~ **高** — 需要一套讓使用者框選區域、並與現有逐像素遮罩合併的機制；局部預覽的裁切選取器是很自然的起點。
  - > **複核備註**：`local_x_pct` / `local_y_pct` / `local_crop_px` 確實存在，是合理的起點，但目前它們只驅動**預覽裁切框**（`_update_overview_only`），並不是 pipeline 實際套用的逐像素遮罩——這是兩條獨立的資料流。要把使用者框選的矩形變成能跟現有自動遮罩合併的東西，需要處理：(1) 百分比座標轉換成全解析度像素座標、(2) 手動區域邊緣羽化（否則邊界會很生硬明顯）、(3) 與自動遮罩的合併策略（取聯集？使用者優先？可調混合權重？）。UI 部分簡單，遮罩合併演算法才是真正的工作量所在。工作量估計由中高上修為**高**。
  - > **實作備註（v1.2.0）**：已完成，位於 `3️⃣b 自動局部拉伸`accordion 內、自動偵測參數下方。最後選擇用獨立的百分比滑桿（X/Y 中心位置、寬/高，皆以整張圖尺寸的百分比表示），而不是直接沿用 `local_x_pct`／`local_y_pct`／`local_crop_px`——如前面複核備註所說，那組滑桿驅動的是另一條資料流（局部預覽的裁切工具）；而且用百分比表示大小（而非局部預覽裁切工具用的絕對像素 `crop_px`），可以讓這個手動區域在任何解析度下都正確運作——全解析度匯出、縮圖預覽、或局部預覽裁切——都不需要額外的 `preview_scale` 換算。原本待決的三個設計問題確認如下：(1) 百分比轉像素座標，是針對 pipeline 當下實際處理的那張圖即時換算；(2) 邊緣羽化用 `gaussian_filter`，並提供「邊緣羽化 (%)」滑桿，羽化程度依框選區域短邊的百分比計算；(3) 合併策略採**取聯集 + 可調權重**——`final_mask = max(自動遮罩, 手動遮罩)`，手動區域的遮罩值由「框選區域加強權重」滑桿（0～1）決定，而非總是強制設為 1.0，所以可以只是輕微加強，也可以是完全覆蓋。手動框選也可以獨立於自動偵測開關運作（自動偵測關閉時，手動區域仍可單獨生效）。沒有另外為手動矩形做專屬的視覺化疊圖——現有的「自動局部拉伸遮罩」預覽本來就會顯示*合併後*的遮罩（自動+手動）疊上半透明紅色與黃色邊界線，因此手動區域實際涵蓋的範圍在那裡就能直接看到，不需要額外的 UI。
  - > **後續調整（v1.2.0）**：在啟用勾選框旁邊加了「框選形狀」切換（矩形／橢圓）。矩形沿用原本「先算硬邊方框、再羽化」的行為；橢圓則改用正規化距離公式（`((x-x0)/(rw/2))² + ((y-y0)/(rh/2))² ≤ 1`）算出範圍，再套用同一套 `gaussian_filter` 羽化，兩種形狀的邊緣柔化行為完全一致。沒有另外做「圓形」選項——橢圓形狀下把寬度%設成跟高度%相等，本來就會是正圓，多一個選項只是重複。`build_manual_target_mask()`與`apply_local_target_boost()`都新增了`shape`參數（預設值`"rectangle"`，所以沒有這個欄位的舊設定檔／快照仍可正常載入，行為不變）；合併遮罩的預覽疊圖不需要改動，因為它本來就是照實際合併後的逐像素遮罩畫出來的，換成橢圓也一樣能正確顯示。

### 低優先

- **~~ML 降噪~~ ✅ 已完成（v1.2.0）（已實作）** — quality 模式（v1.1.0）目前是 Non-local Means，不是學習式降噪。真正的 ML 選項（例如小型預訓練模型）對噪訊較重的疊圖會有幫助，但代價也真實存在：安裝檔會明顯變大、多一個推論相依套件、沒有 GPU 的使用者純 CPU 跑會很慢。建議在前面項目完成後再進一步評估。
  - *工作量*：原先估為高 — 模型選型、封裝大小/相依套件取捨、以及無 GPU 時的退路方案，都需先決定。
  - > **複核備註**：`AstroProcessorPro_Setup.iss` 已註明安裝檔為了 GPU 背景估計加速而可能包含 `torch`。如果發行版本來就含 `torch`，那麼新增 ML 降噪的增量成本主要是權重檔案大小，而非新相依套件。請先確認實際出貨版本再調整優先度。
  - > **實作備註（已實作）**：已將「方案 A：外部工具介接」實際寫入 `Astro_Processor_Pro.py` 並整合到 UI 與參數設定中。主要改動：
    - 新增 `run_external_image_tool()`：共用的 subprocess 介接函式，負責 img01 → 16-bit TIFF 暫存檔 → 呼叫外部 CLI → 讀回結果 → 轉回 img01，支援 `{input}`/`{output}` 佔位符或「執行檔 輸入檔 輸出檔」的慣例，帶 timeout、錯誤攔截與暫存檔清理。
    - `denoise()` 新增 `mode="external"`，`process_stars()` 新增 `mode="external"`；兩者在失敗時皆跳過該步驟並回傳原圖，pipeline 不會中斷，也不會在 UI 上靜默改用 fast/quality。
    - `denoise_mode` 與 `star_mode` 各加入 `"external"` 選項。
    - 對應的降噪/去星 accordion 各新增兩個欄位：外部工具執行檔路徑與額外命令列參數；欄位說明文字已寫明授權／啟用責任在使用者端。
    - 新參數 (`denoise_ext_path/args`, `star_ext_path/args`) 已附加在 `PARAM_NAMES`/`DEFAULTS`/`PARAM_COMPONENTS` 尾端，不影響現有以索引取值的行為。
    - 中英文標籤、設定檔匯出/匯入、快照儲存/載入、Reset、Preset 等均已自動相容（原本就是通用的 `PARAM_NAMES` 迴圈）。
  - > **測試備註**：已啟動完整 Gradio UI，並測試多組情境：未設定路徑、找不到執行檔、外部工具沒產生輸出、成功呼叫並回讀結果；行為符合預期設計。
  - > **已知限制**：即時預覽切到 external 模式時也會呼叫外部工具，速度取決於外部程式。已驗證呼叫失敗及成功回傳的路徑，但尚未在實機上驗證 NoiseXTerminator/StarXTerminator 等工具的 CLI 語法是否完全吻合；若要支援特定第三方工具，可能還需針對其 CLI 參數或輸入輸出格式做微調。注意若外部工具與本地 GPU 都使用同一顯卡，可能發生資源競爭，建議在批次模式下序列化那些步驟以避免 OOM 或性能下降。

### 💬 意見回饋

跟主 roadmap 一樣，這只是方向不是承諾，如果實際使用後發現優先順序不對，歡迎隨時調整。
