# 🗺️ Roadmap & Known Issues

[English](#english) | [繁體中文](#繁體中文)

---

## English

This document tracks known issues and possible future improvements for Astro Processor Pro. Contributions and suggestions are welcome — feel free to open an issue or PR.

### 🐞 Known Issues

- No local/selective stretching for specific nebula regions (whole-image stretch only).
- Denoising uses a basic bilateral filter, not an ML-based denoiser.

### 🚧 Planned Improvements

Roughly ordered by priority (not a strict commitment — just current direction):

**High priority**
- [x] Complete English translation coverage — fixed a handful of components (close button, status placeholders) that weren't switching to `EN`; status messages now only switch language while they're still showing the default placeholder text, so real results aren't overwritten
- [x] Batch processing — new "Batch Processing" tab applies the current parameter set to every image in the source folder, with per-file progress and a success/failure summary
- [x] Multiple parameter snapshots — new Snapshot A/B/C save & apply buttons let you switch between candidate settings instantly (session-only, cleared on page refresh)
- [x] Presets — new "Milky Way" / "Nebula" / "Heavy Light Pollution" preset buttons give beginners a sensible starting point instead of the generic defaults

**Medium priority**
- [x] Auto-detected localized stretching — new "3️⃣b Auto Localized Stretch" section. Detects textured regions (nebulosity, galaxy structure) via large-radius local-contrast analysis and boosts contrast only there, leaving flat sky background untouched, instead of stretching the whole image uniformly. Off by default; adjustable strength/radius/sensitivity. It's a continuous per-pixel mask, not a single-region detector, so multiple separate nebula/galaxy patches in the same frame are all boosted simultaneously. A new preview image in the "Mask & Starless Layers" tab visualizes exactly what got detected: a translucent red overlay shows boost strength, and a yellow contour line traces the boundary of each detected region (works for any number of disjoint regions).
- [x] Stronger denoising option — the Denoise section now has a "fast" (bilateral, unchanged default) vs "quality" mode using `fastNlMeansDenoisingColored`, with its own strength sliders. Quality mode is noticeably slower, so it's best suited to final export rather than live preview.

**Low priority**
- [x] Performance profiling for live preview responsiveness — the status bar had an unused `proc_time` slot that was never actually measured. The live preview, layer preview, and local (crop) preview functions now time their pipeline run with `time.perf_counter()` and show it directly in the status message (e.g. "Preview and RGB histogram updated (Thumbnail processing, 0.42s)"), so slow parameter combinations are now visible rather than silent.
- [x] More inline guidance/tooltips for beginner-friendly default values — added `info=` tooltips to the ~25 sliders that were missing them, mostly in the star/cluster detection and denoise sections (e.g. max area, dilation amount, feather radius, crop position/size), so every adjustable value now explains what it does.

### 🧹 Code Quality Notes

Minor, non-urgent items found during a code review — informational only, none of these affect normal single-user usage.

**Fixed**
- [x] `UI_TRANSLATIONS` defined the `"compare_mode"` key twice (once as a plain string, once as a label/info tuple), with the second silently overwriting the first. Removed the dead first definition; behavior is unchanged since the second one was always the one actually used.
- [x] `_BG_CACHE` used to keep only the most recently processed image, clearing itself entirely on every cache miss. It's now a bounded LRU cache (`OrderedDict`, 8 entries by default) — single-user behavior is unchanged, but concurrent sessions processing different images no longer evict each other's cached background estimate.
- [x] GPU background estimation used to permanently set `USE_GPU = False` after a single failure. It now tracks consecutive failures and only disables GPU for good after 3 in a row; a one-off failure just falls back to CPU for that image, and the next image still retries GPU (the counter resets on any success).

### 💬 Feedback

This is a hobby project built to make simple stretching/post-processing easier for beginners — it's not trying to replace dedicated astro post-processing software. If something's confusing, broken, or you have an idea, please open an issue on GitHub.

---

## 繁體中文

這份文件記錄 Astro Processor Pro 目前已知的問題，以及未來可能的優化方向。歡迎大家提出建議，或直接開 issue / PR 討論。

### 🐞 已知問題

- 尚無針對星雲局部區域單獨拉伸的功能（目前僅支援全圖拉伸）。
- 降噪僅採用基本的雙邊濾波，並非 AI/ML 降噪。

### 🚧 未來優化方向

大致依優先度排序（非硬性承諾，僅代表目前的方向）：

**高優先**
- [x] 補齊英文翻譯 — 修正了幾個切到 `EN` 時不會跟著變的元件（關閉按鈕、狀態列佔位文字）；狀態訊息現在只有在仍顯示預設佔位字時才會跟著切換語言，不會洗掉已經顯示的真實處理結果
- [x] 批次處理 — 新增「批次處理」分頁，可將目前參數套用到來源資料夾內的每一張圖片，逐張顯示進度，並在結束後給出成功/失敗摘要
- [x] 多組參數快照 — 新增快照 A/B/C 的「儲存」「套用」按鈕，可即時在候選參數間切換比較（僅暫存於這次瀏覽器工作階段，重新整理頁面會清空）
- [x] 預設集（Presets）— 新增「銀河模式」「星雲模式」「重光害」三個預設按鈕，給新手一個比通用預設值更貼近情境的起點

**中優先**
- [x] 自動搜尋星雲/目標區域並局部拉伸 — 新增「3️⃣b 自動局部拉伸」區塊，用大半徑局部對比分析自動偵測有結構的區域（星雲、銀河塵埃帶等），只針對該區域加強對比，天空背景幾乎不受影響，取代原本全圖統一拉伸的做法。預設關閉，強度/半徑/靈敏度皆可調整。遮罩是逐像素的連續值，不是只認一個目標，畫面中同時有好幾片不相連的星雲/銀河結構也都會一起被抓到、一起加強。「遮罩 / 去星層」分頁新增一張預覽圖，可以實際看到抓到哪裡：半透明紅色代表加強力道，黃色線則描出每一片被偵測區域的邊界（有幾片就畫幾條，不限單一區域）。
- [x] 更強的降噪選項 — 降噪區塊新增「fast」（雙邊濾波，預設不變）與「quality」（`fastNlMeansDenoisingColored`）兩種模式，quality 模式有獨立的強度滑桿。quality 模式運算明顯較慢，較適合用在最終匯出而非即時預覽。

**低優先**
- [x] 即時預覽效能檢視與優化 — 狀態列原本就有 `proc_time` 欄位，但從未真正被計時過。現在即時預覽、圖層預覽、局部（裁切）預覽都會用 `time.perf_counter()` 計時，並直接顯示在狀態訊息裡（例如「預覽與 RGB 曲線已更新(縮圖運算，0.42s)」），讓比較慢的參數組合不再無聲無息。
- [x] 更多新手友善的滑桿說明文字/建議值 — 補齊約 25 個原本沒有 `info=` 提示的滑桿，主要集中在星點/星團偵測與降噪區塊（例如最大面積、外擴像素、羽化程度、裁切位置/大小等），現在每個可調參數都有說明。

### 🧹 程式碼品質備忘

以下是程式碼審查時發現的幾個小地方，都不影響單機一般使用。

**已修正**
- [x] `UI_TRANSLATIONS` 裡 `"compare_mode"` 這個 key 原本被定義了兩次（一次純字串、一次含 info 的 tuple），後面那次會覆蓋前面。已刪掉前面那份死碼；因為原本生效的就是後面那份，行為完全不變。
- [x] 全域 `_BG_CACHE` 原本只保留最近 1 張圖、每次 cache miss 就整個清空重存。現在改成有容量上限（預設 8 筆）的 LRU 快取（`OrderedDict`）：單機自用行為不變，但未來若多人同時連線、各自處理不同圖片，就不會再互相把對方剛算好的背景快取擠掉。
- [x] GPU 背景估計原本一次失敗就永久設 `USE_GPU = False`。現在改成累計「連續失敗次數」，連續失敗滿 3 次才會永久退回 CPU；單次偶發失敗只會讓那張圖退回 CPU，下一張圖仍會重試 GPU（只要中途成功一次，計數就會歸零）。

### 💬 意見回饋

這是一個興趣專案，目標是讓初學者能更輕鬆做簡單的拉伸/後製，並不是要取代專業的天文後製軟體。如果哪裡卡關、壞掉，或有想法，歡迎直接在 GitHub 開 issue 討論。
