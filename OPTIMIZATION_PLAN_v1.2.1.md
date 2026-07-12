# 🛠️ Optimization Plan — v1.2.1

[English](#english) | [繁體中文](#繁體中文)

> 📝 Scope: two tracks this release — **Track A** broadens v1.2.0's external-tool integration (denoise/star removal) to explicitly cover StarNet/StarNet2, RC-Astro NoiseXTerminator/StarXTerminator, Seti Astro Cosmic Clarity, and GraXpert; **Track B** is unrelated monitor-panel polish (auto-refresh, AMD/Intel GPU detection) that got folded into the same release by choice, not because the two are technically connected.

---

## English

Candidate improvements for the next release, in suggested priority order within each track. Not a strict commitment — same as `ROADMAP.md`.

### Track A — External Tool Integration

#### High priority

- **Tool profile presets for the external-tool fields** — right now the "extra arguments" textbox is free text, and each of the four target tools uses a different CLI convention (DeepSNR: `-i/-o` named flags; RC-Astro: subcommand + `--output`; legacy StarNet: positional args; GraXpert: `-cli -cmd ... -output` with an extension-less output name). Add a dropdown next to `denoise_ext_path`/`star_ext_path` — "DeepSNR / RC-Astro / GraXpert / StarNet (legacy) / Custom" — that auto-fills the extra-arguments field with the correct template when selected, instead of requiring the user to remember or hand-type each tool's syntax.
  - *Effort*: low-medium — pure UI convenience plus a small lookup table of known templates; no change to `run_external_image_tool()`'s core call path.
  - > **Research note**: Confirmed syntax so far — DeepSNR: `--input {input} --output {output}`; RC-Astro: `nxt {input} --output {output} --overwrite` / `sxt {input} --output {output} --overwrite` (single `rc-astro` executable, subcommand-based). GraXpert and legacy StarNet need the extension handling below before they fit the same template shape.

- **`{output_noext}` placeholder for tools that append their own extension** — GraXpert's `-output` flag takes a filename **without an extension**; GraXpert appends one itself. The current placeholder system assumes the caller controls the exact output filename (`{output}` → `output.tif`), which doesn't fit GraXpert's convention and would cause "output file not found" failures even on a successful run.
  - *Effort*: medium — `run_external_image_tool()` needs to (1) offer a `{output_noext}` placeholder alongside `{input}`/`{output}`, and (2) when it's used, search the temp directory for whatever file GraXpert actually wrote (glob on the stem) instead of checking a single hardcoded path.
  - > **Technical note**: Recommended template once this lands: `-cli -cmd denoising {input} -output {output_noext}`. Until this ships, GraXpert cannot be wired up through the existing `{output}`-only convention without the tool silently "failing" due to a filename mismatch — this is a prerequisite, not just a nice-to-have, for GraXpert support specifically.

- **Pause / resume live preview** — every slider `.release()` and toggle `.change()` currently calls `update_preview_fn` directly and unconditionally; there's no way to stop preview from re-running while adjusting several parameters in a row. This matters most for `mode="external"`, where each preview tick actually shells out to the external tool — repeatedly, on every single slider release — which is slow and, for paid/rate-limited tools like RC-Astro, wasteful.
  - *Effort*: low — add a `live_preview_enabled` checkbox and include it in `_PREV_IN`, the shared input list already passed to every existing `.release()`/`.change()` binding. `update_preview_fn` just needs an early-return (`gr.update()` no-ops on all outputs) when the checkbox is off — no need to touch the dozens of individual event-wiring lines (`sliders_for_release`, `toggles_for_change`, etc.), since they all already flow through `_PREV_IN`.
  - > **Design**: default on (current always-live behavior unchanged for anyone who doesn't touch the new checkbox). When re-enabled, fire one `update_preview_fn` call immediately via the checkbox's own `.change()` event, so the preview doesn't sit stale after resuming. Keep the pause state independent of the existing "layer preview" / "local preview" buttons — those already work as their own manual triggers and should still work normally while the main live preview is paused.

#### Medium priority

- **StarNet / StarNet2 — confirm current CLI flags before shipping a preset template** — the *legacy* StarNet++ CLI (still referenced in some Siril docs and forum posts) took **positional** arguments: `starnet++ <input.tif> [output.tif] [stride]`, defaulting the output to `starless.tiff` if omitted — a different shape from DeepSNR entirely, and from the generic "executable input output" fallback (works by coincidence for legacy StarNet, but not guaranteed for StarNet2). The *current* StarNet2 2.5.x release (ONNX-based, packaged on the same site and in the same style as DeepSNR, with a documented `--unscreen` flag) likely adopted the same `-i/-o` named-flag convention DeepSNR uses, but this has not been directly confirmed against StarNet's own CLI reference page.
  - *Effort*: low once confirmed — likely just another entry in the tool-profile table above.
  - > **Open item**: Run `starnet2 -h` (or equivalent `--help`) after download and record the actual flag list before adding a "StarNet2" preset; don't assume it mirrors DeepSNR without checking, even though it's the same vendor and packaging style.

- **RC-Astro NoiseXTerminator/StarXTerminator — full end-to-end verification** — syntax is documented (`rc-astro nxt/sxt <input> --output <output> --overwrite`), but not yet run successfully against a real license in this app.
  - *Effort*: low — this is a testing task, not a code change; blocked only on trial-license approval turnaround.
  - > **Note**: `--overwrite` should be included by default in the preset template even though the app's per-run temp directory normally avoids collisions, since it costs nothing and removes one class of avoidable failure.

#### Low priority

- **Seti Astro Cosmic Clarity — feasibility investigation before committing to support** — publicly available usage descriptions point to a "drop files into an Input folder, run the tool, read from an Output folder" workflow (including a "live monitor" mode for continuous folder watching), rather than the classical single-file argv-in/argv-out pattern the other three tools use. If that's accurate, it doesn't fit `run_external_image_tool()`'s per-call temp-file model without a fairly different code path (write to a persistent watched folder, poll for the corresponding output file, handle stale files from previous runs).
  - *Effort*: unknown until investigated — could range from "same as the others" (if a lesser-known direct-file CLI mode exists) to "needs its own integration path" (if folder-watching is the only mode).
  - > **Recommendation**: Before writing any code, confirm directly (via the tool's own `--help`/README, not secondhand descriptions) whether Cosmic Clarity's denoise/sharpen executables accept explicit input/output file paths as arguments. If not, this item should probably be scoped as its own follow-up rather than folded into the generic external-tool feature, since the underlying mechanism would be different enough to warrant separate design.

### Track B — Monitor Panel Polish

*(Unrelated to Track A — grouped into the same release by choice, not because the work overlaps.)*

- **Auto-refreshing CPU/RAM/GPU monitor** — the system monitor panel currently only updates when the user clicks "🔄 Update System Monitor Info"; there's no polling. Add a periodic auto-refresh using Gradio's `Timer` component (`.tick()` bound to `get_system_stats_html`), so the panel stays live without manual clicks, while keeping the existing button as an immediate/force-refresh option.
  - *Effort*: low — `get_system_stats_html()` already exists and is cheap to call; this is wiring, not new computation.
  - > **Design**: default interval a few seconds (e.g. 2–3s) — frequent enough to feel live, not so frequent it adds meaningful CPU overhead from polling `psutil`/`GPUtil` themselves. Add a checkbox to pause/resume the timer, since some users may prefer the current manual-only behavior (e.g. to avoid any background polling at all while running on battery or a shared machine).

- **Extend GPU monitoring to AMD/Intel (DirectML), not just NVIDIA** — `get_system_stats_html()`'s GPU section only produces output through `GPUtil`, which wraps `nvidia-smi` and only ever finds NVIDIA GPUs; the `torch.cuda` fallback path is also NVIDIA-only. Since the app's own background-estimation acceleration explicitly supports DirectML for AMD/Intel/Windows GPUs (a headline feature in the README), a DirectML user currently sees **no GPU row at all** in the monitor — not even an explanatory message — which reads as a bug rather than an unsupported-vendor limitation.
  - *Effort*: medium — a fully cross-vendor utilization/VRAM reader (matching what `GPUtil` gives for NVIDIA) would need something like Windows Performance Counters via WMI, which is more involved and Windows-only.
  - > **Incremental fix, this version**: at minimum, when `GPUtil` finds no GPUs but a DirectML backend is active (`_TORCH_BACKEND_NAME` already tracks this), show a status line like "GPU: DirectML acceleration active (live usage/VRAM not available for this vendor)" instead of an empty section — turns a silent gap into an honest, explained one.
  - > **Follow-up, deferred**: investigate whether a WMI-based (`Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine` or similar) reader can get real utilization/VRAM numbers for AMD/Intel on Windows without adding a heavy new dependency; scope as its own item if it turns out to need a new package like `wmi`/`pywin32`, since that's a bigger commitment than the rest of this list.

### 💬 Feedback

Same as v1.2.0 — direction, not commitment. If Track A and Track B end up ready at very different times, splitting them into separate releases (e.g. v1.2.1 for Track A, v1.2.2 for Track B) instead of shipping together is a reasonable call to make later — this document bundling them doesn't obligate a single combined release.

---

## 繁體中文

下一版候選優化項目，各主線內部依建議優先度排序。跟 `ROADMAP.md` 一樣，不是硬性承諾。

### 主線 A — 外部工具介接

#### 高優先

- **外部工具欄位加上工具範本下拉選單** — 目前「額外參數」是純文字欄位，而這次要支援的四套工具 CLI 慣例各不相同（DeepSNR 是 `-i/-o` 具名旗標；RC-Astro 是子命令 + `--output`；舊版 StarNet 是位置參數；GraXpert 則是 `-cli -cmd ... -output` 且輸出檔名不含副檔名）。建議在 `denoise_ext_path`/`star_ext_path` 旁邊加一個下拉選單——「DeepSNR / RC-Astro / GraXpert / StarNet（舊版）/ 自訂」——選了之後自動把正確的參數範本帶進額外參數欄位，不用每次自己記或手打各家語法。
  - *工作量*：低到中 — 純 UI 便利性功能，加上一張已知範本的對照表；不動 `run_external_image_tool()` 的核心呼叫邏輯。
  - > **研究筆記**：目前確認過的語法——DeepSNR：`--input {input} --output {output}`；RC-Astro：`nxt {input} --output {output} --overwrite` / `sxt {input} --output {output} --overwrite`（單一個 `rc-astro` 執行檔、走子命令）。GraXpert 跟舊版 StarNet 要先處理下面的副檔名問題，才能套進同一種範本形狀。

- **新增 `{output_noext}` 佔位符，給會自己補副檔名的工具用** — GraXpert 的 `-output` 旗標吃的是**不含副檔名**的檔名，副檔名是它自己加上去的。現有佔位符系統假設呼叫端能完全掌控輸出檔名（`{output}` → `output.tif`），這跟 GraXpert 的慣例對不上，就算 GraXpert 實際上跑成功了，也會因為檔名對不上而被你程式誤判成「找不到輸出檔案」失敗。
  - *工作量*：中 — `run_external_image_tool()` 需要（1）提供 `{output_noext}` 這個新佔位符，（2）用到這個佔位符時，改成用檔名主體去暫存資料夾裡搜尋 GraXpert 實際寫出的檔案（glob 比對），而不是死板檢查單一固定路徑。
  - > **技術筆記**：功能做好後，建議範本會是：`-cli -cmd denoising {input} -output {output_noext}`。在這個功能上線之前，GraXpert 沒辦法套用現有「只有 `{output}`」的慣例——會因為檔名對不上而被誤判失敗，這不只是錦上添花，是 GraXpert 能不能接進來的前提條件。

- **暫停 / 恢復即時預覽** — 目前每個滑桿的 `.release()`、每個切換開關的 `.change()`，都是直接無條件呼叫 `update_preview_fn`，沒有辦法在連續調整好幾個參數的過程中先暫停預覽重算。這對 `mode="external"` 影響最大——切到外部模式後，每放開一次滑桿，預覽就真的會呼叫一次外部工具，連續調參數等於連續呼叫好幾次外部程式，不但慢，對 RC-Astro 這種付費/有配額限制的工具來說也很浪費。
  - *工作量*：低 — 加一個 `live_preview_enabled` 勾選框，放進 `_PREV_IN`（現有所有 `.release()`/`.change()` 綁定都共用的輸入清單）就好。`update_preview_fn` 只需要在勾選框關閉時提早 return（所有輸出都用 `gr.update()` 不做事），完全不用去動 `sliders_for_release`、`toggles_for_change` 那幾十行個別的事件綁定，因為它們本來就都會經過 `_PREV_IN`。
  - > **設計方向**：預設開啟（不碰新勾選框的人，行為完全不變，還是現在這種即時模式）。重新勾選恢復時，透過勾選框自己的 `.change()` 立刻觸發一次 `update_preview_fn`，避免恢復後畫面還停在暫停前的舊狀態。這個暫停狀態要跟現有的「圖層預覽」「局部預覽」按鈕互相獨立——那兩個本來就是各自獨立的手動觸發按鈕，即時預覽暫停時它們應該還是要能正常運作。

#### 中優先

- **StarNet / StarNet2 — 上範本前先確認目前實際旗標** — **舊版** StarNet++ CLI（Siril 文件跟論壇貼文裡還看得到）吃的是**位置參數**：`starnet++ <input.tif> [output.tif] [stride]`，沒給輸出檔名時預設寫成 `starless.tiff`——跟 DeepSNR 完全是不同的形狀，也跟你現有「執行檔 輸入檔 輸出檔」的通用退路慣例不完全一樣（舊版 StarNet 恰好能用位置參數對上，但 StarNet2 不保證）。**目前**的 StarNet2 2.5.x（ONNX 版本，跟 DeepSNR 同網站、同打包風格，且文件裡有提到 `--unscreen` 這個雙槓旗標）很可能已經改成跟 DeepSNR 一樣的 `-i/-o` 具名旗標慣例，但這點還沒有直接對照 StarNet 自己的 CLI 文件頁確認過。
  - *工作量*：一旦確認完就很低 — 大概只是在上面的工具範本對照表裡多加一列。
  - > **待辦事項**：下載後先跑一次 `starnet2 -h`（或對應的 `--help`），把實際旗標清單記錄下來，再動手做「StarNet2」範本；不要因為跟 DeepSNR 同廠牌、同打包風格就假設語法一樣，沒查證就不要照抄。

- **RC-Astro NoiseXTerminator/StarXTerminator — 完整實機驗證** — 語法已經查證過（`rc-astro nxt/sxt <input> --output <output> --overwrite`），但還沒有在有效授權下、在你這套程式裡真的成功跑過一次。
  - *工作量*：低 — 這是測試任務，不是程式改動；唯一的卡點是試用授權審核要等多久。
  - > **備註**：`--overwrite` 建議直接內建在範本裡，雖然你程式每次都是全新的暫存資料夾、理論上不會撞名，但加上去不花什麼成本，還能消掉一種可以避免的失敗情境。

#### 低優先

- **Seti Astro Cosmic Clarity — 先做可行性調查，再決定要不要投入支援** — 目前公開查到的用法描述，指向的是「把檔案丟進 Input 資料夾 → 執行工具 → 從 Output 資料夾讀結果」這種模式（甚至有「即時監控」模式持續盯著資料夾），而不是另外三套工具那種單檔 argv 進/argv 出的經典 CLI 模式。如果屬實，這跟 `run_external_image_tool()` 現在「每次呼叫用一個全新暫存資料夾」的模型不吻合，需要一套相當不同的程式邏輯（寫進一個持續被監控的資料夾、輪詢等對應輸出檔出現、還要處理前一次殘留的舊檔案）。
  - *工作量*：調查完才知道 — 有可能跟其他三套工具一樣簡單（如果其實也有比較少人提到的直接檔案 CLI 模式），也有可能得另外設計一條獨立的介接路徑（如果資料夾監控是唯一模式）。
  - > **建議**：動手寫程式前，先直接查證（用工具本身的 `--help`／README，不要用二手描述）Cosmic Clarity 的降噪/銳化執行檔是否接受明確的輸入/輸出檔案路徑當參數。如果不接受，這項應該獨立列成自己的後續項目，而不是硬塞進現有的通用外部工具介接功能裡，因為底層機制差異大到值得分開設計。

### 主線 B — 監控面板優化

*（跟主線 A 沒有技術關聯，純粹是選擇放進同一份改動，不是兩者剛好有重疊。）*

- **CPU/RAM/GPU 監控自動刷新** — 系統監控面板目前只有點擊「🔄 更新監控資訊」按鈕才會更新，完全沒有輪詢機制。建議用 Gradio 的 `Timer` 元件（`.tick()` 綁定 `get_system_stats_html`）做週期性自動刷新，讓面板保持即時，不用手動點；同時保留現有按鈕當作「立即強制刷新」選項。
  - *工作量*：低 — `get_system_stats_html()` 已經存在而且呼叫成本很低，這次只是接線，不是新運算。
  - > **設計方向**：預設間隔抓幾秒鐘（例如 2–3 秒）——夠快讓人感覺是即時的，又不會因為太頻繁呼叫 `psutil`/`GPUtil` 本身而增加明顯的額外負擔。加一個勾選框可以暫停/恢復這個計時器，因為有些使用者可能偏好現在這種純手動的行為（例如用電池跑、或在共用電腦上，想完全避免任何背景輪詢）。

- **GPU 監控擴大支援 AMD/Intel（DirectML），不只 NVIDIA** — `get_system_stats_html()` 的 GPU 區塊目前完全靠 `GPUtil` 產生內容，這個套件底層包的是 `nvidia-smi`，只找得到 NVIDIA 顯卡；`torch.cuda` 那條備援路徑也一樣只認 NVIDIA。但你程式本身的背景估計加速明確支援 DirectML（AMD/Intel/Windows GPU），README 裡還把這個當主打功能寫——結果用 DirectML 的使用者打開監控面板，GPU 區塊**完全是空的**，連一句說明都沒有，看起來像是程式壞掉，而不是「這個廠牌暫不支援」。
  - *工作量*：中 — 要做到跟 `GPUtil` 給 NVIDIA 那樣完整的跨廠牌使用率/VRAM 讀取，大概需要透過 Windows 效能計數器（WMI）之類的機制，牽涉範圍較大，而且只能限 Windows。
  - > **這一版先做的漸進式修正**：至少在 `GPUtil` 找不到任何顯卡、但 DirectML 後端確實在跑（`_TORCH_BACKEND_NAME` 本來就有記錄這個狀態）的情況下，顯示一行像「GPU：DirectML 加速中（此廠牌暫無法讀取即時使用率/VRAM）」的狀態文字，取代現在完全空白的區塊——把「使用者看不出來的缺口」變成「講清楚的已知限制」。
  - > **後續延伸（先不做）**：研究是否能用 WMI（例如 `Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine` 之類）在不引入太重的新相依套件情況下，讀到 AMD/Intel 在 Windows 上真正的使用率/VRAM 數字；如果最後發現得加 `wmi`/`pywin32` 這類新套件才做得到，建議獨立列成自己的項目，因為那個投入會比這份清單裡其他項目重得多。

### 💬 意見回饋

跟 v1.2.0 一樣，這只是方向不是承諾。如果主線 A 跟主線 B 到時候完成的時間點差很多，之後拆成兩次發版（例如主線 A 出 v1.2.1、主線 B 延到 v1.2.2）也完全合理——這份文件把兩者放在一起寫，不代表一定要綁在同一次發版裡一起出。
