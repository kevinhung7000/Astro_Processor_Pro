# 🛠️ Optimization Plan — v1.2.1

[English](#english) | [繁體中文](#繁體中文)

> 📝 Scope: broadening v1.2.0's external-tool integration (denoise/star removal) to explicitly cover StarNet/StarNet2, RC-Astro NoiseXTerminator/StarXTerminator, Seti Astro Cosmic Clarity, and GraXpert — each with its own CLI conventions, verified where possible against official documentation as of this draft.

---

## English

Candidate improvements for the next release, in suggested priority order. Not a strict commitment — same as `ROADMAP.md`.

### High priority

- **Tool profile presets for the external-tool fields** — right now the "extra arguments" textbox is free text, and each of the four target tools uses a different CLI convention (DeepSNR: `-i/-o` named flags; RC-Astro: subcommand + `--output`; legacy StarNet: positional args; GraXpert: `-cli -cmd ... -output` with an extension-less output name). Add a dropdown next to `denoise_ext_path`/`star_ext_path` — "DeepSNR / RC-Astro / GraXpert / StarNet (legacy) / Custom" — that auto-fills the extra-arguments field with the correct template when selected, instead of requiring the user to remember or hand-type each tool's syntax.
  - *Effort*: low-medium — pure UI convenience plus a small lookup table of known templates; no change to `run_external_image_tool()`'s core call path.
  - > **Research note**: Confirmed syntax so far — DeepSNR: `--input {input} --output {output}`; RC-Astro: `nxt {input} --output {output} --overwrite` / `sxt {input} --output {output} --overwrite` (single `rc-astro` executable, subcommand-based). GraXpert and legacy StarNet need the extension handling below before they fit the same template shape.

- **`{output_noext}` placeholder for tools that append their own extension** — GraXpert's `-output` flag takes a filename **without an extension**; GraXpert appends one itself. The current placeholder system assumes the caller controls the exact output filename (`{output}` → `output.tif`), which doesn't fit GraXpert's convention and would cause "output file not found" failures even on a successful run.
  - *Effort*: medium — `run_external_image_tool()` needs to (1) offer a `{output_noext}` placeholder alongside `{input}`/`{output}`, and (2) when it's used, search the temp directory for whatever file GraXpert actually wrote (glob on the stem) instead of checking a single hardcoded path.
  - > **Technical note**: Recommended template once this lands: `-cli -cmd denoising {input} -output {output_noext}`. Until this ships, GraXpert cannot be wired up through the existing `{output}`-only convention without the tool silently "failing" due to a filename mismatch — this is a prerequisite, not just a nice-to-have, for GraXpert support specifically.

### Medium priority

- **StarNet / StarNet2 — confirm current CLI flags before shipping a preset template** — the *legacy* StarNet++ CLI (still referenced in some Siril docs and forum posts) took **positional** arguments: `starnet++ <input.tif> [output.tif] [stride]`, defaulting the output to `starless.tiff` if omitted — a different shape from DeepSNR entirely, and from the generic "executable input output" fallback (works by coincidence for legacy StarNet, but not guaranteed for StarNet2). The *current* StarNet2 2.5.x release (ONNX-based, packaged on the same site and in the same style as DeepSNR, with a documented `--unscreen` flag) likely adopted the same `-i/-o` named-flag convention DeepSNR uses, but this has not been directly confirmed against StarNet's own CLI reference page.
  - *Effort*: low once confirmed — likely just another entry in the tool-profile table above.
  - > **Open item**: Run `starnet2 -h` (or equivalent `--help`) after download and record the actual flag list before adding a "StarNet2" preset; don't assume it mirrors DeepSNR without checking, even though it's the same vendor and packaging style.

- **RC-Astro NoiseXTerminator/StarXTerminator — full end-to-end verification** — syntax is documented (`rc-astro nxt/sxt <input> --output <output> --overwrite`), but not yet run successfully against a real license in this app.
  - *Effort*: low — this is a testing task, not a code change; blocked only on trial-license approval turnaround.
  - > **Note**: `--overwrite` should be included by default in the preset template even though the app's per-run temp directory normally avoids collisions, since it costs nothing and removes one class of avoidable failure.

### Low priority

- **Seti Astro Cosmic Clarity — feasibility investigation before committing to support** — publicly available usage descriptions point to a "drop files into an Input folder, run the tool, read from an Output folder" workflow (including a "live monitor" mode for continuous folder watching), rather than the classical single-file argv-in/argv-out pattern the other three tools use. If that's accurate, it doesn't fit `run_external_image_tool()`'s per-call temp-file model without a fairly different code path (write to a persistent watched folder, poll for the corresponding output file, handle stale files from previous runs).
  - *Effort*: unknown until investigated — could range from "same as the others" (if a lesser-known direct-file CLI mode exists) to "needs its own integration path" (if folder-watching is the only mode).
  - > **Recommendation**: Before writing any code, confirm directly (via the tool's own `--help`/README, not secondhand descriptions) whether Cosmic Clarity's denoise/sharpen executables accept explicit input/output file paths as arguments. If not, this item should probably be scoped as its own follow-up rather than folded into the generic external-tool feature, since the underlying mechanism would be different enough to warrant separate design.

### 💬 Feedback

Same as v1.2.0 — direction, not commitment.

---

## 繁體中文

下一版候選優化項目，依建議優先度排序。跟 `ROADMAP.md` 一樣，不是硬性承諾。

### 高優先

- **外部工具欄位加上工具範本下拉選單** — 目前「額外參數」是純文字欄位，而這次要支援的四套工具 CLI 慣例各不相同（DeepSNR 是 `-i/-o` 具名旗標；RC-Astro 是子命令 + `--output`；舊版 StarNet 是位置參數；GraXpert 則是 `-cli -cmd ... -output` 且輸出檔名不含副檔名）。建議在 `denoise_ext_path`/`star_ext_path` 旁邊加一個下拉選單——「DeepSNR / RC-Astro / GraXpert / StarNet（舊版）/ 自訂」——選了之後自動把正確的參數範本帶進額外參數欄位，不用每次自己記或手打各家語法。
  - *工作量*：低到中 — 純 UI 便利性功能，加上一張已知範本的對照表；不動 `run_external_image_tool()` 的核心呼叫邏輯。
  - > **研究筆記**：目前確認過的語法——DeepSNR：`--input {input} --output {output}`；RC-Astro：`nxt {input} --output {output} --overwrite` / `sxt {input} --output {output} --overwrite`（單一個 `rc-astro` 執行檔、走子命令）。GraXpert 跟舊版 StarNet 要先處理下面的副檔名問題，才能套進同一種範本形狀。

- **新增 `{output_noext}` 佔位符，給會自己補副檔名的工具用** — GraXpert 的 `-output` 旗標吃的是**不含副檔名**的檔名，副檔名是它自己加上去的。現有佔位符系統假設呼叫端能完全掌控輸出檔名（`{output}` → `output.tif`），這跟 GraXpert 的慣例對不上，就算 GraXpert 實際上跑成功了，也會因為檔名對不上而被你程式誤判成「找不到輸出檔案」失敗。
  - *工作量*：中 — `run_external_image_tool()` 需要（1）提供 `{output_noext}` 這個新佔位符，（2）用到這個佔位符時，改成用檔名主體去暫存資料夾裡搜尋 GraXpert 實際寫出的檔案（glob 比對），而不是死板檢查單一固定路徑。
  - > **技術筆記**：功能做好後，建議範本會是：`-cli -cmd denoising {input} -output {output_noext}`。在這個功能上線之前，GraXpert 沒辦法套用現有「只有 `{output}`」的慣例——會因為檔名對不上而被誤判失敗，這不只是錦上添花，是 GraXpert 能不能接進來的前提條件。

### 中優先

- **StarNet / StarNet2 — 上範本前先確認目前實際旗標** — **舊版** StarNet++ CLI（Siril 文件跟論壇貼文裡還看得到）吃的是**位置參數**：`starnet++ <input.tif> [output.tif] [stride]`，沒給輸出檔名時預設寫成 `starless.tiff`——跟 DeepSNR 完全是不同的形狀，也跟你現有「執行檔 輸入檔 輸出檔」的通用退路慣例不完全一樣（舊版 StarNet 恰好能用位置參數對上，但 StarNet2 不保證）。**目前**的 StarNet2 2.5.x（ONNX 版本，跟 DeepSNR 同網站、同打包風格，且文件裡有提到 `--unscreen` 這個雙槓旗標）很可能已經改成跟 DeepSNR 一樣的 `-i/-o` 具名旗標慣例，但這點還沒有直接對照 StarNet 自己的 CLI 文件頁確認過。
  - *工作量*：一旦確認完就很低 — 大概只是在上面的工具範本對照表裡多加一列。
  - > **待辦事項**：下載後先跑一次 `starnet2 -h`（或對應的 `--help`），把實際旗標清單記錄下來，再動手做「StarNet2」範本；不要因為跟 DeepSNR 同廠牌、同打包風格就假設語法一樣，沒查證就不要照抄。

- **RC-Astro NoiseXTerminator/StarXTerminator — 完整實機驗證** — 語法已經查證過（`rc-astro nxt/sxt <input> --output <output> --overwrite`），但還沒有在有效授權下、在你這套程式裡真的成功跑過一次。
  - *工作量*：低 — 這是測試任務，不是程式改動；唯一的卡點是試用授權審核要等多久。
  - > **備註**：`--overwrite` 建議直接內建在範本裡，雖然你程式每次都是全新的暫存資料夾、理論上不會撞名，但加上去不花什麼成本，還能消掉一種可以避免的失敗情境。

### 低優先

- **Seti Astro Cosmic Clarity — 先做可行性調查，再決定要不要投入支援** — 目前公開查到的用法描述，指向的是「把檔案丟進 Input 資料夾 → 執行工具 → 從 Output 資料夾讀結果」這種模式（甚至有「即時監控」模式持續盯著資料夾），而不是另外三套工具那種單檔 argv 進/argv 出的經典 CLI 模式。如果屬實，這跟 `run_external_image_tool()` 現在「每次呼叫用一個全新暫存資料夾」的模型不吻合，需要一套相當不同的程式邏輯（寫進一個持續被監控的資料夾、輪詢等對應輸出檔出現、還要處理前一次殘留的舊檔案）。
  - *工作量*：調查完才知道 — 有可能跟其他三套工具一樣簡單（如果其實也有比較少人提到的直接檔案 CLI 模式），也有可能得另外設計一條獨立的介接路徑（如果資料夾監控是唯一模式）。
  - > **建議**：動手寫程式前，先直接查證（用工具本身的 `--help`／README，不要用二手描述）Cosmic Clarity 的降噪/銳化執行檔是否接受明確的輸入/輸出檔案路徑當參數。如果不接受，這項應該獨立列成自己的後續項目，而不是硬塞進現有的通用外部工具介接功能裡，因為底層機制差異大到值得分開設計。

### 💬 意見回饋

跟 v1.2.0 一樣，這只是方向不是承諾。
