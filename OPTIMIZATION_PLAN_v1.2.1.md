# 🛠️ Optimization Plan — v1.2.1

[English](#english) | [繁體中文](#繁體中文)

> 📝 Scope: two tracks this release — **Track A** broadens v1.2.0's external-tool integration (denoise/star removal) to explicitly cover StarNet/StarNet2, RC-Astro NoiseXTerminator/StarXTerminator, Seti Astro Cosmic Clarity, and GraXpert; **Track B** is unrelated monitor-panel polish (auto-refresh, AMD/Intel GPU detection) that got folded into the same release by choice, not because the two are technically connected.

---

## English

Candidate improvements for the next release, in suggested priority order within each track. Not a strict commitment — same as `ROADMAP.md`.

### Track A — External Tool Integration

#### High priority

- **Tool profile presets for the external-tool fields** — ✅ **Shipped.** Added `EXTERNAL_TOOL_PROFILES_DENOISE`/`EXTERNAL_TOOL_PROFILES_STAR` lookup tables and a `gr.Dropdown` next to each extra-arguments field that auto-fills the correct template on selection, without touching `run_external_image_tool()`'s core call path.
  - > **Research note (updated)**: DeepSNR's template was corrected to the user's own verified working command — `-i {input} -o {output} -m 2 -s 32` (short flags, plus `-m` model version and `-s` stride; these two don't have one universally-correct value, so the template is a verified-working starting point, not an official recommended default). RC-Astro: `nxt {input} --output {output} --overwrite` / `sxt {input} --output {output} --overwrite` (documented, not yet run against a real license). GraXpert: documented, not yet run. Legacy StarNet: template left blank since positional args already match the generic fallback convention.

- **`{output_noext}` placeholder for tools that append their own extension** — ✅ **Shipped.** `run_external_image_tool()` now detects `{output_noext}` in the extra-arguments string, substitutes a filename-without-extension, and — instead of checking one hardcoded path — globs the temp directory for whatever file the tool actually wrote. Falls back from `tifffile.imread` to `cv2.imread` if the tool didn't write a TIFF. GraXpert's profile template uses this: `-cli -cmd denoising {input} -output {output_noext}`.

- **Pause / resume live preview** — ✅ **Shipped.** Added a `live_preview_enabled` checkbox to `_PREV_IN`, the shared input list already used by every existing `.release()`/`.change()` binding — no changes needed to the dozens of individual event-wiring lines. `update_preview_fn` now early-returns `gr.update()` no-ops (before `run_pipeline` is ever called) when paused, so `mode="external"` doesn't shell out on every tick. The checkbox's own `.change()` fires one immediate update on resume.

#### Medium priority

- **StarNet / StarNet2 — confirm current CLI flags before shipping a preset template** — the *legacy* StarNet++ CLI took **positional** arguments: `starnet++ <input.tif> [output.tif] [stride]`, defaulting the output to `starless.tiff` if omitted (stride default 256) — a different shape from DeepSNR entirely. Cross-checking the current official CLI download page against the "StarNet2 includes `--unscreen` for an optional star-layer output" release note is a meaningful clue: it reads as an *additional* opt-in double-dash flag layered on top of an existing interface, not a rewrite to named `-i/-o` flags — i.e. this is evidence StarNet2 likely **kept the same positional input/output convention** as legacy StarNet, just with new opt-in flags (`--unscreen`, `--eight`) bolted on. This raises confidence that the current "StarNet（舊版，位置參數）" profile (blank extra-args, relying on the generic fallback) should also work for StarNet2's core input/output — but this is still an inference from indirect evidence, not a confirmed flag list from StarNet's own CLI reference page.
  - *Effort*: low once confirmed — likely just a rename/relabel of the existing profile entry, or at most adding optional `--eight`/`--unscreen` flags to it.
  - > **Open item**: Still worth running `starnet2 -h` after download to confirm directly — the evidence above is reasonably convincing but not a substitute for reading the tool's own `--help` output. If confirmed, the "legacy" qualifier in the profile name can be dropped since it'd apply to StarNet2 too.

- **RC-Astro NoiseXTerminator/StarXTerminator — full end-to-end verification** — syntax is documented (`rc-astro nxt/sxt <input> --output <output> --overwrite`), but not yet run successfully against a real license in this app.
  - *Effort*: low — this is a testing task, not a code change; blocked only on trial-license approval turnaround.
  - > **Note**: `--overwrite` should be included by default in the preset template even though the app's per-run temp directory normally avoids collisions, since it costs nothing and removes one class of avoidable failure.

#### Low priority

- **Seti Astro Cosmic Clarity — feasibility investigation before committing to support** — checked the official usage docs directly, and the picture is worse than a folder-watch inconvenience: the workflow is "drop images into an Input folder, run `SetiAstroCosmicClarity_denoise` (or the sharpen equivalent), then the program prompts interactively — e.g. 'Choose Full or Luminance and the amount'." That's an **interactive console app that asks for input after launch**, not just a folder-based batch tool. A plain `subprocess.run(cmd, capture_output=True)` call would hang waiting on stdin unless there's an undocumented non-interactive/headless flag set. It is open source (`github.com/setiastro/cosmicclarity`), so a definitive answer is possible by reading the source for `argparse`/`sys.argv` handling, but that wasn't done yet — this write-up is based on the official usage docs, not the source.
  - *Effort*: unknown until the source is actually checked — if there's a hidden headless flag set (`--input`/`--output`/`--amount`/`--yes`, etc.), this could still fit `run_external_image_tool()`'s model; if the interactive prompt is truly the only path, this needs its own automation approach entirely (e.g. driving stdin programmatically, which is a meaningfully different and more fragile integration than any of the other three tools).
  - > **Recommendation**: next concrete step is reading `setiastro/cosmicclarity`'s source directly (it's open source) for command-line argument parsing before writing any integration code — don't rely on the usage docs alone, since they describe the interactive path but say nothing about whether a headless mode exists. If no headless mode is found, downgrade this from "low priority, do later" to "not a fit for the generic external-tool feature" rather than leaving it open indefinitely.

### Track B — Monitor Panel Polish

*(Unrelated to Track A — grouped into the same release by choice, not because the work overlaps.)*

- **Auto-refreshing CPU/RAM/GPU monitor** — ✅ **Shipped.** Added a `gr.Timer(3, active=True)` whose `.tick()` calls the existing `get_system_stats_html()` — no new computation, just periodic wiring. A checkbox next to the existing refresh button toggles the timer's `active` state (`gr.Timer(value=3, active=bool(enabled))`); the manual button still works normally regardless of the toggle.

- **Extend GPU monitoring to AMD/Intel (DirectML), not just NVIDIA** — ✅ **Shipped (incremental fix only, as scoped).** `get_system_stats_html()`'s GPU section now tracks whether any row was actually added; if `GPUtil` finds nothing but `_IS_DIRECTML` (an existing module-level flag) is true, it shows "⚡ DirectML acceleration active (live usage/VRAM not available for this vendor)" instead of silently showing nothing. The deferred WMI-based real-number follow-up was **not** attempted — still open, as originally scoped. Note: `get_status_bar_html()`'s bottom-bar GPU readout has the identical NVIDIA-only gap and was **not** touched — same underlying limitation, just a different function, left out to stay within this item's original scope; worth a follow-up note if it's worth fixing too.

### Hotfix — Language-switching gaps introduced by this release

*(Found during post-release review, not part of the original candidate list above — logged here since both root causes trace back to Track A/B UI additions in this same release.)*

- **New v1.2.1 controls weren't wired into the language-switch system at all** — ✅ **Fixed.** Four newly-added components (`live_preview_enabled`, `denoise_ext_profile`, `star_ext_profile`, `monitor_auto_refresh`) had their label/info text hardcoded in Traditional Chinese at creation time and were never registered in `UI_TRANSLATIONS` / `TRANSLATED_COMPONENTS_MAP`, so switching to English silently skipped them. Added full zh/en entries for all four and registered them in the switch map; also gave the `star_ext_profile` dropdown's "StarNet（舊版，位置參數）" choice a bilingual label to match the "自訂 / Custom" convention already used elsewhere in that list, since dropdown *choices* aren't touched by the label/info update path at all.
- **A second, wider bug: label switches but the info tooltip doesn't** — ✅ **Fixed.** Systematic audit (cross-checking every widget's `info=` presence against whether its `UI_TRANSLATIONS` entry was a `(label, info)` tuple vs. a bare string) turned up **24 sliders** — spanning denoise, star shrink/removal, cluster detection, and the local-preview tab — where only the label was stored as a translatable tuple; the info line was a bare Chinese string, so toggling to English changed the title but silently left the description text in Chinese. All 24 entries were converted to full `(label, info)` tuples with English translations added, and re-verified with an automated scan (0 remaining mismatches) rather than spot-checking by eye.
- > **Note**: Neither of these was on the original Track A/B candidate list — they're regressions introduced *by* shipping Track A/B, caught in review rather than before release. Worth adding "does every new widget have both label *and* info registered in `UI_TRANSLATIONS`?" as a checklist item for future UI additions, since this class of bug is easy to miss visually (the label switching correctly makes it look done at a glance).

### 💬 Feedback

Same as v1.2.0 — direction, not commitment. If Track A and Track B end up ready at very different times, splitting them into separate releases (e.g. v1.2.1 for Track A, v1.2.2 for Track B) instead of shipping together is a reasonable call to make later — this document bundling them doesn't obligate a single combined release.

---

## 繁體中文

下一版候選優化項目，各主線內部依建議優先度排序。跟 `ROADMAP.md` 一樣，不是硬性承諾。

### 主線 A — 外部工具介接

#### 高優先

- **外部工具欄位加上工具範本下拉選單** — ✅ **已完成**。新增 `EXTERNAL_TOOL_PROFILES_DENOISE`/`EXTERNAL_TOOL_PROFILES_STAR` 對照表，並在每個額外參數欄位旁加了 `gr.Dropdown`，選了自動帶入正確範本，不動 `run_external_image_tool()` 核心呼叫邏輯。
  - > **研究筆記（已更新）**：DeepSNR 範本已改用使用者自己實測跑成功的指令——`-i {input} -o {output} -m 2 -s 32`（短旗標，加上 `-m` 模型版本、`-s` stride；這兩個沒有放諸四海皆準的正確值，範本只是「驗證過可用」的起點，不是官方建議預設值）。RC-Astro：`nxt {input} --output {output} --overwrite` / `sxt {input} --output {output} --overwrite`（已查證文件，尚未在有效授權下實測）。GraXpert：已查證文件，尚未實測。舊版 StarNet：範本留空，因為位置參數本來就跟通用退路慣例相容。

- **新增 `{output_noext}` 佔位符，給會自己補副檔名的工具用** — ✅ **已完成**。`run_external_image_tool()` 現在會偵測額外參數字串裡的 `{output_noext}`，替換成不含副檔名的檔名，並改用 glob 搜尋暫存資料夾裡工具實際寫出的檔案，而不是死板檢查單一固定路徑；如果工具沒吐 TIFF，還會從 `tifffile.imread` 退回 `cv2.imread`。GraXpert 範本已經在用：`-cli -cmd denoising {input} -output {output_noext}`。

- **暫停 / 恢復即時預覽** — ✅ **已完成**。新增 `live_preview_enabled` 勾選框並塞進 `_PREV_IN`（現有所有 `.release()`/`.change()` 綁定共用的輸入清單），完全不用動那幾十行個別的事件綁定。`update_preview_fn` 暫停時會在呼叫 `run_pipeline` 之前就提早 return no-op，所以 `mode="external"` 不會在每次滑桿放開時都真的呼叫外部工具。勾選框自己的 `.change()` 會在恢復時立刻補跑一次。

#### 中優先

- **StarNet / StarNet2 — 上範本前先確認目前實際旗標** — **舊版** StarNet++ CLI 吃的是**位置參數**：`starnet++ <input.tif> [output.tif] [stride]`，沒給輸出檔名時預設寫成 `starless.tiff`（stride 預設 256）——跟 DeepSNR 完全是不同的形狀。這次交叉比對官方 CLI 下載頁跟「StarNet2 新增 `--unscreen`（選用的星點圖層輸出）」這則發布說明，發現一個蠻有意思的線索：這讀起來像是「在既有介面上疊加一個選用的雙槓旗標」，而不是「整個介面改寫成具名的 `-i/-o` 旗標」——也就是說，有跡象顯示 StarNet2 很可能**沿用了舊版同一套位置參數輸入/輸出慣例**，只是多加了 `--unscreen`、`--eight` 這類選用開關。這讓「StarNet（舊版，位置參數）」這個範本（額外參數留空、走通用退路）很可能對 StarNet2 的核心輸入/輸出也適用——但這仍然是從間接證據推論出來的，不是直接查到 StarNet 自己 CLI 文件頁列出的旗標清單。
  - *工作量*：一旦確認完就很低 — 大概只是把現有範本改個名字，或最多加上選用的 `--eight`/`--unscreen` 旗標。
  - > **待辦事項**：還是建議下載後跑一次 `starnet2 -h` 直接確認——上面的證據算相當有說服力，但不能取代真的看一次工具自己的 `--help` 輸出。確認後可以把範本名稱裡的「舊版」拿掉，因為屆時 StarNet2 應該也適用。

- **RC-Astro NoiseXTerminator/StarXTerminator — 完整實機驗證** — 語法已經查證過（`rc-astro nxt/sxt <input> --output <output> --overwrite`），但還沒有在有效授權下、在你這套程式裡真的成功跑過一次。
  - *工作量*：低 — 這是測試任務，不是程式改動；唯一的卡點是試用授權審核要等多久。
  - > **備註**：`--overwrite` 建議直接內建在範本裡，雖然你程式每次都是全新的暫存資料夾、理論上不會撞名，但加上去不花什麼成本，還能消掉一種可以避免的失敗情境。

#### 低優先

- **Seti Astro Cosmic Clarity — 先做可行性調查，再決定要不要投入支援** — 直接查了官方使用說明，狀況比「資料夾監控」還麻煩：流程是「把圖丟進 Input 資料夾 → 執行 `SetiAstroCosmicClarity_denoise`（或對應的銳化版本）→ 程式啟動後會互動詢問，例如『選擇 Full 還是 Luminance、選擇強度』」。這是**啟動後會跳出來問你選項的互動式主控台程式**，不只是資料夾批次工具而已。單純的 `subprocess.run(cmd, capture_output=True)` 呼叫會卡在等待標準輸入那一步，除非有沒寫在文件裡的無人值守/headless 旗標。它是開源的（`github.com/setiastro/cosmicclarity`），理論上可以直接看原始碼裡 `argparse`/`sys.argv` 的處理方式找到確切答案，但這次還沒真的去看——這段結論是根據官方使用說明文件，不是原始碼。
  - *工作量*：要實際看過原始碼才知道——如果藏著隱藏的無人值守旗標組合（例如 `--input`/`--output`/`--amount`/`--yes` 之類），還是有機會套進 `run_external_image_tool()` 現有模型；如果互動輸入真的是唯一路徑，就需要一套完全獨立的自動化做法（例如程式化操控標準輸入），跟其他三套工具比起來明顯更脆弱、更難維護。
  - > **建議**：下一步具體要做的是直接去看 `setiastro/cosmicclarity` 的原始碼（開源可看）確認命令列參數處理方式，再決定要不要動手寫介接程式碼——不要只憑使用說明文件下結論，因為文件只描述了互動路徑，完全沒提到有沒有 headless 模式存在。如果查完確定沒有 headless 模式，建議直接把這項從「低優先、之後再做」降級成「不適合套進通用外部工具介接功能」，而不是無限期掛在待辦清單裡。

### 主線 B — 監控面板優化

*（跟主線 A 沒有技術關聯，純粹是選擇放進同一份改動，不是兩者剛好有重疊。）*

- **CPU/RAM/GPU 監控自動刷新** — ✅ **已完成**。新增 `gr.Timer(3, active=True)`，`.tick()` 綁定現有的 `get_system_stats_html()`——沒有新運算，純粹是週期性接線。既有刷新按鈕旁加了一個勾選框，切換 Timer 的 `active` 狀態（`gr.Timer(value=3, active=bool(enabled))`）；不管勾選框開關與否，手動按鈕都能正常使用。

- **GPU 監控擴大支援 AMD/Intel（DirectML），不只 NVIDIA** — ✅ **已完成（照原本規劃只做漸進式修正）**。`get_system_stats_html()` 的 GPU 區塊現在會追蹤有沒有真的加進任何一行；如果 `GPUtil` 找不到顯卡、但 `_IS_DIRECTML`（既有的模組層級旗標）是 True，就顯示「⚡ DirectML 加速中（此廠牌暫無法讀取即時使用率/VRAM）」，取代原本完全空白的區塊。延後的 WMI 真實數字讀取後續延伸**沒有動**——照原本規劃維持待辦。附註：`get_status_bar_html()` 底部狀態列的 GPU 顯示有一模一樣的 NVIDIA-only 缺口，這次**沒有動**——同樣的底層限制、但是不同函式，為了不超出這項的原始範圍先沒處理；如果之後覺得也值得修，可以再另外列一項。

### 修補項目 — 這次發版本身造成的語言切換缺口

*（發版後複查時才發現的，不在上面原本的候選清單裡——因為兩個根因都是這次主線 A/B 新增 UI 造成的，所以記在這裡。）*

- **v1.2.1 新增的控制項根本沒接上語言切換機制** — ✅ **已修復**。新增的 4 個元件（`live_preview_enabled`、`denoise_ext_profile`、`star_ext_profile`、`monitor_auto_refresh`）建立時 label/info 文字直接寫死中文，完全沒登記進 `UI_TRANSLATIONS`／`TRANSLATED_COMPONENTS_MAP`，切到英文時直接被跳過、毫無反應。已補上四者完整的 zh/en 條目並登記進切換清單；另外 `star_ext_profile` 下拉選單裡的「StarNet（舊版，位置參數）」選項也改成跟清單裡「自訂 / Custom」同樣的雙語寫法，因為下拉選單的**選項本身**完全不在 label/info 的更新路徑裡，語言切換機制本來就不會碰到它。
- **另一個範圍更大的 bug：標題會變英文，但下面的說明文字不會** — ✅ **已修復**。逐一比對每個元件「widget 本身有沒有 `info=`」跟「`UI_TRANSLATIONS` 裡對應條目是 `(label, info)` tuple 還是單純字串」後，抓出 **24 個滑桿**（分布在降噪、縮星/去星、星團偵測、局部預覽分頁）都只把標題存成可翻譯的 tuple，說明文字是單純中文字串——切到英文時標題會變，但下面那行說明文字悄悄維持中文不動。24 筆全部改成完整的 `(label, info)` tuple 並補上英文翻譯，並用自動化掃描重新驗證（剩餘 0 筆不一致），不是靠肉眼抽查確認。
- > **備註**：這兩個都不在主線 A/B 原本的候選清單裡——是主線 A/B 這次上線本身造成的回歸，是複查時才抓到的，不是發版前就抓到的。建議之後新增 UI 元件時，把「這個新元件的 label *跟* info 是不是都登記進 `UI_TRANSLATIONS` 了？」列進檢查清單，因為這類 bug 光用肉眼看很容易漏掉（畢竟標題有正確變英文，乍看會以為已經做完了）。

### 💬 意見回饋

跟 v1.2.0 一樣，這只是方向不是承諾。如果主線 A 跟主線 B 到時候完成的時間點差很多，之後拆成兩次發版（例如主線 A 出 v1.2.1、主線 B 延到 v1.2.2）也完全合理——這份文件把兩者放在一起寫，不代表一定要綁在同一次發版裡一起出。
