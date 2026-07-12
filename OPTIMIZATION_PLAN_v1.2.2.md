# 🛠️ Optimization Plan — v1.2.2

[English](#english) | [繁體中文](#繁體中文)

> 📝 Scope: this is a **single track**, not two — it's the three items carried over unchanged from v1.2.1's Track A (medium/low priority), none of which shipped last release. No unrelated "Track B"-style grab-bag this time; if something unrelated comes up before this ships, it should probably get its own plan doc rather than being folded in by default. A short process note is appended at the end — not a feature, just a housekeeping reminder that came out of a v1.2.1 post-release fix.
>
> **Update:** the investigation/verification work for two of the three carried-over items has since been done (StarNet2 CLI syntax confirmed; Cosmic Clarity feasibility checked against upstream source). Status notes and code changes are marked inline below with ✅. The RC-Astro item is still blocked on external license approval and is unchanged.

---

## English

Candidate improvements for the next release, in suggested priority order. Not a strict commitment — same as `ROADMAP.md`. All three items below are unchanged in substance from the v1.2.1 plan; none of them shipped last release, so they're carried forward rather than rewritten.

### External Tool Integration (carried over from v1.2.1)

#### High priority

- **RC-Astro NoiseXTerminator/StarXTerminator — full end-to-end verification** — syntax is documented and already shipped as a preset template (`nxt {input} --output {output} --overwrite` / `sxt {input} --output {output} --overwrite`), but still hasn't been run successfully against a real license in this app.
  - *Effort*: low — this is a testing task, not a code change; blocked only on trial-license approval turnaround, same as last release.
  - *Priority bump from v1.2.1*: moved from Medium to High for this cycle — it's the lowest-effort of the three remaining items and the only thing blocking it is external (license approval), not more investigation work on our side.
  - *Status: still blocked, unchanged.* No license available to this pass, so no progress possible here beyond what v1.2.1 already had. Genuinely still just waiting on trial approval.

#### Medium priority

- **StarNet / StarNet2 — confirm current CLI flags before shipping a preset template** — ✅ **Resolved this pass.** Checked the current official documentation (`starnetastro.com/documentation/starnet/command-line-tool/`). The original hypothesis was wrong: StarNet2 does **not** reuse legacy StarNet's positional-argument convention. It uses named flags: `-i/--input`, `-o/--output`, plus optional `-m/--mask`, `-n/--unscreen`, `-u/--upsample`, `-e/--eight`, `-s/--stride`, `-w/--weights`. `--unscreen` (mentioned in the v1.2.1 release note that prompted this item) is a genuinely additive optional-output flag, as suspected — but that doesn't mean the base I/O convention carried over.
  - *Effort*: was low once confirmed, and it stayed low — added a new, separate preset entry (`StarNet2, Named Flags` → `--input {input} --output {output}`) rather than relabeling the legacy positional-args entry, since the two tools are not interchangeable. Both entries now coexist in the dropdown.
  - *Status*: shipped in code (`EXTERNAL_TOOL_PROFILES_STAR`), no longer an open item.

#### Low priority

- **Seti Astro Cosmic Clarity — feasibility investigation before committing to support** — ✅ **Resolved this pass, and the answer is "yes, but differently than assumed."** Read the actual upstream source (`SetiAstroCosmicClarity_denoise.py`, `github.com/setiastro/cosmicclarity`) rather than relying on the user-facing docs alone. Two corrections to the original assumption:
  1. The "prompts for Full or Luminance and the amount" flow is a **Tkinter GUI dialog**, not a blocking `input()`/stdin prompt as assumed. More importantly, the source has an explicit headless branch: `process_images()` only opens the GUI `if denoise_strength is None`; supplying `--denoise_strength` on the command line skips the GUI entirely and never touches stdin. `subprocess.run(cmd, capture_output=True)` does **not** hang, contrary to the original concern.
  2. However, the I/O convention is different from every other tool this app integrates with: Cosmic Clarity always reads from `<exe_dir>/input/` and writes to `<exe_dir>/output/` (fixed folders relative to the executable), with a fixed output filename pattern (`<stem>_denoised.<ext>`). It does not accept `--input`/`--output` path arguments. This does not fit the generic `{input}/{output}` single-file placeholder convention the rest of the app's external-tool integration assumes.
  - *Recommendation, updated*: the original recommendation was "if no headless mode exists, downgrade to not-a-fit." A headless mode *does* exist, so instead of downgrading, a dedicated integration path was written: `run_cosmic_clarity_denoise()` (separate from `run_external_image_tool()`) that stages the temp file into the tool's fixed `input/` folder, invokes it with the required `--denoise_strength` flag, and reads the result back from `output/` by the known naming pattern. Wired into `denoise()`'s `external` mode via auto-detection (routes to the dedicated function whenever `ext_args` contains `--denoise_strength`).
  - *Scope note*: this pass only covers the **Denoise** tool (`SetiAstroCosmicClarity_denoise.py`), since that's the one the user-facing docs and this plan referenced. `SetiAstroCosmicClarity.py` (Sharpen) uses the same fixed-folder + argparse convention but a different output suffix (`_sharpened`); a near-identical dedicated function would be needed for it, not yet written.
  - *Status*: shipped in code (`run_cosmic_clarity_denoise()`, new profile entry in `EXTERNAL_TOOL_PROFILES_DENOISE`), no longer an open item for Denoise. Sharpen support remains a small, well-scoped follow-up if wanted.

### 🧹 Process Note (not a feature)

- **New-widget i18n checklist** — v1.2.1 shipped 4 new controls that were never wired into the language switcher at all, plus a separate bug where 24 sliders had their *label* translated but not their *info* tooltip (because the translation dict stored a bare string instead of a `(label, info)` tuple). Both were caught in post-release review, not before ship. For v1.2.2 and beyond: any new `gr.*` component with a `label=` and/or `info=` should be checked off against "is this registered in `UI_TRANSLATIONS`, and if it has `info=`, is the dict entry a tuple (not a bare string)?" before considering the item done — this class of bug is easy to miss visually because the label switching correctly makes it *look* finished.
  - *Status: tooling added.* Wrote `check_i18n_coverage.py`, a static-analysis script that parses `TRANSLATED_COMPONENTS_MAP` (the app's actual registration mechanism — component-variable ↔ translation-key pairs consumed by the language-switch callback, not a per-call lookup at construction time) and `UI_TRANSLATIONS`, then flags any `gr.*` component with non-empty `label=`/`info=` that isn't registered, and any registered component whose `info=` maps to a bare string instead of a tuple. Run as `python3 check_i18n_coverage.py Astro_Processor_Pro.py`. Current codebase passes clean (0 findings) — the tool exists for catching *future* regressions of this same bug class before ship, not as evidence the current code was broken.

### 💬 Feedback

Same as v1.2.1 — direction, not commitment. If the RC-Astro trial license comes through mid-cycle, that item could ship on its own as a fast-follow rather than waiting for the other two. With the StarNet2 and Cosmic Clarity items now resolved, RC-Astro licensing is the only remaining blocker on this track.

---

## 繁體中文

下一版候選優化項目，依建議優先度排序。跟 `ROADMAP.md` 一樣，不是硬性承諾。以下三項內容跟 v1.2.1 計畫裡的版本實質上沒有變化——上一版都沒做，所以是延續過來，不是重寫。

> **更新**：三項延續項目裡，有兩項的查證/調查工作已經完成（StarNet2 CLI 語法已確認；Cosmic Clarity 可行性已對照官方原始碼查證），狀態與程式改動都已用 ✅ 標註在下方對應段落。RC-Astro 那項仍卡在外部授權審核，維持不變。

### 外部工具介接（延續自 v1.2.1）

#### 高優先

- **RC-Astro NoiseXTerminator/StarXTerminator — 完整實機驗證** — 語法已經查證過、也已經上線成範本（`nxt {input} --output {output} --overwrite` / `sxt {input} --output {output} --overwrite`），但還沒有在有效授權下、在這套程式裡真的成功跑過一次。
  - *工作量*：低 — 這是測試任務，不是程式改動；唯一的卡點是試用授權審核要等多久，跟上一版一樣。
  - *相較 v1.2.1 調升優先度*：這次從中優先調到高優先——是剩下三項裡工作量最低的一項，唯一卡住的是外部因素（授權審核），不是我們這邊還要再查證什麼。
  - *狀態：仍卡住，維持不變。* 這次還是沒有可用的授權，所以這項沒辦法比 v1.2.1 更進一步，真的就是還在等試用審核。

#### 中優先

- **StarNet / StarNet2 — 上範本前先確認目前實際旗標** — ✅ **這次已解決。** 查了 StarNet Software 官方最新文件（`starnetastro.com/documentation/starnet/command-line-tool/`）。原本的推測是錯的：StarNet2 **並沒有**沿用舊版 StarNet 的位置參數慣例，而是用具名旗標：`-i/--input`、`-o/--output`，外加可選的 `-m/--mask`、`-n/--unscreen`、`-u/--upsample`、`-e/--eight`、`-s/--stride`、`-w/--weights`。當初促成這項待辦的那則「StarNet2 新增 `--unscreen`」發布說明，確實如猜測是疊加式的可選輸出旗標沒錯——但這不代表底層輸入/輸出慣例也跟著沿用。
  - *工作量*：確認後真的很低——加了一個獨立的新範本項目（`StarNet2（新版，具名旗標）` → `--input {input} --output {output}`），而不是把舊版「位置參數」那個項目改名，因為兩者不能互通。下拉選單裡現在兩項並存。
  - *狀態*：已寫進程式碼（`EXTERNAL_TOOL_PROFILES_STAR`），不再是待辦事項。

#### 低優先

- **Seti Astro Cosmic Clarity — 先做可行性調查，再決定要不要投入支援** — ✅ **這次已解決，而且答案是「可行，但跟原本假設的方式不一樣」。** 沒有只看使用者文件，直接讀了官方原始碼（`SetiAstroCosmicClarity_denoise.py`，`github.com/setiastro/cosmicclarity`）。對原本假設有兩點修正：
  1. 文件講的「選擇 Full 還是 Luminance、選擇強度」其實是 **Tkinter GUI 對話框**，不是卡在 stdin 的 `input()` 提示，跟原本假設不同。更重要的是，原始碼裡有明確的無人值守分支：`process_images()` 只有在 `denoise_strength is None` 時才會跳出 GUI；命令列只要帶了 `--denoise_strength`，就完全不會碰到 GUI 或 stdin。`subprocess.run(cmd, capture_output=True)` **不會**卡住，跟原本擔心的相反。
  2. 但它的輸入/輸出協定跟這套程式目前介接的其他工具都不一樣：Cosmic Clarity 固定從 `<執行檔目錄>/input/` 讀、寫到 `<執行檔目錄>/output/`（相對於執行檔本身的固定資料夾），輸出檔名規則也是固定的（`<原檔名>_denoised.<副檔名>`）。它不接受 `--input`/`--output` 路徑參數，跟這套程式其他外部工具介接假設的單檔 `{input}/{output}` 佔位符慣例不相容。
  - *建議，更新*：原本的建議是「如果查完沒有無人值守模式，就降級成不適合套進通用外部工具介接功能」。查完發現無人值守模式**確實存在**，所以不是降級，而是另外寫了一支專用的介接函式 `run_cosmic_clarity_denoise()`（獨立於 `run_external_image_tool()`），把暫存檔放進工具固定的 `input/` 資料夾、用必要的 `--denoise_strength` 旗標呼叫、再照已知的命名規則從 `output/` 資料夾讀回結果。已接進 `denoise()` 的 `external` 模式，用自動偵測方式分流（只要 `ext_args` 裡有 `--denoise_strength` 字樣，就走專用函式）。
  - *範圍說明*：這次只涵蓋 **Denoise**（`SetiAstroCosmicClarity_denoise.py`），因為使用者文件和這份計畫原本提到的就是這支。`SetiAstroCosmicClarity.py`（Sharpen）用的是同一套固定資料夾 + argparse 慣例，但輸出檔名字尾不同（`_sharpened`），要支援的話需要另外寫一支幾乎一樣的專用函式，這次還沒動手。
  - *狀態*：Denoise 部分已寫進程式碼（`run_cosmic_clarity_denoise()`、`EXTERNAL_TOOL_PROFILES_DENOISE` 新增項目），不再是待辦事項。Sharpen 支援如果之後想做，是一項範圍明確、工作量不大的後續項目。

### 🧹 流程提醒（不是功能項目）

- **新元件的 i18n 檢查清單** — v1.2.1 上線了 4 個完全沒接上語言切換機制的新控制項，另外還有一個獨立的 bug：24 個滑桿的「標題」有翻譯、但「說明文字」沒有（因為翻譯字典存的是單純字串，不是 `(標題, 說明)` 的 tuple）。這兩個都是發版後複查才抓到，不是發版前。建議 v1.2.2 起，任何新增的 `gr.*` 元件只要有 `label=` 或 `info=`，收尾前都先確認「這個有沒有登記進 `UI_TRANSLATIONS`？如果有 `info=`，字典裡存的是不是 tuple（不是單純字串）？」——這類 bug 光用肉眼看很容易漏掉，因為標題有正確變英文，乍看會以為已經做完了。
  - *狀態：已補上工具。* 寫了 `check_i18n_coverage.py`，一支靜態分析腳本，解析這套程式實際的翻譯註冊機制 `TRANSLATED_COMPONENTS_MAP`（元件變數與翻譯 key 的對照表，由語言切換 callback 讀取，而不是每個元件建構時各自查字典）跟 `UI_TRANSLATIONS`，找出兩類問題：帶了非空 `label=`/`info=` 但沒登記進對照表的元件、以及有登記但 `info=` 對應到單純字串而非 tuple 的項目。用法：`python3 check_i18n_coverage.py Astro_Processor_Pro.py`。目前這版程式碼跑起來乾淨（0 項問題）——這支工具是用來擋住之後同類型的回歸，不是說明現在的程式碼有問題。

### 💬 意見回饋

跟 v1.2.1 一樣，這只是方向不是承諾。如果 RC-Astro 的試用授權中途就核發下來，這一項可以自己先單獨出，不用等另外兩項一起。StarNet2 跟 Cosmic Clarity 這兩項現在都解決了，這條軌道上唯一剩下的卡點就是 RC-Astro 的授權審核。
