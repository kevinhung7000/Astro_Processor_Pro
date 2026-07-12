# 🛠️ Optimization Plan — v1.3.3

[English](#english) | [繁體中文](#繁體中文)

> 📝 Scope: same method as the v1.3.0 plan — every item below was found by reading `Astro_Processor_Pro.py` directly (function bodies, not just comments/docstrings), cross-checked against what v1.3.2's release notes and plan doc already disclosed as open. No effort estimates below should be read as more than a starting guess.

---

## English

Candidate improvements for v1.3.3, in suggested priority order. Not a strict commitment — same as `ROADMAP.md`.

### Track A — Cosmic Clarity Sharpen: wire into UI/pipeline

This is the direct continuation of the item v1.3.2 explicitly left open: `run_cosmic_clarity_sharpen()` exists and mirrors `run_cosmic_clarity_denoise()`'s folder protocol, but `apply_clarity_and_sharpen()` has no `mode` concept at all today — it only takes `clarity_blur, clarity_strength, sharpen_blur, sharpen_amount`, unlike `denoise()` (which already has `fast`/`quality`/`external` plus `ext_path`/`ext_args`) and star removal (same pattern via `star_ext_path`/`star_ext_args`). So this is more than "flip a switch" — it needs the same scaffolding denoise/star already have, built from scratch for clarity/sharpen.

#### High priority

- ✅ **DONE (v1.3.3)** — ~~Verify Sharpen's actual CLI flags against source before wiring anything.~~ The function's own docstring already disclosed this wasn't done yet: only the documented parameter *semantics* (Stellar/Non-Stellar/Both, strength, linear-image flag) had been checked, not the literal flag spelling, against `SetiAstroCosmicClarity.py` on `github.com/setiastro/cosmicclarity`. Denoise only got wired safely because that verification happened first in v1.2.2 (confirmed the headless branch, confirmed `--denoise_strength` as the headless trigger). Sharpen needed the equivalent pass — confirm the headless trigger flag, confirm real flag names — before it's exposed as a preset in the UI, since a wrong preset string is worse than no preset (silent bad output or a hung Tkinter window from a background subprocess).
  - *Effort*: low-to-medium — mostly a read of the public source, not code changes.

  **✅ Resolution (v1.3.3)**: `SetiAstroCosmicClarity.py` was downloaded and read line-by-line against its `argparse.add_argument` definitions, not just the official docs. Confirmed: headless mode triggers when both `--sharpening_mode` and `--nonstellar_strength` are present (neither has a default in argparse; `--stellar_amount`/`--nonstellar_amount` do have `default=0.9` and aren't gating flags). The real flag set is `--sharpening_mode {"Stellar Only","Non-Stellar Only","Both"}`, `--nonstellar_strength <1-8>`, `--stellar_amount <0-1>`, `--nonstellar_amount <0-1>`, `--disable_gpu`, `--sharpen_channels_separately`, and `--auto_detect_psf`. The output filename convention (`<original>_sharpened<ext>`) matched what `run_cosmic_clarity_sharpen()` already assumed, so no change was needed there. `run_cosmic_clarity_sharpen()`'s validation was tightened from "extra_args must not be empty" to explicitly checking that both required flags are present.

- ✅ **DONE (v1.3.3)** — ~~Add the missing scaffolding to `apply_clarity_and_sharpen()` / the pipeline, mirroring `denoise()`'s existing pattern.~~ Concretely: a mode selector (at minimum `internal` vs `external`), `clarity_ext_path` / `clarity_ext_args` fields (naming to match the existing `denoise_ext_path`/`star_ext_path` convention), an `EXTERNAL_TOOL_PROFILES_SHARPEN` preset dict (parallel to `EXTERNAL_TOOL_PROFILES_DENOISE`), and a call to `run_cosmic_clarity_sharpen()` from inside `finish_pipeline()`'s clarity/sharpen step when external mode is selected, with the same "log and fall through to unmodified image" failure handling `denoise()` already uses for its `external` branch.
  - *Open question*: should external Sharpen **replace** the internal unsharp-mask clarity/sharpen step, or run **in sequence** with it (e.g. internal clarity, then external sharpen, or vice versa)? Denoise and star removal are each a single either/or choice; clarity+sharpen is two effects bundled in one function today, so this needed an explicit decision before implementation, not an assumption baked into the code.
  - *Effort*: medium — touches `apply_clarity_and_sharpen()`'s signature, `finish_pipeline()`'s call site, `PARAM_NAMES`/`DEFAULTS`, `UI_TRANSLATIONS`, the Gradio component block, the collect-params list, and `TRANSLATED_COMPONENTS_MAP` — same shape of change as star removal's external mode, just applied to a function that didn't have a mode radio yet.
  - *Required before shipping*: run `check_i18n_coverage.py` per the v1.3.2 contributor note, since this adds new `label=`/`info=` controls.

  **✅ Resolution (v1.3.3)**: the open question was resolved as follows — external Sharpen does **not** replace the whole `apply_clarity_and_sharpen()` function. Clarity (local contrast) always stays the built-in unsharp-mask regardless of mode, since Cosmic Clarity's official tool has no equivalent "clarity" model — there's nothing to hand off there. Only the sharpening sub-step gets the mode selector: new `sharpen_mode` (`internal`/`external`), where `internal` keeps the existing GaussianBlur + addWeighted unsharp-mask and `external` calls `run_cosmic_clarity_sharpen()` on the already-Clarity'd image. The two never stack (no internal sharpen followed by an external pass), avoiding double-sharpening halos/ringing. Scaffolding landed mirroring `denoise()`: `sharpen_mode` / `sharpen_ext_path` / `sharpen_ext_args` (appended to the end of `PARAM_NAMES`/`DEFAULTS`/`PARAM_COMPONENTS`, without disturbing existing indices), an `EXTERNAL_TOOL_PROFILES_SHARPEN` preset dict (Both / Stellar Only / Non-Stellar Only, all flag-verified), matching `UI_TRANSLATIONS` (zh/en), the Gradio controls (Radio + Textbox + Dropdown with a profile→args `.change()` binding), plus the `finish_pipeline()` call site and `TRANSLATED_COMPONENTS_MAP` entries.
  - *Not run this pass*: `check_i18n_coverage.py` wasn't available in the files provided for this pass, so it couldn't be run automatically. The 4 new `label=`/`info=` controls were manually cross-checked against zh/en, but maintainers should still run the script before shipping further, per the standing contributor note.
  - *Out of scope, unchanged*: this pass only covers Cosmic Clarity Sharpen itself — the RC-Astro license-approval and GraXpert end-to-end-test items tracked in Track B's Low priority section below are untouched.

### Track B — Maintenance (independent, carried findings)

#### Medium priority

- ✅ **DONE (v1.3.3)** — ~~Status bar's "⏱ Time" field has never actually been wired up — it isn't a regression, it just never had a caller that supplies it.~~ `get_status_bar_html(proc_time=None, lang="zh")` already had the parameter and formatted it correctly (`f"{proc_time:.2f}s"`) when given a value. But all three call sites — after `load_btn.click`, after `export_btn.click`, and after `lang_radio.change` — only passed `[state_lang]` as input, never a real elapsed-time number, so `proc_time` stayed `None` and the field was permanently stuck at `"--"`. The timing data already existed nearby and was just never forwarded: `export_fn()` computes `elapsed = time.perf_counter() - t0` and already prints it inside its own status message text — none of that value reached `get_status_bar_html()`.
  - *Suggested first step*: decide which operation's timing the status bar should reflect — most likely the most recent export, since that's already wired via `export_btn.click().then(fn=get_status_bar_html, ...)` and just needs the extra value threaded through. `export_fn()` would need to also return the raw float (not just the formatted message string) so the `.then()` chain can pass it into `get_status_bar_html(proc_time=...)`.
  - *Effort*: low — no new computation needed, just plumbing an already-computed value through an existing `.then()` chain and adjusting `export_fn()`'s return signature and the corresponding `outputs=[...]` list.

  **✅ Resolution (v1.3.3)**: decided the field reflects the most recent export's elapsed time, as suggested above. `export_fn()` now returns a third value, `elapsed` — the raw float seconds, returned on both success and failure (since "how long before it failed" is still useful) — captured by a new `state_last_export_time = gr.State(None)`. The three existing `.then(fn=get_status_bar_html, ...)` call sites (`load_btn.click`, `export_btn.click`, `lang_radio.change`) now pass `state_last_export_time` alongside `state_lang`; `get_status_bar_html(proc_time, lang)`'s signature already accepted exactly these two positional values, so no function-signature change was needed there.

- ✅ **DONE (v1.3.3)** — ~~Status bar (`status_bar_out`) has no periodic refresh at all, unlike the monitor panel right above it.~~ `monitor_html` (the monitor panel) is wired to `monitor_timer = gr.Timer(3, active=True)` via `monitor_timer.tick(fn=get_system_stats_html, outputs=[monitor_html])`, so it refreshes every 3 seconds on its own. `status_bar_out` had no equivalent — it was only recomputed on three discrete triggers. The RAM read itself was correct at the moment it ran (a live `_psutil.virtual_memory()` call, not cached), but between triggers the displayed numbers just sat at whatever they were last, which read as "stuck" even though the read logic itself wasn't broken.
  - *Suggested first step*: reuse the existing `monitor_timer` (or add a second lightweight timer) and tick `get_status_bar_html(state_lang)` into `status_bar_out`, the same pattern already proven for the monitor panel.
  - *Effort*: low — same shape of change as the monitor panel's existing timer wiring, just pointed at a second output.

  **✅ Resolution (v1.3.3)**: reused the existing `monitor_timer` directly rather than adding a second timer — `monitor_timer.tick(fn=get_status_bar_html, inputs=[state_last_export_time, state_lang], outputs=[status_bar_out])`. No new timer, no change to the underlying computation. Turning off the `monitor_auto_refresh` checkbox pauses this tick too, matching the monitor panel's existing behavior.

- ✅ **DONE (v1.3.3)** — ~~Batch "Stop" doesn't actually interrupt a file that's mid-external-tool-call.~~ Confirmed by reading `batch_process_fn()` together with `run_external_image_tool()` / `run_cosmic_clarity_denoise()` / `run_cosmic_clarity_sharpen()`: the stop flag is checked once per file, at the file boundary, exactly as v1.3.2 documented and intended — no half-written output, by design. But if `denoise_mode`/star mode is `"external"` for that batch, the in-progress file's `subprocess.run(..., timeout=_EXTERNAL_TOOL_TIMEOUT_SEC)` call (default 180s) has to either finish or hit its own timeout before the loop ever reaches the next stop-flag check. From a user's perspective, pressing Stop while file N is calling out to NoiseXTerminator/Cosmic Clarity can look identical to Stop not working, for up to three minutes.
  - *Suggested first step*: not a fix yet, just a decision — is a hard-kill of the in-flight subprocess in scope (would need passing the stop flag/event down into `run_external_image_tool()` and using `Popen` + `poll()` instead of blocking `subprocess.run()`, so it can be killed early), or is this staying "won't fix, document it" behavior? Worth writing down explicitly rather than leaving it as an undocumented surprise, at minimum.
  - *Effort*: low if just documenting the current behavior in the UI's Stop-button tooltip; medium-to-high if actually implementing early termination of a running subprocess.

  **✅ Resolution (v1.3.3)**: went with the documentation path, not the `Popen`+`poll()` rewrite — the per-file stop-flag check itself is unchanged, since that's intentional design, not a bug. Added a permanent `batch_stop_hint` (`gr.Markdown`) on the Batch tab explaining that Stop only takes effect once the current file finishes, and that an active `external`-mode step may take up to its timeout (default 180s) to actually halt. `request_batch_stop_fn()` now also takes `denoise_mode`/`star_mode`/`sharpen_mode` as inputs — if any of them is `"external"` at the moment Stop is pressed, a live reminder is appended to the existing "stop requested" status message.

- ✅ **DONE (v1.3.3, not in the original plan — raised mid-pass)** — ~~The external-only fields (`*_ext_path` / `*_ext_profile` / `*_ext_args`) for Denoise, Star Removal, and Sharpen were always visible regardless of the selected mode, which made it easy to assume they needed filling in even under `internal`/`fast`/`quality`/`shrink` modes.~~ This wasn't a carried-forward item from any prior plan — it came up as a direct consequence of adding Sharpen's `internal`/`external` selector above, which made the "every field always shown" pattern more noticeable across all three panels at once.

  **✅ Resolution (v1.3.3)**: added three independent visibility helper functions — `_sharpen_mode_visibility()`, `_denoise_mode_visibility()`, `_star_mode_visibility()` — each carrying a docstring naming exactly which source functions were checked before deciding what's safe to hide.
  - `_sharpen_mode_visibility()`: verified against `apply_clarity_and_sharpen()` that `sharpen_blur`/`sharpen_amount` are only read on the `internal` branch and never touched by the `external` branch — the two field groups are mutually exclusive with no cross-dependency, so both can be shown/hidden cleanly.
  - `_denoise_mode_visibility()`: verified against `denoise()` that `denoise_d`/`sigma_color`/`sigma_space` are only used when `mode=="fast"`, and `denoise_nlm_h`/`nlm_h_color` only when `mode=="quality"` — all three groups (fast/quality/external) are mutually exclusive. The `denoise_nlm_h`/`denoise_nlm_h_color` components were also given `visible=False` at definition time (since the default mode is `"fast"`, not `"quality"`), matching the initial screen state.
  - `_star_mode_visibility()` (more complex, intentionally partial): only the three Shrink-specific fields (`star_shrink_kernel`/`star_shrink_iter`/`star_shrink_strength` — verified against `process_stars()` as `shrink`-only) plus the three existing external fields were switched to mode-conditional visibility. The star-detection fields and the three Remove-specific fields (`star_inpaint_radius`/`star_feather_px`/`star_noise_strength`) were **deliberately left untouched**: reading `finish_pipeline()` showed that whenever `want_layers=True` (via the "generate layer" preview button, or the export/batch "also export Starless layer" checkbox), the app forces a full multi-scale star-removal pass to build that layer — using exactly those fields — regardless of whether the main mode is `shrink`/`none`/`external`. Hiding them under `star_mode` would leave a user in `shrink` mode unable to reach layer-generation controls that are still actively running, which is worse than the original "always show everything" behavior. This is the one asymmetric case among the three, and the reason is documented directly in `_star_mode_visibility()`'s docstring rather than left as an unexplained gap.
  - *Follow-up correction mid-pass*: the first version only hid `external` fields when a non-`external` mode was selected. Testing surfaced the reverse gap — `internal`/`fast`/`quality`/`shrink`-specific fields stayed visible when `external` was selected — so all three helper functions were rewritten to be bidirectional rather than one-directional.

#### Low priority (tracking, not code changes)

- **RC-Astro NoiseXTerminator/StarXTerminator CLI syntax still hasn't been run against a real license.** Unchanged since v1.2.1 — still blocked on trial-license approval, not on anything in this codebase. Carrying forward as-is; no action item beyond "still waiting."
- **GraXpert CLI syntax was confirmed against official docs in v1.2.2 but has never been run end-to-end either**, per the comment above `EXTERNAL_TOOL_PROFILES_DENOISE`. Lower-stakes than the RC-Astro item (GraXpert doesn't require a paid license to test), so if anyone wants to close one of the two outstanding "documented but unverified" external-tool presets, this is the cheaper one to actually try.
- **"Quality" denoise's `nlm_h`/`nlm_h_color` slider mapping is still the v1.3.1 best-effort linear rescale**, not a verified match to the old OpenCV-internal behavior (disclosed at the time). If anyone revisits this, the concrete next step would be a side-by-side comparison at a few fixed slider values — same synthetic gradient/noise image, old 8-bit `cv2.fastNlMeansDenoisingColored` path vs. the current `skimage` float32 path — to either confirm the current rescale is close enough in practice, or derive a better one. Not urgent; it was already flagged as "re-tune by eye" territory, not a correctness bug.

### 💬 Feedback

As before, this is a direction, not a commitment. Track A's two items were ordered as written on purpose — verifying Sharpen's real CLI flags before the scaffolding work, the same order Denoise followed in v1.2.2, so the UI never ships a preset string nobody's confirmed against the actual tool. Track B's items are independent of Track A and of each other; any one could ship on its own, or not at all — the last three are explicitly "tracking only," not asking to be scheduled.

---

## 繁體中文

v1.3.3 候選優化項目，依建議優先度排序。跟 `ROADMAP.md` 一樣，不是硬性承諾。

### A 軌 — Cosmic Clarity Sharpen：接進 UI／pipeline

這是 v1.3.2 明確留下的後續項目：`run_cosmic_clarity_sharpen()` 已經存在，結構也比照 `run_cosmic_clarity_denoise()` 的資料夾協定，但 `apply_clarity_and_sharpen()` 目前完全沒有「模式」這個概念——只吃 `clarity_blur, clarity_strength, sharpen_blur, sharpen_amount` 四個參數，跟已經有 `fast`/`quality`/`external` 加上 `ext_path`/`ext_args` 的 `denoise()`、以及走同一套模式（`star_ext_path`/`star_ext_args`）的去星功能都不一樣。所以這不是「開個開關」就好，而是要從零幫 clarity/sharpen 補上 denoise／去星早就有的那套骨架。

#### 高優先

- ✅ **已完成（v1.3.3）** — ~~動手接線之前，先對照原始碼查證 Sharpen 實際的 CLI 旗標。~~ 函式自己的說明就已經誠實揭露這件事還沒做：目前只確認了官方文件描述的參數*語意*（Stellar/Non-Stellar/Both、強度、是否為線性影像），沒有對照 `github.com/setiastro/cosmicclarity` 上的 `SetiAstroCosmicClarity.py` 逐行核對過旗標的實際拼法。Denoise 之所以能安全接進 UI，是因為這件事在 v1.2.2 就先做過了（確認了 headless 分支、確認 `--denoise_strength` 是觸發 headless 的旗標）。Sharpen 需要走一樣的流程——先確認觸發 headless 模式的旗標、確認真正的旗標名稱——才能放進 UI 當範本，因為一個錯誤的範本字串比完全沒有範本更糟（可能靜默產出錯誤結果，或讓背景 subprocess 卡在跳出的 Tkinter 視窗）。
  - *工作量*：低～中——主要是讀公開原始碼，不是改程式。

  **✅ 解決方案（v1.3.3）**：直接下載並逐行對照 `SetiAstroCosmicClarity.py` 的 `argparse.add_argument` 定義查證，不是只看官方文件。確認：headless 觸發條件是 `--sharpening_mode` 與 `--nonstellar_strength` 兩個旗標同時帶到（這兩個在 argparse 裡沒有預設值；`--stellar_amount`／`--nonstellar_amount` 有 `default=0.9`，不是觸發旗標）。實際旗標拼法為：`--sharpening_mode {"Stellar Only","Non-Stellar Only","Both"}`、`--nonstellar_strength <1-8>`、`--stellar_amount <0-1>`、`--nonstellar_amount <0-1>`、`--disable_gpu`、`--sharpen_channels_separately`、`--auto_detect_psf`。輸出檔名規則（`<原檔名>_sharpened<副檔名>`）跟原本 `run_cosmic_clarity_sharpen()` 的假設一致，不用改。`run_cosmic_clarity_sharpen()` 的驗證邏輯已從「extra_args 不可為空」的寬鬆檢查，改成明確檢查上述兩個必要旗標是否都存在。

- ✅ **已完成（v1.3.3）** — ~~幫 `apply_clarity_and_sharpen()`／pipeline 補上目前缺少的骨架，比照 `denoise()` 既有的做法。~~ 具體來說：一個模式選擇器（至少要有 `internal`/`external` 兩種）、`clarity_ext_path`／`clarity_ext_args` 欄位（命名比照現有 `denoise_ext_path`／`star_ext_path` 的慣例）、一份 `EXTERNAL_TOOL_PROFILES_SHARPEN` 範本字典（比照 `EXTERNAL_TOOL_PROFILES_DENOISE`），並在 `finish_pipeline()` 的 clarity/sharpen 步驟裡，選到 external 模式時呼叫 `run_cosmic_clarity_sharpen()`，失敗處理比照 `denoise()` 的 `external` 分支（記錄訊息、跳過本步驟，直接沿用未處理的圖）。
  - *待確認的問題*：external Sharpen 應該**取代**內建的 unsharp-mask clarity/sharpen 步驟，還是跟它**串接**（例如先跑內建 clarity 再跑外部 sharpen，或反過來）？降噪跟去星都是單一二選一，但 clarity+sharpen 目前是兩個效果綁在同一支函式裡，這件事需要動手前先明確決定，不能寫死在程式裡再說。
  - *工作量*：中——牽涉 `apply_clarity_and_sharpen()` 的函式簽名、`finish_pipeline()` 呼叫處、`PARAM_NAMES`／`DEFAULTS`、`UI_TRANSLATIONS`、Gradio 元件區塊、參數收集清單，以及 `TRANSLATED_COMPONENTS_MAP`——跟去星的 external 模式改動幅度差不多，只是套用在一支目前還沒有模式選單的函式上。
  - *上線前必做*：比照 v1.3.2 給貢獻者的提醒，跑一次 `check_i18n_coverage.py`，因為這會新增帶 `label=`/`info=` 的控制項。

  **✅ 解決方案（v1.3.3）**：開放問題的決定是——external Sharpen **不是**取代整個 `apply_clarity_and_sharpen()`。**Clarity（局部對比）永遠是內建 unsharp-mask 運算**，不受模式影響——Cosmic Clarity 官方工具只有「銳化」的 AI 模型，沒有對應「clarity」這個效果，沒有東西可以交給外部工具做。真正二選一的只有「銳化」這個子步驟：新增 `sharpen_mode`（`internal`/`external`），`internal` 沿用原本 GaussianBlur + addWeighted 的 unsharp-mask；`external` 則對「已套用 clarity」的影像呼叫 `run_cosmic_clarity_sharpen()`。兩者不會疊加銳化兩次（不會內建銳化完再跑一次 external），避免過度銳化造成光暈／振鈴偽影。骨架落地比照 `denoise()` 既有做法：新增 `sharpen_mode` / `sharpen_ext_path` / `sharpen_ext_args` 三個參數（附加在 `PARAM_NAMES`/`DEFAULTS`/`PARAM_COMPONENTS` 尾端，不動到既有索引位置）、`EXTERNAL_TOOL_PROFILES_SHARPEN` 範本字典（三組已驗證旗標的範本：Both / Stellar Only / Non-Stellar Only）、對應的 `UI_TRANSLATIONS`（zh/en）、Gradio 元件（Radio + Textbox + Dropdown，含 profile→args 的 `.change()` 綁定），以及 `finish_pipeline()` 呼叫處與 `TRANSLATED_COMPONENTS_MAP` 的對應項目。
  - *這次沒做的事*：`check_i18n_coverage.py`（v1.3.2 contributor note 提到的腳本）不在這次拿到的檔案裡，無法代為執行；新增的 4 個 `label=`/`info=` 控制項已手動補上 zh/en 對照，但仍建議維護者照慣例跑一次該腳本做最終確認。
  - *範圍外，未變動*：這次只確認並接上了 Cosmic Clarity（Sharpen）本身；下方 Track B「低優先」列出的 RC-Astro 授權審核、GraXpert 實測等項目維持原狀，本次未變動。

### B 軌 — 維護項目（各自獨立，屬於延續發現）

#### 中優先

- ✅ **已完成（v1.3.3）** — ~~狀態列的「⏱ 時間」欄位，從一開始就沒有真正被接上過——不是退化，是根本沒有任何呼叫端傳過實際數值進去。~~ `get_status_bar_html(proc_time=None, lang="zh")` 本身參數都已經準備好了，收到值就會正確格式化成 `f"{proc_time:.2f}s"`。但目前全部三個呼叫點——`load_btn.click` 之後、`export_btn.click` 之後、`lang_radio.change` 之後——都只傳了 `[state_lang]` 進去，從來沒有傳過真正的耗時數字，所以 `proc_time` 永遠是 `None`，這個欄位永遠卡在 `"--"`。耗時資料其實就近在咫尺、只是沒有被轉傳：`export_fn()` 內部算過 `elapsed = time.perf_counter() - t0`，也已經把它印進自己的狀態訊息文字裡——這個 `elapsed` 數值目前都沒有傳到 `get_status_bar_html()`。
  - *建議第一步*：先決定狀態列的時間應該反映哪個操作——最合理的候選是最近一次匯出（`export_btn.click().then(fn=get_status_bar_html, ...)` 這串串接本來就存在，只差把數值一併傳進去）。`export_fn()` 需要額外回傳原始的浮點數（而不是只回傳格式化好的訊息字串），這樣 `.then()` 串接才能把它傳進 `get_status_bar_html(proc_time=...)`。
  - *工作量*：低——不需要新增任何運算，只是把已經算好的數值透過既有的 `.then()` 串接接上，並調整 `export_fn()` 的回傳簽名跟對應的 `outputs=[...]` 清單。

  **✅ 解決方案（v1.3.3）**：決定讓它反映「最近一次匯出」的耗時，如上面建議。`export_fn()` 新增第 3 個回傳值 `elapsed`（原始 float 秒數，成功／失敗都會回傳，失敗時代表失敗前實際花了多久），由新增的 `state_last_export_time = gr.State(None)` 接住。`load_btn.click`、`export_btn.click`、`lang_radio.change` 三個既有呼叫點的 `.then(fn=get_status_bar_html, ...)` 都改成把 `state_last_export_time` 跟 `state_lang` 一起傳進去——`get_status_bar_html(proc_time, lang)` 的參數順序本來就吃這兩個位置，不需要改函式簽名本身。

- ✅ **已完成（v1.3.3）** — ~~狀態列（`status_bar_out`）完全沒有定時刷新機制，跟正上方的監控面板不一樣。~~ 監控面板（`monitor_html`）綁的是 `monitor_timer = gr.Timer(3, active=True)`，透過 `monitor_timer.tick(fn=get_system_stats_html, outputs=[monitor_html])` 每 3 秒自動刷新一次。`status_bar_out` 沒有對應的機制——它只在三個離散的觸發點才會重新計算一次。記憶體本身的讀值邏輯沒有問題（`_psutil.virtual_memory()` 是即時呼叫，不是快取值），但在這些觸發點之間，畫面上顯示的數字就只是上一次觸發當下的值，看起來會像「卡住」，即使讀值邏輯本身完全正常。
  - *建議第一步*：重複使用現有的 `monitor_timer`（或另外加一顆輕量計時器），把 `get_status_bar_html(state_lang)` 也掛上去、輸出到 `status_bar_out`，比照監控面板已經驗證過的做法。
  - *工作量*：低——跟監控面板現有的計時器接線幅度差不多，只是多指向一個輸出目標。

  **✅ 解決方案（v1.3.3）**：直接重用既有的 `monitor_timer`，沒有另外新增計時器——多掛一個 `monitor_timer.tick(fn=get_status_bar_html, inputs=[state_last_export_time, state_lang], outputs=[status_bar_out])`，運算邏輯本身沒有變動。`monitor_auto_refresh` 勾選框關閉時，這個刷新也會一併暫停，行為跟監控面板一致。

- ✅ **已完成（v1.3.3）** — ~~批次「停止」對正在呼叫外部工具的那張圖，實際上沒有作用。~~ 對照讀過 `batch_process_fn()` 跟 `run_external_image_tool()`／`run_cosmic_clarity_denoise()`／`run_cosmic_clarity_sharpen()` 後確認：停止旗標確實是在每張圖片開始前、單張圖片邊界檢查一次，跟 v1.3.2 文件描述、設計初衷完全一致——不會留下寫到一半的殘缺輸出檔。但如果那個批次的降噪／去星模式是 `"external"`，正在處理的那張圖片裡的 `subprocess.run(..., timeout=_EXTERNAL_TOOL_TIMEOUT_SEC)`（預設 180 秒）必須先跑完或先撞到自己的逾時，迴圈才會走到下一次停止旗標檢查。從使用者角度看，在第 N 張圖正在呼叫 NoiseXTerminator／Cosmic Clarity 時按下停止，最長可能有三分鐘看起來跟「停止沒有反應」沒兩樣。
  - *建議第一步*：還不是動手改，先做決定——要不要把「強制中止正在跑的 subprocess」納入範圍（需要把停止旗標／event 傳進 `run_external_image_tool()`，改用 `Popen` + `poll()` 而非會卡住的 `subprocess.run()`，才能提早殺掉）？還是維持現狀、只是把這個行為明確寫進文件？至少應該白紙黑字寫下來，而不是留一個沒說明的意外行為。
  - *工作量*：如果只是在停止按鈕的說明文字裡註明現有行為，工作量低；如果真的要實作提早中止正在跑的 subprocess，工作量中～高。

  **✅ 解決方案（v1.3.3）**：選擇了文件化這條路，沒有動 `subprocess.run` 改 `Popen`+`poll` 的那個「中～高工作量」選項——停止旗標本身逐張圖片檢查一次的行為完全沒變，因為那是設計本身，不是要修的錯誤。批次頁面新增一段固定的 `batch_stop_hint`（`gr.Markdown`），說明停止只在目前這張圖處理完後生效、external 模式下最長可能要等外部工具逾時（預設 180 秒）。`request_batch_stop_fn()` 新增 `denoise_mode`／`star_mode`／`sharpen_mode` 三個輸入，按下停止當下若偵測到任一步驟是 `external`，會在既有的「已送出停止要求」訊息後面即時附加提醒。

- ✅ **已完成（v1.3.3，追加項目，不在原規劃清單內）** — ~~降噪、去星、銳化三組面板各自的 `*_ext_path` / `*_ext_profile` / `*_ext_args` 三個外部工具專用欄位，過去不管選哪個 mode 都一直顯示，容易讓人誤以為 `internal`/`fast`/`quality`/`shrink` 模式下也要填。~~ 這不是延續自任何先前計畫的項目——是這次幫 Sharpen 補上 `internal`/`external` 模式選單之後，順勢讓「每個欄位不管模式一律全部顯示」這個既有模式在三組面板上同時變得更明顯，才臨時提出的。

  **✅ 解決方案（v1.3.3）**：新增三支各自獨立的可見性輔助函式——`_sharpen_mode_visibility()`、`_denoise_mode_visibility()`、`_star_mode_visibility()`——每支的 docstring 都寫清楚了決定隱藏哪些欄位之前實際查過哪些程式碼。
  - `_sharpen_mode_visibility()`：讀過 `apply_clarity_and_sharpen()` 確認 `sharpen_blur`/`sharpen_amount` 只有 internal 分支會用到、external 分支完全不讀，兩組欄位互斥、沒有交叉依賴，可以直接兩邊都隱藏。
  - `_denoise_mode_visibility()`：讀過 `denoise()` 確認 `denoise_d`/`sigma_color`/`sigma_space` 只在 `mode=="fast"` 分支用到、`denoise_nlm_h`/`nlm_h_color` 只在 `mode=="quality"` 分支用到，三組（fast/quality/external）互斥顯示；`denoise_nlm_h`/`denoise_nlm_h_color` 兩個元件定義時也補上 `visible=False`（因為預設 mode 是 `"fast"` 不是 `"quality"`），跟畫面初始狀態對齊。
  - `_star_mode_visibility()`（較複雜，刻意只做了一半）：只把「縮星」三個欄位（`star_shrink_kernel`/`star_shrink_iter`/`star_shrink_strength`，讀過 `process_stars()` 確認只有 `mode=="shrink"` 分支會讀）加上原本就有的 external 三欄，改成只在對應模式時顯示。**刻意沒有**對星點偵測欄位和「去星」專用三欄（`star_inpaint_radius`/`star_feather_px`/`star_noise_strength`）做同樣處理：讀過 `finish_pipeline()` 才發現只要 `want_layers=True`（按「產生圖層」預覽、或匯出/批次勾了「同時輸出去星圖層」），不管主畫面選的是 shrink／none／external，都會強制跑一次完整多尺度去星去產生 starless 圖層，用的正是這兩組欄位——如果照 `star_mode` 隱藏，會讓使用者在 shrink 模式下調不到其實仍在運作的圖層去星參數，比原本「全部顯示」更容易誤導使用者，所以維持原樣不隱藏。這是三組裡唯一「不對稱」的例外，原因記在程式裡 `_star_mode_visibility()` 的 docstring，不是遺漏，是查過程式碼行為後的刻意決定。
  - *中途的補充修正*：第一版只做了「非 external 時隱藏 external 欄位」，測試後發現反向也有缺口——選 external 時 `internal`/`fast`/`quality`/`shrink` 專用欄位沒有跟著隱藏——所以三支輔助函式都改寫成雙向可見性，而不是單向。

#### 低優先（先追蹤，非程式改動）

- **RC-Astro NoiseXTerminator/StarXTerminator 的 CLI 語法仍未在有效授權下實測。** 跟 v1.2.1 一樣沒有進展——卡在試用授權審核，不是這個 codebase 本身的問題。原樣延續追蹤，除了「還在等」之外沒有其他行動項目。
- **GraXpert 的 CLI 語法在 v1.2.2 已對照官方文件確認過，但一樣還沒實機跑過完整流程**，見 `EXTERNAL_TOOL_PROFILES_DENOISE` 上方註解。比起 RC-Astro 那項風險低一些（GraXpert 測試不需要付費授權），如果有人想先解決「文件已查證但未實測」的兩項外部工具範本之一，這項成本比較低、值得優先試試看。
- **「quality」降噪模式的 `nlm_h`／`nlm_h_color` 滑桿換算，目前仍是 v1.3.1 當時「盡力而為」的線性近似換算**，不是跟舊版 OpenCV 內部行為逐項驗證過的對應關係（當時已主動揭露）。如果之後有人想重新檢視，具體下一步可以是在幾個固定滑桿數值上做並排比較——同一張合成漸層/雜訊圖，分別跑舊版 8-bit `cv2.fastNlMeansDenoisingColored` 路徑跟現在的 `skimage` float32 路徑——藉此確認現有的換算實際上夠不夠接近，或者推導出更好的換算方式。不算急件，畢竟當初就已經標註為「建議肉眼微調」的範疇，不是正確性上的錯誤。

### 💬 意見回饋

跟以往一樣，這只是方向，不是承諾。A 軌兩個項目刻意照這個順序寫——查證 Sharpen 真正的 CLI 旗標在動手做骨架**之前**完成，比照 v1.2.2 當時 Denoise 走過的順序，這樣 UI 才不會上線一個沒人對照過實際工具核實過的範本字串。B 軌各項跟 A 軌、也跟彼此互相獨立，任何一項都可以單獨先出，也可以完全不做——最後三項明講只是「先追蹤」，不是在要求排進排程。
