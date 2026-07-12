# 🌌 Astro Processor Pro

[English](#english) | [繁體中文](#繁體中文)

---

## English

### 📥 Download (Windows)

No Python required — grab the ready-to-use installer from the **[Releases](../../releases)** page and run `AstroProcessorPro_Setup.exe` (~250 MB, CPU + DirectML build). See [Build a Standalone Windows .exe](#️-build-a-standalone-windows-exe-optional) below if you'd rather build it yourself or need a CUDA-enabled build.

> By default, the app saves exported images and config backups to `Documents\AstroProcessorPro\outputs` (writable by any user, regardless of where the app itself is installed). You can change this in the "Output Directory" field before exporting.

> **Latest version: v1.3.3** — adds an "external" mode for Sharpen (Cosmic Clarity Sharpen), mode-aware parameter panels, a working status-bar elapsed-time field with auto-refresh, and a clearer batch-Stop explanation during external-tool calls. See the [v1.3.3 release notes](release_notes_v1.3.3.md) for details.

---

An interactive, high-performance astro-imaging post-processing application built on Gradio. This tool specializes in wide-field astrophotography and Milky Way stack post-processing.

### ✨ Key Features

- **📂 Folder Scan & Drag-Drop Loading**: Easily scan local directories for fast image switching or drag-and-drop `.tif`, `.jpg`, `.png`, and RAW format images.
- **📷 RAW Camera WB Pre-processing**: Full integration with `rawpy` to read RAW formats (`.cr2`, `.nef`, `.arw`, etc.) and preserve camera white balance.
- **🌌 Background Gradient Removal**: Automatically estimates the sky's light-pollution gradient and subtracts it at an adjustable strength, helping reduce the washed-out/hazy look from ambient light. Supports multi-threaded CPU processing and automatic GPU hardware acceleration (**NVIDIA CUDA** & **Windows DirectML** for AMD/Intel GPUs).
- **📊 Real-time Logarithmic RGB Histogram**: Customized pure NumPy/OpenCV rendering engine for near-instant (1–5ms) histogram updates.
- **🖼️ Dual Compare Modes**:
  - **Side by Side**: Show original and processed images side-by-side.
  - **Slider Overlay (Lightroom Style)**: Drag a central split handle to wipe between the original and processed image.
- **⛶ Focus Mode**: Single-click toolbar button to collapse side panels and maximize the preview size.
- **⭐ Star Reduction & Multi-scale Removal**:
  - **Star Shrink**: Erode stars using structuring elements.
  - **Star Removal**: Detects stars and larger halo/cluster regions based on size and shape, uses randomized noise backfills to match surrounding grains, and performs feathered inpainting.
  - **External**: Optionally hand off to a user-installed ML star-removal tool (e.g. legacy StarNet, StarNet2, RC-Astro StarXTerminator CLI — StarNet2 and legacy StarNet use different CLI conventions and have separate preset entries) instead of the built-in shrink/remove algorithms.
- **🧹 Denoising**: "Fast" mode (bilateral filter) for responsive live preview, "Quality" mode (Non-local Means, full float32 precision if `scikit-image` is installed — otherwise falls back gracefully to 8-bit internal processing) for a cleaner final export, or "External" mode to call a user-installed ML denoiser (e.g. DeepSNR, RC-Astro NoiseXTerminator CLI, Seti Astro Cosmic Clarity) — each with its own controls.
- **✨ Sharpening**: Local-contrast Clarity always runs as the built-in unsharp-mask (Cosmic Clarity's official tool has no equivalent "clarity" model, so there's nothing to hand off there). The separate Sharpen sub-step has its own `internal` / `external` mode selector — `internal` keeps the classic GaussianBlur + addWeighted unsharp-mask, `external` calls Cosmic Clarity Sharpen on the already-Clarity'd image. The two never stack, avoiding double-sharpening halos/ringing.
- **🧭 Mode-Aware Parameter Panels**: The Denoise, Star Removal, and Sharpen panels now only show the controls relevant to whichever mode you've selected, instead of always showing every field. (Star Removal's detection sliders and Remove-specific sliders stay visible in every mode, since they're also used to build the optional Starless export layer regardless of your main Star Removal mode.)
- **🎯 Auto Localized Stretch**: Automatically detects textured regions (nebulosity, galaxy structure) via local-contrast analysis and boosts contrast only there, leaving the flat sky background untouched. A mask preview shows exactly what got detected and how strongly. An optional manual region (rectangle or ellipse, with adjustable feathering and boost weight) lets you override or supplement auto-detection for a target it missed.
- **🎨 Presets & Snapshots**: One-click "Milky Way" / "Nebula" / "Heavy Light Pollution" starting points for beginners, plus named Snapshot A/B/C slots that can be saved to and loaded from files, so a favorite setup survives a page refresh.
- **📁 Batch Processing**: Apply the current parameter set to every image in a folder in one go, with per-file progress, elapsed time, and a running ETA, plus a success/failure summary. A "Stop Batch" button lets you cancel mid-run — it finishes whatever file is currently in progress (no half-written outputs), then stops before starting the next one.
- **🔌 External ML Tool Integration**: Denoise, Star Removal, and Sharpen all support an "external" mode that shells out to a command-line tool you've installed and licensed yourself (e.g. DeepSNR, RC-Astro NoiseXTerminator/StarXTerminator CLI, GraXpert, legacy StarNet, StarNet2, Seti Astro Cosmic Clarity Denoise/Sharpen) instead of the built-in algorithms — the app never bundles or redistributes any third-party model. Built-in tool presets auto-fill the correct CLI syntax for supported tools (still editable by hand); tools that append their own file extension (like GraXpert) are supported via an `{output_noext}` placeholder, and tools with a fixed input/output folder convention instead of path arguments (like Cosmic Clarity) are handled by a dedicated integration path behind the same preset dropdown. During batch processing, the Stop flag is checked once per file at the file boundary (no half-written outputs) — if the active step is in "external" mode, the in-progress file's subprocess call (default 180s timeout) still has to finish or time out first, which the UI now calls out explicitly under the Stop button and in the stop-request status message, instead of silently looking stuck.
- **⏸️ Pausable Live Preview**: Toggle live preview on/off — useful when adjusting several parameters in a row, or in external mode where every preview update would otherwise re-invoke the external tool.
- **💾 Full-Resolution Export**: Outputs genuine full-precision results — the pipeline runs end-to-end in float32 and writes true 16-bit TIFFs (not an 8-bit result upscaled into a 16-bit container), plus a high-quality JPEG. Choose which format(s) to write via checkboxes (skip JPEG or TIFF entirely to save export time/disk space on large batches), with the option to also save independent Star Mask and Starless layers.
- **⚙️ Config Backup & Restore**: Export and import your parameters as `.json` files.
- **🖥️ Live System Monitor & Status Bar**: Keep track of CPU, RAM, and GPU VRAM usage right from the UI, with an auto-refreshing readout (togglable). Both the monitor panel and the bottom status bar show an explicit "DirectML active" message for AMD/Intel GPUs, where live usage/VRAM numbers aren't available the way they are for NVIDIA — instead of silently going blank. The status bar's "⏱ Time" field now reflects the most recent export's elapsed time (updated on success and on failure), and the whole status bar auto-refreshes on the same 3-second timer as the monitor panel instead of only updating after loading, exporting, or switching language.

---

### 📦 Installation

#### 1. Install Base Dependencies
```bash
pip install gradio tifffile opencv-python-headless scipy numpy --break-system-packages
```

#### 2. Optional Add-ons (Recommended)
- **RAW File Support**:
  ```bash
  pip install rawpy --break-system-packages
  ```
- **System Monitoring Panel**:
  ```bash
  pip install psutil gputil --break-system-packages
  ```
- **Full-precision "Quality" denoise mode**: without this, `"Quality"` denoise mode still works but internally rounds to 8-bit before/after processing.
  ```bash
  pip install scikit-image --break-system-packages
  ```

#### 3. GPU Hardware Acceleration (Optional)
If a compatible GPU is detected, background gradient removal runs on the GPU:
- **For NVIDIA GPU (CUDA)**:
  ```bash
  pip install torch --break-system-packages
  ```
- **For AMD / Intel / Windows GPU (DirectML)**:
  ```bash
  pip install torch-directml --break-system-packages
  ```

---

### 🚀 Running the App

Run the application using Python:
```bash
python Astro_Processor_Pro.py
```
After launching, your default web browser will open automatically at `http://127.0.0.1:7860`.

---

### 🏗️ Build a Standalone Windows `.exe` (Optional)

Prefer not to install Python at all? You can package this app into a self-contained Windows executable with [PyInstaller](https://pyinstaller.org/).

> **Note:** Pre-built `.exe` binaries are **not included in this repository** — the bundled build (which includes PyTorch/DirectML) can reach several GB, well past what's practical to host directly in a Git repo. Instead, check the **[Releases](../../releases)** page for downloadable installers, or build it yourself using the steps below.

#### 1. Set up a build environment
```bash
python -m venv venv
.\venv\Scripts\activate
pip install gradio tifffile opencv-python-headless scipy numpy rawpy psutil gputil pyinstaller
pip install scikit-image        # optional — full-precision "Quality" denoise mode
pip install torch-directml   # AMD / Intel / Windows GPU
pip install torch            # NVIDIA GPU
```

#### 2. Build with PyInstaller
```bash
python -m PyInstaller --onedir --name="AstroProcessorPro" --noconsole `
  --collect-all gradio --collect-all gradio_client --collect-all safehttpx `
  --collect-all groovy --collect-all hf-gradio --collect-all torch `
  --collect-all torch_directml --collect-all skimage `
  --copy-metadata gradio --copy-metadata gradio_client --copy-metadata safehttpx `
  --copy-metadata groovy --copy-metadata hf-gradio --copy-metadata huggingface_hub `
  --copy-metadata hf-xet --copy-metadata tqdm --copy-metadata pyyaml `
  --copy-metadata packaging --copy-metadata filelock --copy-metadata numpy `
  --copy-metadata scikit-image --copy-metadata imageio --copy-metadata networkx `
  --copy-metadata lazy-loader `
  --copy-metadata tomlkit --copy-metadata fastapi --copy-metadata starlette `
  --copy-metadata uvicorn --copy-metadata python-multipart --copy-metadata semantic-version `
  --copy-metadata pydantic --copy-metadata typer --copy-metadata click `
  --copy-metadata orjson --copy-metadata rich --copy-metadata anyio `
  --copy-metadata httpx --copy-metadata httpcore --copy-metadata h11 `
  --copy-metadata pillow --copy-metadata pandas `
  Astro_Processor_Pro.py
```
The finished app will be in `dist\AstroProcessorPro\`. Distribute the **entire folder** (not just the `.exe`) — it depends on the `_internal\` directory next to it.

> **`scikit-image` is optional**: the `--collect-all skimage` / `--copy-metadata scikit-image imageio networkx lazy-loader` flags above are only needed if you want the bundled `.exe` to carry the full-precision "Quality" denoise path — that's why `scikit-image` also needs to actually be `pip install`-ed in the build venv (step 1) before PyInstaller can collect it. Skipping all of this is fine too; the app just falls back to the pre-existing 8-bit-internal "Quality" mode behavior if `scikit-image` isn't present in the built environment.

> **Size note:** installing `torch-directml` (CPU + DirectML, as above) keeps the final installer around ~250 MB. Installing the CUDA build of `torch` instead bundles NVIDIA's cuDNN/cuBLAS libraries and can push the installer size to several GB.


#### 3. (Optional) Build a proper installer
An [Inno Setup](https://jrsoftware.org/isdl.php) script (`AstroProcessorPro_Setup.iss`) is provided to package `dist\AstroProcessorPro\` into a single `AstroProcessorPro_Setup.exe` installer with Start Menu / desktop shortcuts. Open the `.iss` file in Inno Setup and press **Compile** (`F9`).

#### 4. (Contributors) Check i18n coverage before adding new UI controls
If you're adding a new `gr.*` component with a `label=` and/or `info=`, run `python3 check_i18n_coverage.py Astro_Processor_Pro.py` first — it flags any new control that isn't registered in the language switcher, and any registered tooltip (`info=`) stored as a bare string instead of a translated pair.

---

## 繁體中文

### 📥 下載（Windows）

不需要安裝 Python — 直接到 **[Releases](../../releases)** 頁面下載 `AstroProcessorPro_Setup.exe`（約 250 MB，CPU + DirectML 版本），雙擊安裝即可使用。若想自行打包或需要 CUDA 版本，請參考下方的[打包成獨立 Windows .exe](#️-打包成獨立-windows-exe選用)章節。

> 預設情況下，匯出的圖片與參數備份會存到 `文件\AstroProcessorPro\outputs`（不論程式安裝在哪裡，這個位置都保證可寫入）。若要改存到別的地方，可以在匯出前直接修改「輸出資料夾」欄位。

> **目前版本：v1.3.3** —— 新增銳化的 "external" 模式（Cosmic Clarity Sharpen）、依模式顯示的參數面板、真正接上的狀態列耗時欄位與自動刷新，以及更清楚的批次停止說明（外部工具呼叫期間）。詳見 [v1.3.3 更新說明](release_notes_v1.3.3.md)。

---


基於 Gradio 打造的高互動性、高性能星野天文影像後處理應用程式，特別針對星野攝影、銀河疊圖後製進行優化。

### ✨ 核心功能

- **📂 本機選圖與拖放上傳**：支援掃描本機資料夾快速切換圖片，或直接拖放上傳 `.tif`、`.jpg`、`.png` 及 RAW 格式檔案。
- **📷 RAW 格式與相機白平衡**：整合 `rawpy` 直接讀取單眼相機 RAW 檔（`.cr2`、`.nef`、`.arw` 等），並保留相機原始白平衡。
- **🌌 背景漸層去除（去光害）**：自動估算天空光害漸層，並可依強度扣除，有效緩解城市光害造成的畫面朦朧感。支援 CPU 多執行緒，並自動偵測雙 GPU 加速後端（NVIDIA **CUDA** / Windows AMD & Intel **DirectML**）。
- **📊 即時對數 RGB 直方圖**：採用純 NumPy/OpenCV 渲染，直方圖更新速度提升至 1–5 毫秒。
- **🖼️ 雙對照預覽模式**：
  - **並排顯示**：左右兩張圖分開對照。
  - **滑桿疊圖（Lightroom 風格）**：同一張圖左右拖曳把手，即時切換顯示處理前後的效果。
- **⛶ 專注預覽模式**：一鍵隱藏左右參數與監控面板，將畫面預覽區域最大化。
- **⭐ 星點縮小與多尺度去星**：
  - **星點縮小 (Star Shrink)**：透過形態學侵蝕技術等比例收縮星點。
  - **多尺度去星 (Star Removal)**：依大小與形狀偵測星點與較大範圍的暈光/星團區域，使用羽化遮罩與背景雜訊顆粒回填進行 Inpaint 修補，讓效果更貼近原始星空的顆粒感。
  - **外部工具 (External)**：可選擇改呼叫使用者自行安裝的 ML 去星工具（例如舊版 StarNet、StarNet2、RC-Astro StarXTerminator CLI——StarNet2 跟舊版 StarNet 的 CLI 慣例不同，範本下拉選單裡是各自獨立的項目），取代內建的縮小/去星演算法。
- **🧹 降噪**："fast"（雙邊濾波）模式讓即時預覽保持流暢，"quality"（Non-local Means，若有安裝 `scikit-image` 則全程 float32 全精度，未安裝則自動優雅退回 8-bit 內部處理）模式在最終匯出時提供更乾淨的結果，或選擇 "external" 模式改呼叫使用者自行安裝的 ML 降噪工具（例如 DeepSNR、RC-Astro NoiseXTerminator CLI、Seti Astro Cosmic Clarity），三者各有獨立的控制項。
- **✨ 銳化**：局部對比 Clarity 永遠是內建 unsharp-mask 運算（Cosmic Clarity 官方工具沒有對應的「clarity」模型，沒有東西可以交給外部工具做）。獨立的「銳化」子步驟則有自己的 `internal`/`external` 模式選單——`internal` 沿用原本 GaussianBlur + addWeighted 的 unsharp-mask，`external` 則對已套用 Clarity 的影像呼叫 Cosmic Clarity Sharpen。兩者不會疊加，避免過度銳化造成光暈／振鈴偽影。
- **🧭 依模式顯示參數面板**：降噪、去星、銳化三組面板現在只顯示跟目前選取模式有關的欄位，不再一律全部顯示。（去星的星點偵測滑桿與「去星」專用滑桿在所有模式下都維持顯示，因為不論主模式為何，只要有勾選輸出去星圖層，這些滑桿仍在運作。）
- **🎯 自動局部拉伸**：透過局部對比分析自動偵測有結構的區域（星雲、銀河塵埃帶等），只加強該處對比，天空背景幾乎不受影響。內建遮罩預覽圖，可實際看到偵測到哪裡、加強力道多少。也支援選填的手動框選區域（矩形或橢圓，邊緣羽化與加強權重皆可調），用來覆蓋或補足自動偵測漏掉的目標。
- **🎨 預設集與快照**：一鍵套用「銀河模式」「星雲模式」「重光害」等新手起手式，並提供具名的快照 A/B/C 插槽，可存成檔案、也能從檔案載入，讓常用組合不會因重新整理頁面而消失。
- **📁 批次處理**：一次將目前參數套用到整個資料夾內的所有圖片，逐張顯示進度、耗時與即時 ETA，並於結束後給出成功/失敗摘要。新增「停止批次」按鈕，可隨時中止——目前正在處理的那張圖片一定會正常跑完（不會留下寫到一半的殘缺檔案），才會在下一張開始前停止。
- **🔌 外部 ML 工具介接**：降噪、去星、銳化都支援 "external" 模式，可改呼叫使用者自行安裝、自行取得授權的命令列工具（例如 DeepSNR、RC-Astro NoiseXTerminator/StarXTerminator CLI、GraXpert、舊版 StarNet、StarNet2、Seti Astro Cosmic Clarity Denoise/Sharpen），取代內建演算法——程式本身完全不內建、不重新散布任何第三方模型。內建工具範本可自動帶入對應的 CLI 語法（仍可手動調整）；會自己補副檔名的工具（如 GraXpert）透過 `{output_noext}` 佔位符支援，而輸入/輸出走固定資料夾慣例、不接受路徑參數的工具（如 Cosmic Clarity）則在同一個範本下拉選單背後，改用專用的介接邏輯處理。批次處理時，停止旗標本來就是逐張圖片邊界檢查一次（不會留下寫到一半的殘檔）——但如果正在處理的步驟是 "external" 模式，那張圖的 subprocess 呼叫（預設逾時 180 秒）還是得先跑完或先逾時，介面現在會在停止按鈕下方與停止當下的狀態訊息裡明確說明這件事，不再讓畫面看起來像卡住。
- **⏸️ 可暫停的即時預覽**：可切換即時預覽開關——適合連續調整多個參數，或使用 external 模式時避免每次更新都重複呼叫外部工具。
- **💾 高解析度匯出**：全程 float32 運算，輸出真正全精度的結果——16-bit TIFF 是貨真價實算出來的，不是把 8-bit 結果撐大塞進 16-bit 容器；同時輸出高品質 JPEG。可用勾選框選擇要輸出哪些格式（批次量大時，只需要其中一種格式可省下匯出時間與硬碟空間），也可選擇額外輸出「星點遮罩」與「去星背景圖層」供後續手動疊圖。
- **⚙️ 參數備份與還原**：支援將當前調圖參數匯出為 `.json` 檔案備份，並能隨時載入復原。
- **🖥️ 實時系統監控與狀態列**：介面內建 CPU、記憶體與 GPU 顯存 (VRAM) 使用率監控，支援自動刷新（可關閉）。監控面板與底部狀態列現在都會對 AMD/Intel（DirectML）顯示卡顯示明確的「DirectML 加速中」訊息，取代原本讀不到即時數字時靜默留白的情況。狀態列的「⏱ 時間」欄位現在會反映最近一次匯出的耗時（成功、失敗都會更新），整個狀態列也改用跟監控面板相同的 3 秒計時器自動刷新，不再只靠載入圖片、匯出、切換語言這三個時機點才更新。

---

### 📦 安裝說明

#### 1. 安裝基礎依賴套件
```bash
pip install gradio tifffile opencv-python-headless scipy numpy --break-system-packages
```

#### 2. 安裝選用套件（強烈推薦）
- **單眼 RAW 檔支援**：
  ```bash
  pip install rawpy --break-system-packages
  ```
- **系統資源監控**：
  ```bash
  pip install psutil gputil --break-system-packages
  ```
- **"quality" 降噪模式全精度支援**：不裝的話 "quality" 模式仍可正常使用，只是內部處理前後會先量化成 8-bit。
  ```bash
  pip install scikit-image --break-system-packages
  ```

#### 3. GPU 硬體加速（選用）
安裝後可大幅加速背景漸層去除的估算速度：
- **NVIDIA 顯示卡 (CUDA)**：
  ```bash
  pip install torch --break-system-packages
  ```
- **AMD / Intel / Windows 顯示卡 (DirectML)**：
  ```bash
  pip install torch-directml --break-system-packages
  ```

---

### 🚀 啟動程式

在終端機中執行：
```bash
python Astro_Processor_Pro.py
```
啟動後，瀏覽器會自動打開本機網頁 `http://127.0.0.1:7860`。

---

### 🏗️ 打包成獨立 Windows `.exe`（選用）

如果不想安裝 Python，也可以用 [PyInstaller](https://pyinstaller.org/) 把整個程式打包成免安裝的 Windows 執行檔。

> **注意：** 本 repo **不包含預先打包好的 `.exe`** —— 完整打包後（含 PyTorch/DirectML）體積可能達數 GB，不適合直接放進 Git repo。請至 **[Releases](../../releases)** 頁面下載現成的安裝檔，或依照下方步驟自行打包。

#### 1. 建立打包用的虛擬環境
```bash
python -m venv venv
.\venv\Scripts\activate
pip install gradio tifffile opencv-python-headless scipy numpy rawpy psutil gputil pyinstaller
pip install scikit-image        # 選用 — "quality" 降噪模式全精度支援
pip install torch-directml   # AMD / Intel / Windows 顯示卡
pip install torch            # NVIDIA 顯示卡
```

#### 2. 執行 PyInstaller 打包
```bash
python -m PyInstaller --onedir --name="AstroProcessorPro" --noconsole `
  --collect-all gradio --collect-all gradio_client --collect-all safehttpx `
  --collect-all groovy --collect-all hf-gradio --collect-all torch `
  --collect-all torch_directml --collect-all skimage `
  --copy-metadata gradio --copy-metadata gradio_client --copy-metadata safehttpx `
  --copy-metadata groovy --copy-metadata hf-gradio --copy-metadata huggingface_hub `
  --copy-metadata hf-xet --copy-metadata tqdm --copy-metadata pyyaml `
  --copy-metadata packaging --copy-metadata filelock --copy-metadata numpy `
  --copy-metadata scikit-image --copy-metadata imageio --copy-metadata networkx `
  --copy-metadata lazy-loader `
  --copy-metadata tomlkit --copy-metadata fastapi --copy-metadata starlette `
  --copy-metadata uvicorn --copy-metadata python-multipart --copy-metadata semantic-version `
  --copy-metadata pydantic --copy-metadata typer --copy-metadata click `
  --copy-metadata orjson --copy-metadata rich --copy-metadata anyio `
  --copy-metadata httpx --copy-metadata httpcore --copy-metadata h11 `
  --copy-metadata pillow --copy-metadata pandas `
  Astro_Processor_Pro.py
```
打包完成後會產生在 `dist\AstroProcessorPro\`。發布時請給出**整個資料夾**（而不只是 `.exe`），因為它需要旁邊的 `_internal\` 資料夾才能執行。

> **`scikit-image` 是選用的**：上面的 `--collect-all skimage`／`--copy-metadata scikit-image imageio networkx lazy-loader` 這幾個參數，只有在你希望打包出來的 `.exe` 也帶有全精度「quality」降噪路徑時才需要——這也是為什麼打包用的虛擬環境（步驟 1）要先實際 `pip install scikit-image`，PyInstaller 才抓得到東西可以收集。全部跳過也沒關係，只要打包環境裡沒有 `scikit-image`，程式就會照樣退回原本「quality 模式內部量化成 8-bit」的行為。

> **體積備註：** 安裝 `torch-directml`（CPU + DirectML，如上）打包出來的安裝檔大約 250 MB。若改裝 CUDA 版的 `torch`，會連帶打包 NVIDIA 的 cuDNN/cuBLAS 等函式庫，安裝檔體積可能暴增到數 GB。


#### 3.（選用）做成正式安裝程式
本 repo 附上一份 [Inno Setup](https://jrsoftware.org/isdl.php) 腳本（`AstroProcessorPro_Setup.iss`），可以把 `dist\AstroProcessorPro\` 打包成單一個 `AstroProcessorPro_Setup.exe` 安裝檔，含開始功能表／桌面捷徑。用 Inno Setup 打開 `.iss` 檔後按 **Compile**（`F9`）即可編譯。

#### 4.（貢獻者適用）新增 UI 控制項前先檢查 i18n 覆蓋率
如果你要新增帶 `label=` 或 `info=` 的 `gr.*` 元件，建議先跑一次 `python3 check_i18n_coverage.py Astro_Processor_Pro.py`——它會抓出還沒登記進語言切換機制的新元件，以及已登記但 `info=` 說明文字存的是單純字串（不是翻譯 tuple）的項目。
