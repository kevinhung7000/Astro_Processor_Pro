# 🌌 Astro Processor Pro

[English](#english) | [繁體中文](#繁體中文)

---

## English

### 📥 Download (Windows)

No Python required — grab the ready-to-use installer from the **[Releases](../../releases)** page and run `AstroProcessorPro_Setup.exe` (~250 MB, CPU + DirectML build). See [Build a Standalone Windows .exe](#️-build-a-standalone-windows-exe-optional) below if you'd rather build it yourself or need a CUDA-enabled build.

> By default, the app saves exported images and config backups to `Documents\AstroProcessorPro\outputs` (writable by any user, regardless of where the app itself is installed). You can change this in the "Output Directory" field before exporting.

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
- **🧹 Denoising**: "Fast" mode (bilateral filter) for responsive live preview, or "Quality" mode (Non-local Means) for a cleaner final export — each with its own strength controls.
- **🎯 Auto Localized Stretch**: Automatically detects textured regions (nebulosity, galaxy structure) via local-contrast analysis and boosts contrast only there, leaving the flat sky background untouched. A mask preview shows exactly what got detected and how strongly.
- **🎨 Presets & Snapshots**: One-click "Milky Way" / "Nebula" / "Heavy Light Pollution" starting points for beginners, plus Snapshot A/B/C slots to save and instantly switch between candidate parameter sets during a session.
- **📁 Batch Processing**: Apply the current parameter set to every image in a folder in one go, with per-file progress and a success/failure summary.
- **💾 Full-Resolution Export**: Outputs high-quality JPEG and 16-bit TIFF files, with option to save independent Star Mask and Starless layers.
- **⚙️ Config Backup & Restore**: Export and import your parameters as `.json` files.
- **🖥️ Live System Monitor**: Keep track of CPU, RAM, and GPU VRAM usage right from the UI.

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
pip install torch-directml   # AMD / Intel / Windows GPU
pip install torch            # NVIDIA GPU
```

#### 2. Build with PyInstaller
```bash
python -m PyInstaller --onedir --name="AstroProcessorPro" --noconsole `
  --collect-all gradio --collect-all gradio_client --collect-all safehttpx `
  --collect-all groovy --collect-all hf-gradio --collect-all torch `
  --collect-all torch_directml `
  --copy-metadata gradio --copy-metadata gradio_client --copy-metadata safehttpx `
  --copy-metadata groovy --copy-metadata hf-gradio --copy-metadata huggingface_hub `
  --copy-metadata hf-xet --copy-metadata tqdm --copy-metadata pyyaml `
  --copy-metadata packaging --copy-metadata filelock --copy-metadata numpy `
  --copy-metadata tomlkit --copy-metadata fastapi --copy-metadata starlette `
  --copy-metadata uvicorn --copy-metadata python-multipart --copy-metadata semantic-version `
  --copy-metadata pydantic --copy-metadata typer --copy-metadata click `
  --copy-metadata orjson --copy-metadata rich --copy-metadata anyio `
  --copy-metadata httpx --copy-metadata httpcore --copy-metadata h11 `
  --copy-metadata pillow --copy-metadata pandas `
  Astro_Processor_Pro.py
```
The finished app will be in `dist\AstroProcessorPro\`. Distribute the **entire folder** (not just the `.exe`) — it depends on the `_internal\` directory next to it.

> **Size note:** installing `torch-directml` (CPU + DirectML, as above) keeps the final installer around ~250 MB. Installing the CUDA build of `torch` instead bundles NVIDIA's cuDNN/cuBLAS libraries and can push the installer size to several GB.


#### 3. (Optional) Build a proper installer
An [Inno Setup](https://jrsoftware.org/isdl.php) script (`AstroProcessorPro_Setup.iss`) is provided to package `dist\AstroProcessorPro\` into a single `AstroProcessorPro_Setup.exe` installer with Start Menu / desktop shortcuts. Open the `.iss` file in Inno Setup and press **Compile** (`F9`).

---

## 繁體中文

### 📥 下載（Windows）

不需要安裝 Python — 直接到 **[Releases](../../releases)** 頁面下載 `AstroProcessorPro_Setup.exe`（約 250 MB，CPU + DirectML 版本），雙擊安裝即可使用。若想自行打包或需要 CUDA 版本，請參考下方的[打包成獨立 Windows .exe](#️-打包成獨立-windows-exe選用)章節。

> 預設情況下，匯出的圖片與參數備份會存到 `文件\AstroProcessorPro\outputs`（不論程式安裝在哪裡，這個位置都保證可寫入）。若要改存到別的地方，可以在匯出前直接修改「輸出資料夾」欄位。

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
- **🧹 降噪**："fast"（雙邊濾波）模式讓即時預覽保持流暢，"quality"（Non-local Means）模式則在最終匯出時提供更乾淨的結果，兩者各有獨立的強度控制。
- **🎯 自動局部拉伸**：透過局部對比分析自動偵測有結構的區域（星雲、銀河塵埃帶等），只加強該處對比，天空背景幾乎不受影響。內建遮罩預覽圖，可實際看到偵測到哪裡、加強力道多少。
- **🎨 預設集與快照**：一鍵套用「銀河模式」「星雲模式」「重光害」等新手起手式，並提供快照 A/B/C 插槽，可在同一次工作階段中儲存並即時切換候選參數組合。
- **📁 批次處理**：一次將目前參數套用到整個資料夾內的所有圖片，逐張顯示進度並於結束後給出成功/失敗摘要。
- **💾 高解析度匯出**：一鍵匯出高品質 JPEG 與 16-bit 無損 TIFF。可選額外輸出「星點遮罩」與「去星背景圖層」供後續手動疊圖。
- **⚙️ 參數備份與還原**：支援將當前調圖參數匯出為 `.json` 檔案備份，並能隨時載入復原。
- **🖥️ 實時系統監控**：介面內建 CPU、記憶體與 GPU 顯存 (VRAM) 使用率監控。

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
pip install torch-directml   # AMD / Intel / Windows 顯示卡
pip install torch            # NVIDIA 顯示卡
```

#### 2. 執行 PyInstaller 打包
```bash
python -m PyInstaller --onedir --name="AstroProcessorPro" --noconsole `
  --collect-all gradio --collect-all gradio_client --collect-all safehttpx `
  --collect-all groovy --collect-all hf-gradio --collect-all torch `
  --collect-all torch_directml `
  --copy-metadata gradio --copy-metadata gradio_client --copy-metadata safehttpx `
  --copy-metadata groovy --copy-metadata hf-gradio --copy-metadata huggingface_hub `
  --copy-metadata hf-xet --copy-metadata tqdm --copy-metadata pyyaml `
  --copy-metadata packaging --copy-metadata filelock --copy-metadata numpy `
  --copy-metadata tomlkit --copy-metadata fastapi --copy-metadata starlette `
  --copy-metadata uvicorn --copy-metadata python-multipart --copy-metadata semantic-version `
  --copy-metadata pydantic --copy-metadata typer --copy-metadata click `
  --copy-metadata orjson --copy-metadata rich --copy-metadata anyio `
  --copy-metadata httpx --copy-metadata httpcore --copy-metadata h11 `
  --copy-metadata pillow --copy-metadata pandas `
  Astro_Processor_Pro.py
```
打包完成後會產生在 `dist\AstroProcessorPro\`。發布時請給出**整個資料夾**（而不只是 `.exe`），因為它需要旁邊的 `_internal\` 資料夾才能執行。

> **體積備註：** 安裝 `torch-directml`（CPU + DirectML，如上）打包出來的安裝檔大約 250 MB。若改裝 CUDA 版的 `torch`，會連帶打包 NVIDIA 的 cuDNN/cuBLAS 等函式庫，安裝檔體積可能暴增到數 GB。


#### 3.（選用）做成正式安裝程式
本 repo 附上一份 [Inno Setup](https://jrsoftware.org/isdl.php) 腳本（`AstroProcessorPro_Setup.iss`），可以把 `dist\AstroProcessorPro\` 打包成單一個 `AstroProcessorPro_Setup.exe` 安裝檔，含開始功能表／桌面捷徑。用 Inno Setup 打開 `.iss` 檔後按 **Compile**（`F9`）即可編譯。
