"""
🌌 Astro Processor Pro - 星野/銀河疊圖後製 Gradio 互動介面
======================================================
基於 process_astro_v2.py 的處理邏輯改寫，提供專業的天文影像後處理流程，包含：
  - 本機資料夾快速選圖與瀏覽（或直接拖放上傳各式檔案，如 tif, jpg, png）
  - 支援 RAW 檔直接讀取與相機白平衡前處理（需安裝 rawpy）
  - 所有處理參數即時滑桿調整，並提供「並排顯示」與「滑桿疊圖（Lightroom 風格）」雙模式對照預覽
  - 超高速實時預覽（導入純 NumPy 對數直方圖繪製與背景估計快取，滑動滑桿流暢不卡頓）
  - 支援專注預覽模式（隱藏兩側面板，最大化預覽畫面）
  - 多執行緒 (CPU) 與雙 GPU 後端自動偵測與加速 (NVIDIA CUDA / AMD & Intel DirectML)
  - 全解析度批次匯出 (16-bit TIFF + 高品質 JPEG，可同步輸出星點遮罩與去星背景圖層)
  - 當前參數配置一鍵備份 (JSON 匯出) 與還原 (JSON 載入)
  - 內建 CPU、RAM 與 VRAM 系統監控面板

環境安裝說明：
    # 核心基本依賴安裝：
    pip install gradio tifffile opencv-python-headless scipy numpy --break-system-packages

    # RAW 檔支援（選用，若需處理 RAW 格式）：
    pip install rawpy --break-system-packages

    # 系統監控支援（選用）：
    pip install psutil gputil --break-system-packages

    # 降噪「quality」模式全精度支援（選用；沒裝則自動退回 v1.3.0 之前的
    # 8-bit-in/float-out 行為，功能仍可運作，只是這個模式內部精度受限）：
    pip install scikit-image --break-system-packages

    # GPU 運算硬體加速後端（選用，二選一，偵測成功後可自動加速背景漸層估算）：
    # 1. NVIDIA 顯示卡：
    pip install torch --break-system-packages
    # 2. AMD / Intel / 內顯等支援 DirectML 的 Windows 系統：
    pip install torch-directml --break-system-packages

執行方式：
    python Astro_Processor_Pro.py

啟動後瀏覽器會自動打開本機頁面 http://127.0.0.1:7860
"""


import os
import sys
import json
import base64
import time
import subprocess
import tempfile
import shlex
import glob
import uuid

# --- PyInstaller --noconsole 修正：無終端機時 sys.stdout/stderr 為 None ---
# uvicorn 的 logging 設定會呼叫 stream.isatty()，None 沒有這個方法會直接崩潰，
# 這裡給一個假的 stream 頂替，避免程式在打包成 exe 後啟動就閃退。
if getattr(sys, "frozen", False):
    class _NullStream:
        def write(self, *args, **kwargs):
            pass
        def flush(self, *args, **kwargs):
            pass
        def isatty(self):
            return False

    if sys.stdout is None:
        sys.stdout = _NullStream()
    if sys.stderr is None:
        sys.stderr = _NullStream()

from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import tifffile
import gradio as gr
from scipy.ndimage import gaussian_filter, minimum_filter


try:
    import rawpy
    HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False

# v1.3.0 追加：denoise() 的 "quality" 模式改用 skimage 的 Non-local Means，
# 原生支援 float32，藉此補上 cv2.fastNlMeansDenoisingColored 只吃 8-bit 的限制
# （詳見 denoise() docstring）。純選用依賴——沒裝的話自動退回舊的
# 「內部量化成 8-bit 再跑 OpenCV」行為，"quality" 模式功能不受影響，
# 只是那一步的精度瓶頸會回來。
try:
    from skimage.restoration import denoise_nl_means
    from skimage.color import rgb2lab, lab2rgb
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

cv2.setNumThreads(os.cpu_count())

# ============================================================
# ===================== 選用 GPU 加速後端 =====================
# ============================================================
# 「有能用的 GPU 就用,沒有就自動退回 CPU」的三層 fallback:
#   1. NVIDIA 顯卡 → CUDA(pip install torch 就有,免額外編譯)
#   2. Windows 上的 AMD / Intel 顯卡 → DirectML(pip install torch-directml)
#   3. 以上都沒有,或裝了但初始化失敗 → 完全退回原本的 CPU 多執行緒版本
# 這兩個套件都是「選用」,沒裝也不影響程式其餘功能正常運作。
HAS_TORCH = False
_TORCH_DEVICE = None
_TORCH_BACKEND_NAME = "CPU(未偵測到可用 GPU 後端)"
try:
    import torch
    import torch.nn.functional as _F
    HAS_TORCH = True
    if torch.cuda.is_available():
        _TORCH_DEVICE = torch.device("cuda")
        _TORCH_BACKEND_NAME = f"CUDA - {torch.cuda.get_device_name(0)}"
    else:
        try:
            import torch_directml
            _TORCH_DEVICE = torch_directml.device()
            _TORCH_BACKEND_NAME = "DirectML(AMD/Intel/Windows GPU)"
        except ImportError:
            _TORCH_DEVICE = None
except ImportError:
    HAS_TORCH = False

USE_GPU = HAS_TORCH and (_TORCH_DEVICE is not None)
print(f"[加速後端] {'已啟用 GPU 加速: ' + _TORCH_BACKEND_NAME if USE_GPU else '未啟用 GPU,使用 CPU 多執行緒運算'}")

# ⚠️ DirectML 背景估計旁路（Background Estimation Bypass）
#
# 根本原因：DirectML (AMD/Intel/Windows GPU) 後端的 F.pad(mode='reflect') 算子
# 會「靜默回傳錯誤數值」而不丟任何例外，且此問題無法透過修改呼叫方式迴避——
# 因為 F.pad 本身就是那個 broken 的 DirectML 算子，不論你在外面怎麼包裝。
#
# 表現症狀：背景估計圖在影像邊緣（包含下方地景區）有通道不對稱的數值錯誤，
# 扣除背景後在地平線與地景出現肉眼可見的橘色/綠色彩虹色帶。
#
# 修正策略：DirectML 後端強制讓背景估計走 CPU scipy 路徑（速度與修改前完全一致），
# 其餘 GPU 功能（如果未來有）不受影響。CUDA 後端完全不受限制。
_IS_DIRECTML = USE_GPU and ("DirectML" in _TORCH_BACKEND_NAME)
_USE_GPU_FOR_BG = USE_GPU and not _IS_DIRECTML
if _IS_DIRECTML:
    print("[加速後端] 偵測到 DirectML 後端 — 背景梯度估計強制走 CPU 路徑（避免 DirectML reflect pad 靜默錯誤導致彩虹色帶）")

# GPU 背景估計失敗容忍度：允許「這張圖失敗、下一張圖再試」，
# 只有連續失敗達到門檻才視為 GPU 環境真的有問題，永久退回 CPU（避免每張圖都白白花時間重試一個確定壞掉的 GPU）。
_GPU_BG_FAIL_STREAK = 0
_GPU_BG_FAIL_STREAK_LIMIT = 3

# ③ 全域 ThreadPoolExecutor：避免每次 CPU 背景運算都重新建立/銷毀執行緒池
_BG_POOL = ThreadPoolExecutor(max_workers=4)

# ② 背景漸層快取：key=(img_id, downscale, min_filter, blur_sigma)
#    引入執行緒鎖，確保 Gradio 併發連線時的快取操作安全
#    改用 OrderedDict + 容量上限（LRU），而非「只保留最近 1 張圖、每次 miss 就整個清空」。
#    單人使用時行為不變（反正只會有 1 張圖在跑）；但若未來變成多人同時連線，
#    不同使用者處理不同圖片時就不會互相把對方剛算好的背景快取擠掉。
import threading
from collections import OrderedDict
_BG_CACHE_LOCK = threading.Lock()
_BG_CACHE_MAX_ENTRIES = 8
_BG_CACHE: "OrderedDict" = OrderedDict()   # { key: (bg_full_perchannel, bg_lum) }, LRU-ordered


def _gaussian_kernel1d_torch(sigma, device, dtype):
    """建立與 scipy.ndimage.gaussian_filter 相近的 1D 高斯核(truncate≈4 個標準差)。"""
    radius = max(1, int(round(4.0 * float(sigma))))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(x ** 2) / (2 * float(sigma) ** 2))
    kernel = kernel / kernel.sum()
    return kernel, radius


def _gaussian_blur_torch(x, sigma):
    """可分離高斯模糊（GPU）。

    ⚠️ DirectML（AMD / Intel / Windows GPU）後端的 F.pad(..., mode='reflect') 已知
       會「靜默回傳錯誤數值」而不丟例外，導致 R/G/B 三通道在邊界處理結果不一致，
       經強力 arcsinh 拉伸後放大為肉眼可見的彩虹色帶。

    修正方式：比照 _min_filter_torch 的做法——先手動做 reflect padding，
    再用 padding=0 的 conv2d，完全繞開 DirectML 原生 reflect pad 運算子。
    CUDA 後端與 CPU 路徑行為完全一致，反而更快（少一次 padding kernel launch）。
    """
    if sigma is None or sigma <= 0:
        return x
    kernel, radius = _gaussian_kernel1d_torch(sigma, x.device, x.dtype)
    kx = kernel.view(1, 1, 1, -1)
    ky = kernel.view(1, 1, -1, 1)
    h, w = x.shape[-2], x.shape[-1]
    # X 方向：手動 reflect pad 後以 padding=0 卷積
    pad_w = min(radius, max(0, w - 1))
    x_padded = _F.pad(x, (pad_w, pad_w, 0, 0), mode='reflect')
    x = _F.conv2d(x_padded, kx, padding=0)
    # 若圖太窄導致 pad_w < radius，以 replicate 補齊剩餘邊界
    if pad_w < radius:
        x = _F.pad(x, (radius - pad_w, radius - pad_w, 0, 0), mode='replicate')
    # Y 方向：手動 reflect pad 後以 padding=0 卷積
    pad_h = min(radius, max(0, h - 1))
    x_padded = _F.pad(x, (0, 0, pad_h, pad_h), mode='reflect')
    x = _F.conv2d(x_padded, ky, padding=0)
    if pad_h < radius:
        x = _F.pad(x, (0, 0, radius - pad_h, radius - pad_h), mode='replicate')
    return x



def _min_filter_torch(x, size):
    """用 -maxpool(-x) 實現與 scipy.ndimage.minimum_filter(size=size, mode='reflect') 相近的灰階侵蝕效果。

    修正說明：PyTorch 的 max_pool2d 內建 padding 是用「-inf」補邊(對取負後的訊號來說,
    等於用 +inf 補「最小值」),這跟 scipy 預設的 reflect(鏡像邊界)完全不是同一回事——
    邊界附近算出來的背景值會偏離 CPU 版本。這裡改成先手動做 reflect padding,
    再用 padding=0 的 max_pool2d,讓邊界行為對齊 CPU 版本。"""
    size = max(1, int(round(size)))
    if size % 2 == 0:
        size += 1
    if size <= 1:
        return x
    pad = size // 2
    h, w = x.shape[-2], x.shape[-1]
    pad_h = min(pad, max(0, h - 1))
    pad_w = min(pad, max(0, w - 1))
    x_padded = _F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
    result = -_F.max_pool2d(-x_padded, kernel_size=size, stride=1, padding=0)
    if pad_h < pad or pad_w < pad:
        result = _F.pad(result, (pad - pad_w, pad - pad_w, pad - pad_h, pad - pad_h), mode='replicate')
    return result


def _compute_channel_background_gpu(channels_stack, small_w, small_h, w, h, min_filter_size, blur_sigma, device):
    """把 R/G/B + 亮度 4 張圖一次性疊成一個 batch 丟進 GPU 運算(比逐張搬資料快),
    做 縮小 → 最小值濾波 → 高斯模糊 後搬回 CPU/numpy。

    修正說明：最後「放大回原尺寸」這一步改回用 cv2.resize(INTER_CUBIC)，跟 CPU 版本
    完全一致的插值實作，不再用 torch 的 F.interpolate(mode='bicubic')。原因：
      1. torch 的雙三次插值(取樣錨點、係數)跟 cv2 不是同一套實作，對高對比邊緣
         (例如銀河亮帶)容易產生不同程度的 overshoot/ringing；R/G/B 各自獨立跑,
         三個色版 overshoot 程度不同，扣除背景後就會出現局部彩色條紋。
      2. 在 DirectML(AMD/Intel GPU)後端上，bicubic/reflect pad 等運算子的實作
         並不完整，常見狀況是不丟例外、但算出來的數值就是錯的。
    這一步資料量很小(只是把縮圖放大回全解析度)，交給 CPU 做幾乎不影響總體速度,
    卻能讓 GPU 版跟 CPU 版的背景估計結果幾乎完全一致。"""
    # 效能優化：在 CPU 上先縮小影像，再將資料送往 GPU。
    # 這樣傳輸的資料量從數百 MB (全解析度影像) 驟降至數十 KB，消除 VRAM 頻寬瓶頸。
    small_channels = [
        cv2.resize(channels_stack[i], (small_w, small_h), interpolation=cv2.INTER_AREA)
        for i in range(channels_stack.shape[0])
    ]
    small_stack = np.stack(small_channels, axis=0)

    x = torch.from_numpy(small_stack).to(device=device, dtype=torch.float32).unsqueeze(1)  # (N,1,small_h,small_w)
    small = _min_filter_torch(x, min_filter_size)

    # 高斯模糊在 GPU 上執行（_gaussian_blur_torch 已內建 DirectML 安全的手動 reflect padding，
    # 不依賴 DirectML 原生 F.pad reflect 運算子，消除彩虹色帶根本原因）
    small = _gaussian_blur_torch(small, blur_sigma)
    small_np = small.squeeze(1).to('cpu').numpy().astype(np.float32)


    # 健全性檢查：DirectML 等後端可能不丟例外卻算出 NaN/Inf 或明顯離譜的數值，
    # 一旦偵測到就主動丟例外，讓外層 remove_background_gradient 的 try/except
    # 自動退回 CPU 模式，而不是把壞掉的背景估計值悄悄用下去。
    if not np.all(np.isfinite(small_np)):
        raise RuntimeError("GPU 背景估計結果出現 NaN/Inf，判定為此後端運算不可信。")
    in_min, in_max = float(channels_stack.min()), float(channels_stack.max())
    margin = 0.5 * max(1.0, in_max - in_min)
    if small_np.min() < in_min - margin or small_np.max() > in_max + margin:
        raise RuntimeError("GPU 背景估計結果數值範圍異常，判定為此後端運算不可信。")

    results = np.stack(
        [cv2.resize(small_np[i], (w, h), interpolation=cv2.INTER_CUBIC) for i in range(small_np.shape[0])],
        axis=0,
    )
    return results

RAW_EXTS = ('.cr2', '.cr3', '.nef', '.arw', '.dng', '.raf', '.orf', '.rw2')
IMG_EXTS = ('.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp') + (RAW_EXTS if HAS_RAWPY else ())

# ============================================================
# ===================== 預設輸出路徑 =====================
# ============================================================

def _default_output_base():
    """回傳預設的輸出資料夾路徑。

    以原始碼執行（`python Astro_Processor_Pro.py`）時，沿用專案資料夾底下的
    相對路徑 `outputs/`，方便開發時查看檔案。

    但打包成 PyInstaller `.exe` 並安裝到類似 `C:\\Program Files\\...` 這種
    需要系統管理員權限才能寫入的路徑時，若還是用「相對於程式所在資料夾」的
    `outputs/`，一般使用者權限的行程會在 os.makedirs()/寫檔時直接
    PermissionError（[WinError 5] 存取被拒）。因此偵測到是 PyInstaller 打包的
    frozen 執行檔時，改用使用者「文件」資料夾底下固定的子資料夾，一定可寫入，
    也比較符合「輸出的圖片應該放在使用者文件裡」的直覺。
    """
    if getattr(sys, "frozen", False):
        return os.path.join(os.path.expanduser("~"), "Documents", "AstroProcessorPro", "outputs")
    return os.path.abspath("outputs")


# ============================================================
# ===================== 圖片讀取 / 前處理 =====================
# ============================================================

def load_image_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.tif', '.tiff'):
        img = tifffile.imread(path).astype(np.float32)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.shape[-1] == 4:
            img = img[..., :3]
        max_val = img.max()
        if max_val > 1.5:
            img = img / (65535.0 if max_val > 255 else 255.0)
        return np.clip(img, 0, 1)
    elif ext in RAW_EXTS:
        if not HAS_RAWPY:
            raise RuntimeError("尚未安裝 rawpy，無法讀取 RAW 檔。請執行: pip install rawpy --break-system-packages")
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(output_bps=16, no_auto_bright=True, use_camera_wb=True)
        return np.clip(rgb.astype(np.float32) / 65535.0, 0, 1)
    else:
        bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if bgr is None:
            raise ValueError(f"無法讀取圖片: {path}")
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        if bgr.shape[-1] == 4:
            bgr = bgr[..., :3]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        max_val = rgb.max()
        rgb = rgb / (65535.0 if max_val > 255 else 255.0)
        return np.clip(rgb, 0, 1)


def make_preview_base(img01, max_dim):
    """回傳 (預覽縮圖, 實際縮放比例)。縮放比例 = 縮圖長邊 / 原圖長邊，
    供星點偵測相關參數（核大小、面積、外擴像素等）在預覽時等比例換算之用。"""
    h, w = img01.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(img01, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # 用實際四捨五入後的尺寸回推真實比例，避免因取整數造成的誤差
        actual_scale = new_w / w
        return resized, actual_scale
    return img01.copy(), 1.0


# ============================================================
# ========================= 核心演算法 =========================
# ============================================================

def _compute_channel_background(channel, small_w, small_h, w, h, min_filter_size, blur_sigma):
    """單一色版(或亮度版)的背景估計：縮小→最小值濾波(壓平星點)→高斯模糊→放大回原尺寸。
    獨立成一個函式,是為了讓 3 個色版 + 1 個亮度版可以丟進多執行緒平行跑,
    充分利用多核心 CPU(scipy/OpenCV 這類 C 擴充函式呼叫時會釋放 GIL,
    所以用 threads 就能達到接近真正平行運算的效果,不需要改用 multiprocessing)。"""
    small = cv2.resize(channel, (small_w, small_h), interpolation=cv2.INTER_AREA)
    bg_small = minimum_filter(small, size=min_filter_size)
    bg_small = gaussian_filter(bg_small, sigma=blur_sigma)
    return cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_CUBIC)


def remove_background_gradient(img, downscale, min_filter_size, blur_sigma, subtract_strength):
    global USE_GPU, _USE_GPU_FOR_BG, _BG_CACHE, _GPU_BG_FAIL_STREAK
    h, w, _ = img.shape
    small_h, small_w = max(8, int(h * downscale)), max(8, int(w * downscale))
    min_filter_size = max(1, int(round(min_filter_size)))

    # ② 快取 key：用影像的「內容特徵」取代不安全的 id(img)。
    #    Python 的記憶體回收機制可能讓新圖拿到與已回收舊圖完全相同的位址，
    #    純靠 id() 會觸發錯誤的快取命中（新圖套用舊圖的背景）。
    #    改用 (高, 寬, 四角像素總和) 作為內容指紋，計算成本極低且區分度極高。
    #    subtract_strength 只影響「扣除比例」，不影響背景本體估計，所以不納入 key。
    _content_sig = (
        h, w,
        float(img[0, 0, 0])   + float(img[0, -1, -1]) +
        float(img[-1, 0, 0])  + float(img[-1, -1, -1])
    )
    _cache_key = (_content_sig, round(downscale, 4), min_filter_size, round(blur_sigma, 4))

    with _BG_CACHE_LOCK:
        cached = _BG_CACHE.get(_cache_key)
        if cached is not None:
            _BG_CACHE.move_to_end(_cache_key)  # 標記為最近使用

    if cached is not None:
        bg_full_perchannel, bg_lum = cached
    else:
        luminance = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        bg_full_perchannel = None
        bg_lum = None

        if _USE_GPU_FOR_BG:
            try:
                stack = np.stack(
                    [img[:, :, 0], img[:, :, 1], img[:, :, 2], luminance], axis=0
                ).astype(np.float32)
                results = _compute_channel_background_gpu(
                    stack, small_w, small_h, w, h, min_filter_size, blur_sigma, _TORCH_DEVICE
                )
                bg_full_perchannel = np.stack([results[0], results[1], results[2]], axis=-1)
                bg_lum = results[3]
                _GPU_BG_FAIL_STREAK = 0  # 這次成功了，重置連續失敗計數
            except Exception as e:
                _GPU_BG_FAIL_STREAK += 1
                bg_full_perchannel = None
                bg_lum = None
                if _GPU_BG_FAIL_STREAK >= _GPU_BG_FAIL_STREAK_LIMIT:
                    # 連續失敗達門檻，判定 GPU 環境真的有問題，才永久退回 CPU
                    print(f"[加速後端] GPU 運算連續失敗 {_GPU_BG_FAIL_STREAK} 次,已永久退回 CPU 模式。錯誤訊息: {e}")
                    USE_GPU = False
                    _USE_GPU_FOR_BG = False
                else:
                    # 只有這一張圖退回 CPU，下一張圖仍會再嘗試 GPU
                    print(f"[加速後端] GPU 運算失敗(第 {_GPU_BG_FAIL_STREAK}/{_GPU_BG_FAIL_STREAK_LIMIT} 次),本張圖退回 CPU,下一張圖仍會重試 GPU。錯誤訊息: {e}")

        if bg_full_perchannel is None:
            # ③ bg_downscale 預設 0.06，小圖面積極小（例如 6000×4000 → 360×240）。
            #    在這麼小的資料量上，執行緒池的 context switch 與 GIL 競爭開銷
            #    反而大於平行帶來的收益，改為單執行緒一條龍跑完更快更省資源。
            #    _BG_POOL 保留定義，供未來需要時或其他呼叫點使用。
            bg_channels = [
                _compute_channel_background(
                    img[:, :, c], small_w, small_h, w, h, min_filter_size, blur_sigma
                )
                for c in range(3)
            ]
            bg_full_perchannel = np.stack(bg_channels, axis=-1)
            bg_lum = _compute_channel_background(
                luminance, small_w, small_h, w, h, min_filter_size, blur_sigma
            )

        with _BG_CACHE_LOCK:
            # 保留最近使用的最多 _BG_CACHE_MAX_ENTRIES 筆結果（LRU，避免記憶體無限成長）
            _BG_CACHE[_cache_key] = (bg_full_perchannel, bg_lum)
            _BG_CACHE.move_to_end(_cache_key)
            while len(_BG_CACHE) > _BG_CACHE_MAX_ENTRIES:
                _BG_CACHE.popitem(last=False)  # 丟掉最久未使用的那筆

    # 以下使用快取或剛計算完的 bg_full_perchannel / bg_lum
    luminance = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114

    ratio = img / np.maximum(luminance[:, :, None], 1e-4)
    # 防護：地景/死黑區域的 luminance 極低，ratio 會炸到 10000 倍以上，
    # 乘上 bg_lum 後形成巨大虛假「中性背景」，扣除後在地平線處產生橘/綠/黃彩虹色帶。
    # 合理的色彩比例範圍大約在 [0, 4]（某通道比平均亮 4 倍已屬極端情況）。
    ratio = np.clip(ratio, 0.0, 4.0)
    ratio = gaussian_filter(ratio, sigma=(2, 2, 0))
    bg_full_neutral = bg_lum[:, :, None] * ratio

    signal_above_bg = np.clip(luminance - bg_lum, 0, None)
    confidence = gaussian_filter(signal_above_bg, sigma=blur_sigma * 3)
    conf_ref = np.percentile(confidence, 97)
    confidence = np.clip(confidence / max(conf_ref, 1e-6), 0, 1) ** 0.6
    confidence = confidence[:, :, None]

    bg_full = confidence * bg_full_perchannel + (1 - confidence) * bg_full_neutral
    out = img - bg_full * subtract_strength
    return np.clip(out, 0, None)



def correct_color_cast(img, enable, gain_min, gain_max):
    if not enable:
        return img
    med = np.median(img.reshape(-1, 3), axis=0)
    # 防護：地景剪影等極度死黑情況下，med 通道可能為 0。
    # 限制下限防止除以零或極端色偏（某通道增益暴衝）
    med = np.maximum(med, 1e-4)
    target = med.mean()
    gains = target / med
    gains = np.clip(gains, gain_min, gain_max)
    return np.clip(img * gains[None, None, :], 0, None)


def stretch_dynamic_range(img, black_pct, stretch_factor, white_pct):
    """非線性拉伸。
    修正說明：percentile 改從固定尺度（長邊 ≤ 1200px）的參考圖計算，
    確保縮圖預覽與全解析度匯出使用完全相同的黑點/白點基準，
    避免縮圖預覽因 INTER_AREA 平均掉亮星峰值，導致 white_point 偏低而整體偏白。
    """
    h, w = img.shape[:2]
    # 固定以長邊不超過 1200px 的縮圖計算 percentile（純統計，不影響輸出像素值）
    _STAT_MAX = 1200
    scale_stat = min(1.0, _STAT_MAX / max(h, w, 1))
    if scale_stat < 0.95:
        sw, sh = max(1, int(w * scale_stat)), max(1, int(h * scale_stat))
        stat_img = cv2.resize(img.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        stat_img = img

    black_point = float(np.percentile(stat_img, black_pct))
    stat_clipped = np.clip(stat_img - black_point, 0, None)
    stretched_stat = np.arcsinh(stat_clipped * stretch_factor) / np.arcsinh(stretch_factor)
    white_point = float(np.percentile(stretched_stat, white_pct))

    img = np.clip(img - black_point, 0, None)
    stretched = np.arcsinh(img * stretch_factor) / np.arcsinh(stretch_factor)
    stretched = stretched / max(white_point, 1e-6)
    return np.clip(stretched, 0, 1)



def detect_target_regions(img01, radius=40.0, sensitivity=1.0):
    """自動偵測「有結構的目標區域」(星雲/銀河等)，回傳 0~1 的柔和遮罩。

    原理：用一個偏大的半徑估計局部平均亮度(local_mean)，
    計算原圖與該平均亮度的差異量(detail)，代表該處局部細節/對比的豐富程度——
    平坦的天空背景 detail 接近 0，星雲的雲氣結構、銀河的塵埃帶則有連續且較大範圍的 detail。

    刻意用「大半徑估計 + 再次中尺度平滑」而非直接抓單一像素的高頻雜訊，
    是為了讓單顆亮星這種「小範圍尖峰」不容易被誤判成大片目標區域——
    真正的星雲/銀河結構通常涵蓋較大範圍、有連續的中尺度細節起伏，而不是孤立的尖點。
    """
    h, w = img01.shape[:2]
    lum = img01[:, :, 0] * 0.299 + img01[:, :, 1] * 0.587 + img01[:, :, 2] * 0.114

    # 降到小圖跑，加速運算，同時順便濾掉像素級雜訊，只保留中大尺度結構
    _STAT_MAX = 500
    scale = min(1.0, _STAT_MAX / max(h, w, 1))
    if scale < 0.95:
        sw, sh = max(8, int(w * scale)), max(8, int(h * scale))
        lum_small = cv2.resize(lum, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        lum_small = lum

    r_small = max(2.0, radius * scale)
    local_mean = gaussian_filter(lum_small, sigma=r_small)
    detail = np.abs(lum_small - local_mean)
    structure = gaussian_filter(detail, sigma=r_small * 0.5)  # 中尺度平均，壓掉單顆星造成的尖峰雜訊

    # 用 percentile 正規化到 0~1，避免單一極端值把整張遮罩洗掉
    lo = np.percentile(structure, 5)
    hi = np.percentile(structure, 99)
    mask_small = np.clip((structure - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    mask_small = mask_small ** (1.0 / max(sensitivity, 0.05))  # sensitivity 越高，越容易判定為目標區域

    mask = cv2.resize(mask_small.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    mask = np.clip(gaussian_filter(mask, sigma=max(2.0, radius * 0.15)), 0.0, 1.0)  # 邊緣羽化，避免遮罩硬邊
    return mask


def build_manual_target_mask(h, w, x_pct, y_pct, w_pct, h_pct, weight, feather_pct, shape="rectangle"):
    """把使用者手動框選的區域轉換成 0~1 柔和遮罩，供 apply_local_target_boost 與自動遮罩合併。

    座標系統採用「百分比」而非絕對像素，因此無論套用在全解析度原圖或縮圖預覽上，
    框選的相對位置與大小都一致，不需要額外依 preview_scale 換算。

    shape 支援兩種：
      - "rectangle"：矩形框選（原本的行為）。
      - "ellipse"：橢圓框選，寬高百分比相等時就是正圓，不需要另外做「圓形」選項。

    做法：
      1. 依中心點百分比 (x_pct, y_pct) 與寬高百分比 (w_pct, h_pct) 算出區域的像素邊界/半軸長。
      2. 區域內填入 weight（0~1，代表這塊區域被判定為「目標」的程度），區域外為 0。
      3. 用 gaussian_filter 羽化邊緣（sigma 依區域短邊的 feather_pct% 決定），
         避免手動框選區域出現生硬的邊界（矩形的直角、橢圓的邊緣皆適用）。
    """
    x0 = w * x_pct / 100.0
    y0 = h * y_pct / 100.0
    rw = max(2.0, w * w_pct / 100.0)
    rh = max(2.0, h * h_pct / 100.0)

    mask = np.zeros((h, w), dtype=np.float32)
    w_clip = float(np.clip(weight, 0.0, 1.0))

    if shape == "ellipse":
        # 以 (x0, y0) 為中心、半軸長 (rw/2, rh/2) 的橢圓遮罩；rw == rh 時即為正圓。
        yy, xx = np.ogrid[:h, :w]
        norm = ((xx - x0) / (rw / 2.0)) ** 2 + ((yy - y0) / (rh / 2.0)) ** 2
        mask[norm <= 1.0] = w_clip
    else:
        x_min = int(round(max(0, x0 - rw / 2.0)))
        x_max = int(round(min(w, x0 + rw / 2.0)))
        y_min = int(round(max(0, y0 - rh / 2.0)))
        y_max = int(round(min(h, y0 + rh / 2.0)))
        if x_max > x_min and y_max > y_min:
            mask[y_min:y_max, x_min:x_max] = w_clip

    if feather_pct > 0:
        feather_sigma = max(0.5, min(rw, rh) * (feather_pct / 100.0))
        mask = gaussian_filter(mask, sigma=feather_sigma)

    return np.clip(mask, 0.0, 1.0)


def apply_local_target_boost(img01, enable, strength, radius, sensitivity,
                              manual_enable=False, manual_x_pct=50.0, manual_y_pct=50.0,
                              manual_w_pct=25.0, manual_h_pct=25.0,
                              manual_weight=1.0, manual_feather_pct=20.0,
                              manual_shape="rectangle"):
    """在自動偵測到的目標區域(星雲/銀河等)內加強局部對比，天空背景則幾乎不受影響。

    做法：對亮度做一次大半徑的 unsharp-mask(clarity 概念)，但套用強度依遮罩逐像素加權——
    遮罩值高(有結構的區域)才吃到明顯的對比增強，遮罩值接近 0 的平坦天空背景幾乎不變，
    藉此取代「整張圖統一拉伸」的做法。
    色彩維持方式與 remove_background_gradient() 相同：只算亮度的增減比例，
    再乘回三個色版，避免整體色偏。

    遮罩來源有兩個，取兩者逐像素最大值（聯集）合併：
      1. 自動遮罩：detect_target_regions() 自動偵測有結構的區域（enable 控制是否計算）。
      2. 手動遮罩：build_manual_target_mask() 依使用者框選的矩形位置產生（manual_enable 控制），
         用來覆蓋自動偵測漏掉的微弱目標，或是使用者想強制加強的特定區域。
    手動遮罩可獨立於自動偵測開關使用：即使自動偵測關閉，仍可只用手動框選的區域加強。
    """
    if strength <= 0 or (not enable and not manual_enable):
        return img01, None

    h, w = img01.shape[:2]
    mask = detect_target_regions(img01, radius=radius, sensitivity=sensitivity) if enable else np.zeros((h, w), dtype=np.float32)

    if manual_enable:
        manual_mask = build_manual_target_mask(
            h, w, manual_x_pct, manual_y_pct, manual_w_pct, manual_h_pct,
            manual_weight, manual_feather_pct, shape=manual_shape,
        )
        mask = np.maximum(mask, manual_mask)

    if not np.any(mask > 1e-4):
        return img01, None

    lum = img01[:, :, 0] * 0.299 + img01[:, :, 1] * 0.587 + img01[:, :, 2] * 0.114
    blur = gaussian_filter(lum, sigma=max(2.0, radius))
    detail = lum - blur
    boosted_lum = np.clip(lum + detail * strength * mask, 0.0, 1.0)

    ratio = np.clip(boosted_lum / np.maximum(lum, 1e-4), 0.2, 5.0)
    out = np.clip(img01 * ratio[:, :, None], 0.0, 1.0).astype(np.float32)
    return out, mask



def boost_saturation(img01, sat_boost, bright_boost, r_gain, g_gain, b_gain):
    gains = np.array([r_gain, g_gain, b_gain], dtype=np.float32)
    img01_color = np.clip(img01 * gains[None, None, :], 0, 1).astype(np.float32)

    # ⑤ float32 HSV 路徑：避免 uint8 量化誤差
    # cv2 float32 HSV 規格：H ∈ [0, 360], S ∈ [0, 1], V ∈ [0, 1]
    # 注意：必須確保輸入為 float32（CV_32F），cv2 不接受 float64（CV_64F）
    hsv = cv2.cvtColor(img01_color, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_boost, 0, 1)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * bright_boost, 0, 1)
    rgb01 = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    # v1.3.0：不再在這裡量化成 uint8——這是舊版整條 pipeline 最終輸出精度損失
    # 的最後一步（finish_pipeline() 靠這個函式的回傳值決定最終圖片精度）。
    # 現在維持 float32 [0,1]，由呼叫端決定何時／是否需要轉成 uint8（例如畫面
    # 預覽用，或 JPEG 匯出用），真正需要全精度的 16-bit TIFF 匯出則完全不轉。
    return np.clip(rgb01, 0, 1).astype(np.float32)



def apply_clarity_and_sharpen(img01, clarity_blur, clarity_strength, sharpen_blur, sharpen_amount,
                               sharpen_mode="internal", sharpen_ext_path="", sharpen_ext_args=""):
    """v1.3.0：改為全程 float32 [0,1] 運算（之前是 uint8 [0,255]）。

    clarity_strength / sharpen_amount 都是「原圖與模糊版本之間的差值」乘上的
    比例係數，是相對運算，不受數值範圍是 0–255 或 0–1 影響，所以參數本身不用
    另外換算，跟舊版的觀感是一致的。GaussianBlur 跟 addWeighted 都原生支援
    float32（實測確認過，不像 fastNlMeansDenoisingColored 那樣有 8-bit 限制），
    所以這裡不需要為了 OpenCV 相容性而分段量化。

    舊版在 clarity 算完後、銳化前，會先把中間結果量化成 uint8 一次（`clarity =
    np.clip(clarity, 0, 255).astype(np.uint8)`），這是本函式原本唯一的精度損失
    來源，現在移除了。

    v1.3.3：接上 Track A（Cosmic Clarity Sharpen 外部介接）。設計決定（見
    OPTIMIZATION_PLAN_v1_3_3.md）：Clarity（局部對比）永遠是內建 unsharp-mask
    運算，不受 sharpen_mode 影響——Cosmic Clarity 官方工具只有「銳化」的 AI
    模型，沒有對應「clarity」這個效果，兩者不是同一件事的兩種實作方式，
    所以不適合像 denoise()／去星那樣整段二選一。真正二選一的是「銳化」這
    個子步驟：
        sharpen_mode = "internal"（預設）：沿用原本 GaussianBlur + addWeighted
            的 unsharp-mask 銳化，套用在 clarity 處理完的影像上。
        sharpen_mode = "external"：改呼叫 run_cosmic_clarity_sharpen()（見該
            函式），對 clarity 處理完的影像做 AI 銳化。失敗時直接跳過本步驟、
            回傳「僅套用 clarity、未銳化」的影像，不中斷 pipeline，也不會
            靜默改用 internal 頂替（避免使用者誤以為套用了 external 設定，
            實際上卻是內建演算法的結果）。
    兩種模式都不會疊加銳化兩次（不會內建銳化完再跑一次 external，也不會反過
    來），避免過度銳化造成光暈／振鈴偽影。
    """
    img_f = img01.astype(np.float32)
    blur = cv2.GaussianBlur(img_f, (0, 0), sigmaX=clarity_blur)
    clarity = img_f + (img_f - blur) * clarity_strength
    clarity = np.clip(clarity, 0, 1).astype(np.float32)

    if sharpen_mode == "external":
        out01, err = run_cosmic_clarity_sharpen(clarity, sharpen_ext_path, sharpen_ext_args)
        if err is not None:
            print(f"[外部銳化 external sharpen] 已跳過，改用僅套用 Clarity、未銳化的影像: {err}")
            return clarity
        return np.clip(out01, 0, 1).astype(np.float32)

    blur2 = cv2.GaussianBlur(clarity, (0, 0), sigmaX=sharpen_blur)
    sharpened = cv2.addWeighted(clarity, sharpen_amount, blur2, 1 - sharpen_amount, 0)
    return np.clip(sharpened, 0, 1).astype(np.float32)


# ============================================================
# ============= 方案 A：ML 降噪 / 去星 外部工具介接 ================
# ============================================================
# 設計依據：OPTIMIZATION_PLAN_v1_2_0.md「ML 降噪 / ML 去星」低優先項目下的
# 「方案 A（外部工具介接）技術評估」。本程式不內建、不重新散布任何第三方 ML
# 模型或安裝檔，只負責呼叫使用者電腦上「自行安裝、自行取得授權」的外部命令列
# 工具（例如 RC-Astro NoiseXTerminator / StarXTerminator CLI、DeepSNR、
# Cosmic Clarity 等），授權責任在使用者自己身上。
#
# 資料流：img01(記憶體陣列) → 寫成暫存 16-bit TIFF → subprocess 呼叫外部 CLI
#        → 讀回輸出檔 → 轉回 img01 供後續 pipeline 步驟接手。
_EXTERNAL_TOOL_TIMEOUT_SEC = 180


def run_external_image_tool(img01, exe_path, extra_args, timeout=_EXTERNAL_TOOL_TIMEOUT_SEC):
    """呼叫外部命令列工具處理一張影像（方案 A：外部工具介接）。

    img01: float32 [0,1] RGB numpy 陣列 (H, W, 3)。
    exe_path: 使用者在設定欄位指定的外部工具執行檔路徑（使用者自行安裝/授權）。
    extra_args: 額外命令列參數字串。支援三種佔位符：
        {input}        → 輸入暫存檔完整路徑（含副檔名 .tif）
        {output}       → 輸出暫存檔完整路徑（含副檔名 .tif），呼叫端假設工具會
                          原封不動寫到這個路徑
        {output_noext} → 輸出暫存檔路徑，但「不含副檔名」——給像 GraXpert 這種
                          自己決定輸出副檔名的工具用。用了這個佔位符時，本函式
                          事後會用 glob 搜尋 tmp_dir 底下實際被寫出的檔案，而不是
                          死板檢查單一固定路徑。
        若三種佔位符都沒出現在 extra_args 裡，則相容「exe [options] <輸入檔> <輸出檔>」
        這種多數 CLI 常見慣例，自動把輸入/輸出檔路徑（完整路徑）附加在參數字串最後。

    回傳 (out_img01, error_message)：
        成功時 error_message 為 None；失敗時 out_img01 為 None，error_message
        帶有可直接顯示給使用者看的失敗原因，呼叫端應該「跳過該步驟」而非讓
        整條 pipeline 或整個 batch 中斷（對應複核筆記的錯誤處理與降級策略）。
    """
    if not exe_path or not str(exe_path).strip():
        return None, "尚未設定外部工具執行檔路徑（External tool path not configured）"
    exe_path = str(exe_path).strip()
    if not os.path.isfile(exe_path):
        return None, f"找不到外部工具執行檔：{exe_path}"

    tmp_dir = tempfile.mkdtemp(prefix="astro_ext_")
    in_path = os.path.join(tmp_dir, "input.tif")
    out_path = os.path.join(tmp_dir, "output.tif")
    out_stem = os.path.join(tmp_dir, "output")  # 給 {output_noext} 用，工具自己補副檔名
    used_output_noext = False
    try:
        # 寫成 16-bit TIFF 暫存檔，避免先損失動態範圍再交給外部工具處理
        img16 = (np.clip(img01, 0, 1) * 65535.0).astype(np.uint16)
        tifffile.imwrite(in_path, img16)

        try:
            arg_tokens = shlex.split(extra_args) if extra_args else []
        except ValueError as e:
            return None, f"額外命令列參數格式錯誤：{e}"

        used_output_noext = any("{output_noext}" in t for t in arg_tokens)
        if used_output_noext or any(("{input}" in t) or ("{output}" in t) for t in arg_tokens):
            arg_tokens = [
                t.replace("{output_noext}", out_stem).replace("{output}", out_path).replace("{input}", in_path)
                for t in arg_tokens
            ]
            cmd = [exe_path] + arg_tokens
        else:
            cmd = [exe_path] + arg_tokens + [in_path, out_path]

        try:
            result = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            return None, f"外部工具執行逾時（超過 {timeout} 秒）"
        except FileNotFoundError:
            return None, f"無法執行外部工具：{exe_path}"
        except OSError as e:
            return None, f"呼叫外部工具失敗：{e}"

        if result.returncode != 0:
            stderr_tail = (result.stderr or "").strip()[-300:]
            extra = f"：{stderr_tail}" if stderr_tail else ""
            return None, f"外部工具回傳非 0 錯誤代碼 ({result.returncode}){extra}"

        if used_output_noext:
            # 工具自己決定副檔名，用檔名主體 glob 搜尋實際寫出的檔案
            candidates = [f for f in glob.glob(out_stem + ".*") if os.path.isfile(f)]
            if not candidates:
                return None, "外部工具執行完畢，但找不到輸出檔案（已用 {output_noext} 搜尋對應副檔名），請確認額外參數設定是否正確"
            candidates.sort(key=os.path.getmtime, reverse=True)
            actual_out_path = candidates[0]
        else:
            if not os.path.isfile(out_path):
                return None, "外部工具執行完畢，但找不到輸出檔案，請確認額外參數設定是否正確"
            actual_out_path = out_path

        try:
            out16 = tifffile.imread(actual_out_path)
        except Exception:
            # 有些工具搭配 {output_noext} 時可能吐出非 TIFF 格式，退而用一般影像讀取
            try:
                out16_bgr = cv2.imread(actual_out_path, cv2.IMREAD_UNCHANGED)
                if out16_bgr is None:
                    raise ValueError("cv2.imread 回傳 None")
                out16 = out16_bgr[:, :, ::-1] if out16_bgr.ndim == 3 else out16_bgr
            except Exception as e2:
                return None, f"讀取外部工具輸出檔失敗：{e2}"

        if out16.ndim == 2:
            out16 = np.stack([out16] * 3, axis=-1)
        if out16.shape[-1] == 4:
            out16 = out16[:, :, :3]

        if out16.dtype == np.uint16:
            out01 = out16.astype(np.float32) / 65535.0
        elif out16.dtype == np.uint8:
            out01 = out16.astype(np.float32) / 255.0
        else:
            out01 = np.clip(out16.astype(np.float32), 0.0, 1.0)

        if out01.shape[:2] != img01.shape[:2]:
            out01 = cv2.resize(out01, (img01.shape[1], img01.shape[0]), interpolation=cv2.INTER_CUBIC)

        return np.clip(out01, 0.0, 1.0).astype(np.float32), None
    finally:
        cleanup_targets = [in_path, out_path]
        if used_output_noext:
            cleanup_targets += glob.glob(out_stem + ".*")
        for f in cleanup_targets:
            try:
                if os.path.isfile(f):
                    os.remove(f)
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass


# ── v1.3.2：Cosmic Clarity Sharpen 專用介接（承接 v1.2.2 計畫的後續項目）──
# 官方 SetiAstroCosmicClarity.py（Sharpen）跟已支援的 SetiAstroCosmicClarity_denoise.py
# 一樣，用的是同一套「固定 <執行檔所在目錄>/input、<執行檔所在目錄>/output 資料夾」
# 協定，差別只在輸出檔名字尾是 "_sharpened" 而不是 "_denoised"。以下直接照
# run_cosmic_clarity_denoise() 的結構做一份近似複製。
#
# ✅ v1.3.3 更新：查證狀態已補齊，跟 run_cosmic_clarity_denoise() 一致。已直接
# 讀過 github.com/setiastro/cosmicclarity 上 SetiAstroCosmicClarity.py 的
# 原始碼（非官方文件，是逐行對照 argparse 定義），確認：
#   - headless 觸發條件：process_images() 開頭判斷
#     `sharpening_mode is None or nonstellar_strength is None or
#      stellar_amount is None or nonstellar_amount is None` 才會跳出 PyQt6
#     GUI（SharpeningConfigDialog）；由於 --stellar_amount／--nonstellar_amount
#     在 argparse 裡各自有 default=0.9，不傳也不會是 None，實際上「只要」
#     --sharpening_mode 與 --nonstellar_strength 這兩個沒有預設值的旗標同時
#     帶到，就一定會走 headless、不會跳窗卡住。
#   - 實際旗標拼法（確認於 argparse.add_argument 呼叫）：
#       --sharpening_mode {"Stellar Only","Non-Stellar Only","Both"}（必要，見上）
#       --nonstellar_strength <float 1-8>（必要，見上）
#       --stellar_amount <float 0-1>（可選，預設 0.9）
#       --nonstellar_amount <float 0-1>（可選，預設 0.9）
#       --disable_gpu（可選旗標，不帶值）
#       --sharpen_channels_separately（可選旗標，不帶值）
#       --auto_detect_psf（可選旗標，不帶值）
#   - 輸出檔名規則跟 run_cosmic_clarity_denoise() 一致：
#     <原檔名不含副檔名>_sharpened<原副檔名>（TIFF 輸入會維持副檔名不變），
#     跟下面 out_glob 用的 "*_sharpened.*" 搜尋規則相符，這部分原本就正確、
#     這次只是一併確認。
# 因此下面把原本「只要求 extra_args 非空」的寬鬆檢查，改成比照
# denoise()／run_cosmic_clarity_denoise() 的做法，明確檢查這兩個必要旗標
# 是否都存在，避免使用者帶了其他參數但漏了其中一個，仍然卡在跳出的 GUI
# 視窗上（背景 subprocess 會一直等窗口，直到逾時）。
def run_cosmic_clarity_sharpen(img01, exe_path, extra_args="", timeout=_EXTERNAL_TOOL_TIMEOUT_SEC):
    """呼叫 Seti Astro Cosmic Clarity（Sharpen）處理一張影像。

    img01: float32 [0,1] RGB numpy 陣列 (H, W, 3)。
    exe_path: SetiAstroCosmicClarity（Sharpen）執行檔路徑（使用者自行安裝/授權）。
    extra_args: 直接附加在命令列的旗標字串。必須同時包含 --sharpening_mode
        與 --nonstellar_strength（見上方查證說明），否則視為未設定 headless
        參數，直接跳過本次呼叫，避免卡住等待 GUI。其餘旗標（--stellar_amount /
        --nonstellar_amount / --disable_gpu / --sharpen_channels_separately /
        --auto_detect_psf）為可選。

    回傳 (out_img01, error_message)，慣例同 run_cosmic_clarity_denoise()：
    失敗時回傳原圖給呼叫端自行決定要不要跳過本步驟。
    """
    if not exe_path or not str(exe_path).strip():
        return None, "尚未設定 Cosmic Clarity Sharpen 執行檔路徑"
    exe_path = str(exe_path).strip()
    if not os.path.isfile(exe_path):
        return None, f"找不到 Cosmic Clarity Sharpen 執行檔：{exe_path}"
    _args_str = extra_args or ""
    if "--sharpening_mode" not in _args_str or "--nonstellar_strength" not in _args_str:
        return None, ("額外參數必須同時包含 --sharpening_mode 與 --nonstellar_strength 才會觸發 "
                       "headless 模式，缺一就可能跳出 GUI 視窗導致卡住（未執行）；"
                       "請參考範本或官方 SetiAstroCosmicClarity.py 的旗標說明")

    exe_dir = os.path.dirname(os.path.abspath(exe_path))
    in_dir = os.path.join(exe_dir, "input")
    out_dir = os.path.join(exe_dir, "output")
    try:
        os.makedirs(in_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        return None, f"無法建立 Cosmic Clarity Sharpen 的 input/output 資料夾：{e}"

    stem = f"astro_ext_{uuid.uuid4().hex}"
    in_path = os.path.join(in_dir, stem + ".tif")
    out_glob = os.path.join(out_dir, stem + "_sharpened.*")

    try:
        img16 = (np.clip(img01, 0, 1) * 65535.0).astype(np.uint16)
        tifffile.imwrite(in_path, img16)

        try:
            arg_tokens = shlex.split(extra_args) if extra_args else []
        except ValueError as e:
            return None, f"額外命令列參數格式錯誤：{e}"

        cmd = [exe_path] + arg_tokens
        try:
            result = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            return None, f"Cosmic Clarity Sharpen 執行逾時（超過 {timeout} 秒）"
        except FileNotFoundError:
            return None, f"無法執行 Cosmic Clarity Sharpen：{exe_path}"
        except OSError as e:
            return None, f"呼叫 Cosmic Clarity Sharpen 失敗：{e}"

        if result.returncode != 0:
            stderr_tail = (result.stderr or "").strip()[-300:]
            extra = f"：{stderr_tail}" if stderr_tail else ""
            return None, f"Cosmic Clarity Sharpen 回傳非 0 錯誤代碼 ({result.returncode}){extra}"

        candidates = [f for f in glob.glob(out_glob) if os.path.isfile(f)]
        if not candidates:
            return None, "Cosmic Clarity Sharpen 執行完畢，但在 output/ 資料夾找不到對應輸出檔（檔名規則：<原檔名>_sharpened.<副檔名>）"
        candidates.sort(key=os.path.getmtime, reverse=True)
        out_path = candidates[0]

        try:
            out16 = tifffile.imread(out_path)
        except Exception:
            try:
                out16_bgr = cv2.imread(out_path, cv2.IMREAD_UNCHANGED)
                if out16_bgr is None:
                    raise ValueError("cv2.imread 回傳 None")
                out16 = out16_bgr[:, :, ::-1] if out16_bgr.ndim == 3 else out16_bgr
            except Exception as e2:
                return None, f"讀取 Cosmic Clarity Sharpen 輸出檔失敗：{e2}"

        if out16.ndim == 2:
            out16 = np.stack([out16] * 3, axis=-1)
        if out16.shape[-1] == 4:
            out16 = out16[:, :, :3]

        if out16.dtype == np.uint16:
            out01 = out16.astype(np.float32) / 65535.0
        elif out16.dtype == np.uint8:
            out01 = out16.astype(np.float32) / 255.0
        else:
            out01 = np.clip(out16.astype(np.float32), 0.0, 1.0)

        if out01.shape[:2] != img01.shape[:2]:
            out01 = cv2.resize(out01, (img01.shape[1], img01.shape[0]), interpolation=cv2.INTER_CUBIC)

        return np.clip(out01, 0.0, 1.0).astype(np.float32), None
    finally:
        cleanup_targets = [in_path] + glob.glob(out_glob)
        for f in cleanup_targets:
            try:
                if os.path.isfile(f):
                    os.remove(f)
            except OSError:
                pass


# ── v1.2.2：Seti Astro Cosmic Clarity 專用介接 ─────────────────
# 依 OPTIMIZATION_PLAN_v1.2.2.md 低優先項目要求，實機查證 setiastro/cosmicclarity
# 官方原始碼（SetiAstroCosmicClarity_denoise.py, github.com/setiastro/cosmicclarity）
# 後確認：
#   1. 官方文件描述的「Full or Luminance 互動選單」是 Tkinter GUI（不是卡在 stdin
#      的 input() 提示），而且原始碼裡有明確的 headless 分支：
#          if denoise_strength is None:      # 沒帶 --denoise_strength 才跳 GUI
#              ... get_user_input() ...      # Tkinter 對話框
#          else:                             # headless path
#              ...
#      → 只要在命令列帶 --denoise_strength，就完全不會跳出視窗，可以正常用
#        subprocess 呼叫，不會像原本假設的那樣卡住。
#   2. 但它的輸入/輸出協定跟 run_external_image_tool() 假設的「單一輸入檔＋單一
#      輸出檔路徑」慣例不同：Cosmic Clarity 的 input_dir/output_dir 是寫死在程式
#      裡的 <執行檔所在目錄>/input、<執行檔所在目錄>/output，不能用 --input/--output
#      指定成別的路徑；輸出檔名規則固定是「<輸入檔主檔名>_denoised.<副檔名>」。
#      所以不能直接套用通用 {input}/{output} 佔位符機制，需要專用的
#      run_cosmic_clarity_denoise()。
#   3. 因此原計畫「先看原始碼再決定要不要投入支援」的建議事項已有明確答案：
#      可行，不是「不適合套進通用外部工具介接功能」——只是需要獨立的資料流
#      （寫進 exe 旁的 input/ 資料夾、從 output/ 資料夾照命名規則讀回），而非
#      「不適合」。此區塊當初只涵蓋 Denoise（官方文件明確提到的那支）；Sharpen
#      (SetiAstroCosmicClarity.py) 走的是同一套 exe_dir/input、exe_dir/output
#      固定資料夾 + argparse 慣例，但輸出檔名是 "_sharpened" 字尾。
#      v1.3.2 更新：Sharpen 版本的近似複製函式 run_cosmic_clarity_sharpen()
#      已經補上（見下方，緊接在 run_cosmic_clarity_denoise() 之後）——查證
#      程度比 Denoise 版本淺，實際 headless CLI 旗標名稱尚未逐行核對原始碼，
#      詳見該函式開頭的說明。目前尚未接進 UI／pipeline（apply_clarity_and_sharpen()
#      還沒有對應的 ext_path/ext_args 參數與下拉選單），需要的話可以後續再串接，
#      串法可比照 denoise() 呼叫 run_cosmic_clarity_denoise() 的方式。
def run_cosmic_clarity_denoise(img01, exe_path, extra_args="", timeout=_EXTERNAL_TOOL_TIMEOUT_SEC):
    """呼叫 Seti Astro Cosmic Clarity（Denoise）處理一張影像。

    img01: float32 [0,1] RGB numpy 陣列 (H, W, 3)。
    exe_path: SetiAstroCosmicClarity_denoise 執行檔路徑（使用者自行安裝/授權）。
    extra_args: 直接附加在命令列的旗標字串，例如：
        "--denoise_strength 0.5 --denoise_mode luminance"
        "--denoise_strength 0.5 --denoise_mode full --disable_gpu"
        必須至少包含 --denoise_strength，否則官方腳本會判定成沒有走 headless
        path，改跳出 Tkinter GUI 視窗等使用者操作，subprocess 呼叫端會卡住
        直到逾時。

    回傳 (out_img01, error_message)，慣例同 run_external_image_tool()：
    失敗時回傳原圖給呼叫端自行決定要不要跳過本步驟。
    """
    if not exe_path or not str(exe_path).strip():
        return None, "尚未設定 Cosmic Clarity 執行檔路徑"
    exe_path = str(exe_path).strip()
    if not os.path.isfile(exe_path):
        return None, f"找不到 Cosmic Clarity 執行檔：{exe_path}"
    if "--denoise_strength" not in extra_args:
        return None, "額外參數必須包含 --denoise_strength，否則會跳出 GUI 視窗導致卡住（未執行）"

    exe_dir = os.path.dirname(os.path.abspath(exe_path))
    in_dir = os.path.join(exe_dir, "input")
    out_dir = os.path.join(exe_dir, "output")
    try:
        os.makedirs(in_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        return None, f"無法建立 Cosmic Clarity 的 input/output 資料夾：{e}"

    stem = f"astro_ext_{uuid.uuid4().hex}"
    in_path = os.path.join(in_dir, stem + ".tif")
    out_glob = os.path.join(out_dir, stem + "_denoised.*")

    try:
        img16 = (np.clip(img01, 0, 1) * 65535.0).astype(np.uint16)
        tifffile.imwrite(in_path, img16)

        try:
            arg_tokens = shlex.split(extra_args) if extra_args else []
        except ValueError as e:
            return None, f"額外命令列參數格式錯誤：{e}"

        cmd = [exe_path] + arg_tokens
        try:
            result = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            return None, f"Cosmic Clarity 執行逾時（超過 {timeout} 秒）"
        except FileNotFoundError:
            return None, f"無法執行 Cosmic Clarity：{exe_path}"
        except OSError as e:
            return None, f"呼叫 Cosmic Clarity 失敗：{e}"

        if result.returncode != 0:
            stderr_tail = (result.stderr or "").strip()[-300:]
            extra = f"：{stderr_tail}" if stderr_tail else ""
            return None, f"Cosmic Clarity 回傳非 0 錯誤代碼 ({result.returncode}){extra}"

        candidates = [f for f in glob.glob(out_glob) if os.path.isfile(f)]
        if not candidates:
            return None, "Cosmic Clarity 執行完畢，但在 output/ 資料夾找不到對應輸出檔（檔名規則：<原檔名>_denoised.<副檔名>）"
        candidates.sort(key=os.path.getmtime, reverse=True)
        out_path = candidates[0]

        try:
            out16 = tifffile.imread(out_path)
        except Exception:
            try:
                out16_bgr = cv2.imread(out_path, cv2.IMREAD_UNCHANGED)
                if out16_bgr is None:
                    raise ValueError("cv2.imread 回傳 None")
                out16 = out16_bgr[:, :, ::-1] if out16_bgr.ndim == 3 else out16_bgr
            except Exception as e2:
                return None, f"讀取 Cosmic Clarity 輸出檔失敗：{e2}"

        if out16.ndim == 2:
            out16 = np.stack([out16] * 3, axis=-1)
        if out16.shape[-1] == 4:
            out16 = out16[:, :, :3]

        if out16.dtype == np.uint16:
            out01 = out16.astype(np.float32) / 65535.0
        elif out16.dtype == np.uint8:
            out01 = out16.astype(np.float32) / 255.0
        else:
            out01 = np.clip(out16.astype(np.float32), 0.0, 1.0)

        if out01.shape[:2] != img01.shape[:2]:
            out01 = cv2.resize(out01, (img01.shape[1], img01.shape[0]), interpolation=cv2.INTER_CUBIC)

        return np.clip(out01, 0.0, 1.0).astype(np.float32), None
    finally:
        cleanup_targets = [in_path] + glob.glob(out_glob)
        for f in cleanup_targets:
            try:
                if os.path.isfile(f):
                    os.remove(f)
            except OSError:
                pass


# ── v1.2.1 主線 A：外部工具範本對照表 ─────────────────────────
# 純 UI 便利性用途：選了範本，自動把對應「額外命令列參數」字串帶進欄位，
# 不需要使用者自己記或手打各家工具的 CLI 語法。不影響 run_external_image_tool()
# 本身的呼叫邏輯，也不會被寫進 PARAM_NAMES/DEFAULTS（純粹是填欄位用的捷徑，
# 不是需要跟著參數快照/設定檔一起保存的處理參數）。
#
# 語法查證狀態（v1.2.2 更新）：
#   DeepSNR                    — ✅ 使用者實機驗證成功（-i/-o 短旗標 + -m 模型版本 +
#     -s stride；-m/-s 沒有固定「正確答案」，依圖片內容/硬體可能需要自行調整，
#     這裡的 2 / 32 是目前驗證過可用的組合，不代表官方建議預設值）
#   RC-Astro NoiseXTerminator/  — 語法已對照官方文件確認，仍未在有效授權下實測——
#     StarXTerminator               這項還是卡在等試用授權審核，跟 v1.2.1 一樣。
#   GraXpert                   — 語法已對照官方文件確認，尚未實測；{output_noext}
#     機制是為了它專門加的
#   StarNet2                   — ✅ v1.2.2 已對照官方文件（starnetastro.com/
#     documentation/starnet/command-line-tool/）確認：StarNet2 用的是具名旗標
#     `-i/--input`、`-o/--output`（外加可選的 `-m/--mask`、`-n/--unscreen`、
#     `-u/--upsample`、`-e/--eight`），**不是**位置參數——跟舊版 StarNet 的慣例
#     不同，原本「應該會沿用舊版慣例」的推測不成立。因此下面另開一個獨立的
#     StarNet2 範本項目，不能沿用「舊版，位置參數」那一項。
#   Cosmic Clarity（Denoise）  — ✅ v1.2.2 已直接查證 setiastro/cosmicclarity 原始碼，
#     確認有 headless CLI 路徑，但走的是「固定 input/output 資料夾」協定，不是
#     單檔 {input}/{output} 路徑慣例，所以**不透過**這個字典/run_external_image_tool()，
#     改用專用的 run_cosmic_clarity_denoise()（見上方函式與其開頭的說明區塊）。
#     這裡列出來只是為了在下拉選單提示使用者「這台工具走另一套路徑，選了會用
#     不同的呼叫方式」。
EXTERNAL_TOOL_PROFILES_DENOISE = {
    "自訂 / Custom": "",
    "DeepSNR": "-i {input} -o {output} -m 2 -s 32",
    "RC-Astro NoiseXTerminator": "nxt {input} --output {output} --overwrite",
    "GraXpert": "-cli -cmd denoising {input} -output {output_noext}",
    "Seti Astro Cosmic Clarity（Denoise，走專用資料夾協定）/ "
    "Cosmic Clarity Denoise (dedicated folder-based protocol)":
        "--denoise_strength 0.5 --denoise_mode luminance",
}

EXTERNAL_TOOL_PROFILES_STAR = {
    "自訂 / Custom": "",
    "RC-Astro StarXTerminator": "sxt {input} --output {output} --overwrite",
    "StarNet（舊版，位置參數）/ Legacy, Positional Args": "",
    "StarNet2（新版，具名旗標）/ StarNet2, Named Flags": "--input {input} --output {output}",
}

# v1.3.3：Cosmic Clarity Sharpen 專用範本字典（比照 EXTERNAL_TOOL_PROFILES_DENOISE
# 裡「Cosmic Clarity（Denoise）」那一項的做法）。跟 Denoise 一樣走「固定
# input/output 資料夾」協定，不透過 run_external_image_tool()，改由
# apply_clarity_and_sharpen() 在 sharpen_mode == "external" 時直接呼叫
# run_cosmic_clarity_sharpen()。這裡列出來只是為了在下拉選單提示使用者
# 可用的旗標組合。
#
# ✅ 語法查證狀態（v1.3.3）：已直接讀過 setiastro/cosmicclarity 原始碼確認，
# 見 run_cosmic_clarity_sharpen() 開頭的說明區塊——--sharpening_mode 與
# --nonstellar_strength 為觸發 headless 模式的必要旗標，其餘為可選。
EXTERNAL_TOOL_PROFILES_SHARPEN = {
    "自訂 / Custom": "",
    "Seti Astro Cosmic Clarity（Sharpen，走專用資料夾協定，Both 模式）/ "
    "Cosmic Clarity Sharpen (dedicated folder-based protocol, Both mode)":
        "--sharpening_mode Both --nonstellar_strength 3 --stellar_amount 0.5 --nonstellar_amount 0.5",
    "Seti Astro Cosmic Clarity（Sharpen，僅 Stellar）/ "
    "Cosmic Clarity Sharpen (Stellar Only)":
        "--sharpening_mode \"Stellar Only\" --nonstellar_strength 3 --stellar_amount 0.5",
    "Seti Astro Cosmic Clarity（Sharpen，僅 Non-Stellar）/ "
    "Cosmic Clarity Sharpen (Non-Stellar Only)":
        "--sharpening_mode \"Non-Stellar Only\" --nonstellar_strength 3 --nonstellar_amount 0.5",
}


def denoise(img01, enable, mode, d, sigma_color, sigma_space, nlm_h, nlm_h_color,
            ext_path="", ext_args=""):
    """降噪。輸入/輸出皆為 float32 [0,1]（v1.3.0 起，之前是 uint8 [0,255]，
    詳見 finish_pipeline() 開頭的 v1.3.0 說明）。

    mode = "fast"（預設）：雙邊濾波(bilateralFilter)，速度快，適合即時預覽與一般使用。
        cv2.bilateralFilter 原生支援 8u 與 32f 兩種格式（不支援 16u），直接在
        float32 上執行，不需要中途量化成 8-bit。sigmaColor 原本是針對 0–255
        範圍調校的參數，這裡除以 255 換算成 0–1 範圍下等效的門檻，維持跟舊版
        相同的相對降噪強度／參數觀感。
    mode = "quality"：Non-local Means 演算法，會在整張圖搜尋相似的小區塊來平均，
        降噪效果通常比雙邊濾波乾淨、更能保留細節邊緣，但運算量遠高於雙邊濾波
        （一般會慢上數倍到十倍以上，v1.3.1 換成 skimage 實作後更是三個通道分開跑，
        比原本 OpenCV 版本又更慢一截——細節見下方），較適合最終高解析度匯出而非即時預覽。

        v1.3.1 更新：改用 skimage.restoration.denoise_nl_means，原生支援 float32，
        補上 v1.3.0 遺留的已知限制（原本用的 cv2.fastNlMeansDenoisingColored 只吃
        CV_8UC3/4，沒有 16-bit/float 版本，內部一定要先量化成 8-bit 才能跑）。
        做法：先轉到 CIELAB 色彩空間（跟 OpenCV 原本內部做的事一樣——L 通道跟
        a/b 色度通道分開跑，各自套用 nlm_h／nlm_h_color 對應的強度），三個通道都
        在 float32 全精度下個別呼叫 denoise_nl_means，再轉回 RGB。全程沒有任何
        8-bit 量化步驟。
          - patch_size=7、patch_distance=10，對應原本 OpenCV 呼叫用的
            templateWindowSize=7、searchWindowSize=21（21 = 2×10+1）。
          - nlm_h／nlm_h_color 滑桿數值是舊版針對 0-255 尺度校準的，這裡做了
            最佳近似換算（L 用 ×100/255 換算到 skimage Lab 的 0-100 尺度；
            a/b 因為換算後量級跟原本 OpenCV 內部 0-255 尺度的 a/b 差不多，
            數值直接沿用不額外縮放）——這是「盡力而為的近似換算」，不是
            逐像素驗證過的精確等價，實際降噪感覺可能跟舊版有些微落差，
            建議切換後依實際效果微調滑桿。
          - ⚠️ 若未安裝 skimage（`HAS_SKIMAGE=False`），自動退回 v1.3.0 之前
            的行為：內部量化成 8-bit 呼叫 cv2.fastNlMeansDenoisingColored，
            跑完再轉回 float32。這不會讓程式壞掉或跳錯誤，只是那一步的
            精度瓶頸會回來，且僅限使用者主動選擇 "quality" 模式時才有影響
            （預設模式是 "fast"，不受此限制影響）。
    mode = "external"：方案 A 外部工具介接——呼叫使用者指定的外部降噪 CLI
        （例如 NoiseXTerminator / DeepSNR CLI）處理，失敗時直接跳過本步驟並
        回傳原圖（未降噪），不中斷 pipeline，也不會靜默改用 fast/quality 頂替。
        外部工具本來就走 16-bit TIFF 暫存檔（見 run_external_image_tool()），
        全程精度完整，不受這裡的修改影響。
    """
    if not enable:
        return img01
    if mode == "external":
        if "--denoise_strength" in (ext_args or ""):
            out01, err = run_cosmic_clarity_denoise(img01, ext_path, ext_args)
        else:
            out01, err = run_external_image_tool(img01, ext_path, ext_args)
        if err is not None:
            print(f"[外部降噪 external denoise] 已跳過，改用原圖（未降噪）: {err}")
            return img01
        return np.clip(out01, 0, 1).astype(np.float32)
    if mode == "quality":
        if HAS_SKIMAGE:
            # v1.3.1：全程 float32，不再有內部 8-bit 量化（詳見上方 docstring）。
            img01c = np.clip(img01, 0, 1).astype(np.float32)
            lab = rgb2lab(img01c)  # L: 0-100, a/b: 約 -128..127
            h_l = float(nlm_h) * (100.0 / 255.0)
            h_ab = float(nlm_h_color)
            patch_size, patch_distance = 7, 10
            nlm_kwargs = dict(patch_size=patch_size, patch_distance=patch_distance,
                               fast_mode=True, preserve_range=True)
            L_out = denoise_nl_means(lab[..., 0], h=h_l, **nlm_kwargs)
            a_out = denoise_nl_means(lab[..., 1], h=h_ab, **nlm_kwargs)
            b_out = denoise_nl_means(lab[..., 2], h=h_ab, **nlm_kwargs)
            lab_out = np.stack([L_out, a_out, b_out], axis=-1).astype(np.float32)
            out01 = lab2rgb(lab_out).astype(np.float32)
            return np.clip(out01, 0, 1).astype(np.float32)
        else:
            # skimage 未安裝：退回 v1.3.0 之前的行為（內部量化成 8-bit）。
            print("[降噪 quality 模式] 未安裝 scikit-image，退回 8-bit 內部量化的舊行為 "
                  "（pip install scikit-image 可解除此限制）。")
            img8 = (np.clip(img01, 0, 1) * 255.0).astype(np.uint8)
            out8 = cv2.fastNlMeansDenoisingColored(
                img8, None,
                h=float(nlm_h), hColor=float(nlm_h_color),
                templateWindowSize=7, searchWindowSize=21,
            )
            return out8.astype(np.float32) / 255.0
    sigma_color_01 = float(sigma_color) / 255.0
    out01 = cv2.bilateralFilter(
        img01.astype(np.float32), d=int(round(d)),
        sigmaColor=sigma_color_01, sigmaSpace=sigma_space,
    )
    return np.clip(out01, 0, 1).astype(np.float32)


def detect_star_mask(img01, kernel_size, thresh, max_area, max_area_large, aspect_thresh, dilate_base, dilate_scale):
    kernel_size = max(1, int(round(kernel_size)))
    if kernel_size % 2 == 0: kernel_size += 1
    
    gray = cv2.cvtColor((img01 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    tophat = cv2.subtract(gray, opened)
    _, raw_mask = cv2.threshold(tophat, thresh, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]
    widths = stats[:, cv2.CC_STAT_WIDTH]
    heights = stats[:, cv2.CC_STAT_HEIGHT]
    long_side = np.maximum(widths, heights)
    short_side = np.maximum(np.minimum(widths, heights), 1)
    aspect = long_side / short_side

    small_star = areas <= max_area
    bright_halo = (areas > max_area) & (areas <= max_area_large) & (aspect <= aspect_thresh)
    valid = small_star | bright_halo
    valid[0] = False

    mask = (valid[labels].astype(np.uint8)) * 255

    if dilate_base > 0 or dilate_scale > 0:
        dilate_amounts = np.zeros(num_labels, dtype=np.int32)
        dilate_amounts[valid] = np.maximum(
            1, np.round(dilate_base + long_side[valid] * dilate_scale).astype(np.int32)
        )
        mask_out = np.zeros_like(mask)
        for amt in np.unique(dilate_amounts[valid]):
            bucket_labels = valid & (dilate_amounts == amt)
            sub_mask = (bucket_labels[labels].astype(np.uint8)) * 255
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (amt * 2 + 1, amt * 2 + 1))
            sub_mask = cv2.dilate(sub_mask, dilate_kernel)
            mask_out = cv2.bitwise_or(mask_out, sub_mask)
        mask = mask_out

    return mask


def detect_cluster_mask(img01, kernel_large, thresh_large, min_area, max_area, aspect_thresh, cluster_dilate, exclude_mask=None):
    kernel_large = max(1, int(round(kernel_large)))
    if kernel_large % 2 == 0: kernel_large += 1
    
    gray = cv2.cvtColor((img01 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_large, kernel_large))
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    tophat = cv2.subtract(gray, opened)
    _, raw_mask = cv2.threshold(tophat, thresh_large, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]
    widths = stats[:, cv2.CC_STAT_WIDTH]
    heights = stats[:, cv2.CC_STAT_HEIGHT]
    long_side = np.maximum(widths, heights)
    short_side = np.maximum(np.minimum(widths, heights), 1)
    aspect = long_side / short_side

    valid = (areas >= min_area) & (areas <= max_area) & (aspect <= aspect_thresh)
    valid[0] = False

    mask = (valid[labels].astype(np.uint8)) * 255

    cluster_dilate = int(round(cluster_dilate))
    if cluster_dilate > 0:
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cluster_dilate * 2 + 1, cluster_dilate * 2 + 1))
        mask = cv2.dilate(mask, dilate_kernel)

    if exclude_mask is not None:
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(exclude_mask))

    return mask


def shrink_stars(img01, mask, kernel_size, iterations, strength):
    kernel_size = max(1, int(round(kernel_size)))
    iterations = max(1, int(round(iterations)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = img01.copy()
    for c in range(3):
        eroded[:, :, c] = cv2.erode(img01[:, :, c], kernel, iterations=iterations)

    mask_f = (mask.astype(np.float32) / 255.0)[:, :, None] * strength
    out = img01 * (1 - mask_f) + eroded * mask_f
    return np.clip(out, 0, 1)


def estimate_local_noise(img01, mask):
    """估計原圖『非遮罩區域』的雜訊強度(標準差)，用來合成回填雜訊，
    避免修補區因為 inpaint 內插而顯得過度平滑乾淨、跟周圍有雜訊顆粒感的星空格格不入。

    效能考量：雜訊統計特性對解析度不敏感，這裡先縮小到長邊約 700px 再估計，
    避免在全解析度大圖(例如 6000px 以上)上重複做高斯模糊拖慢速度。"""
    gray = (np.clip(img01, 0, 1) * 255).astype(np.float32).mean(axis=2)
    h, w = gray.shape
    scale = min(1.0, 700.0 / max(h, w))
    if scale < 1.0:
        small_gray = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(mask, (small_gray.shape[1], small_gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        small_gray, small_mask = gray, mask

    blur = cv2.GaussianBlur(small_gray, (0, 0), sigmaX=3)
    residual = small_gray - blur
    valid = small_mask == 0
    # 防護：遮罩覆蓋率過高（如密集星團/大範圍曝光）時，若無足夠代表性樣本，回傳預設值 2.0
    if np.sum(valid) < 100:
        return 2.0
    sigma = float(np.std(residual[valid]))
    return float(np.clip(sigma, 0.5, 12.0))


# ============================================================
# ===================== 雜訊預生成優化 ========================
# ============================================================
_STATIC_NOISE_SIZE = 4096
# 預先生成一個標準差為 1.0 的高斯模糊雜訊基底
_STATIC_NOISE = np.random.normal(0, 1.0, size=(_STATIC_NOISE_SIZE, _STATIC_NOISE_SIZE)).astype(np.float32)
_STATIC_NOISE = cv2.GaussianBlur(_STATIC_NOISE, (0, 0), sigmaX=0.6)
# 由於高斯模糊會降低標準差，我們將其重新歸一化至標準差 = 1.0，確保後續乘上 sigma 的精確性
_std = np.std(_STATIC_NOISE)
if _std > 1e-5:
    _STATIC_NOISE /= _std


def synthesize_noise_like(shape_hw, sigma):
    """從預先生成的雜訊紋理中裁切並縮放，避免在高解析度下重複進行 np.random.normal 與 GaussianBlur 帶來的巨大 CPU 開銷。
    優化：當影像尺寸大於 4096 時使用 cv2.resize 平滑拉伸，消除 np.tile 的拼貼邊界斷層。
    """
    h, w = shape_hw
    if h <= _STATIC_NOISE_SIZE and w <= _STATIC_NOISE_SIZE:
        # 隨機選取起始點以增加隨機性
        sy = np.random.randint(0, _STATIC_NOISE_SIZE - h + 1)
        sx = np.random.randint(0, _STATIC_NOISE_SIZE - w + 1)
        noise = _STATIC_NOISE[sy:sy+h, sx:sx+w] * sigma
    else:
        # 尺寸大於 4096 時，採用雙三次插值拉伸（不會產生拼貼縫隙，且輕微拉伸不會破壞星空的顆粒感）
        noise = cv2.resize(_STATIC_NOISE, (w, h), interpolation=cv2.INTER_CUBIC) * sigma
    return noise


def feather_alpha(mask, feather_px):
    """把二值遮罩模糊成 0~1 的漸層 alpha，讓修補區與原圖邊界不再是一刀切的硬邊。"""
    feather_px = max(0.0, float(feather_px))
    if feather_px <= 0.05:
        return (mask.astype(np.float32) / 255.0)
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=feather_px)
    return np.clip(alpha / 255.0, 0, 1)


def remove_stars_adaptive(img01, mask, base_radius, size_factor=0.15, max_radius=None, max_buckets=6):
    """依遮罩內每一個連通分量(每顆星/每團星團)的實際外接框大小，動態調整 inpaint 取樣半徑。
    範圍越大的星點/暈光，取樣半徑也跟著加大，避免統一用小半徑內插大面積區域造成失真、
    看起來像貼一塊平坦色塊。

    效能考量：為避免星點數量一多、半徑連續變化導致要對整張圖跑上百次 inpaint(每次都是
    全圖運算)而拖慢速度，這裡把半徑「量化」收斂成最多 max_buckets 個分桶，
    同一桶內的星點合併成一個遮罩、只呼叫一次 inpaint，兼顧效果與速度。"""
    if mask is None or not np.any(mask):
        return img01.copy()

    base_radius = max(1, int(round(base_radius)))
    if max_radius is None:
        max_radius = base_radius * 4
    max_radius = max(base_radius, int(round(max_radius)))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    img8 = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
    out = img8.copy()

    radius_span = max(1, max_radius - base_radius)
    bucket_step = max(1, int(round(radius_span / max_buckets)))

    # ④ 建立查表陣列：label_to_radius[i] = 第 i 個 label 對應的分桶半徑（0 = 背景，不處理）
    label_to_radius = np.zeros(num_labels, dtype=np.int32)
    for lbl in range(1, num_labels):
        long_side = max(stats[lbl, cv2.CC_STAT_WIDTH], stats[lbl, cv2.CC_STAT_HEIGHT])
        raw_radius = base_radius + long_side * size_factor
        quantized = base_radius + round((raw_radius - base_radius) / bucket_step) * bucket_step
        label_to_radius[lbl] = int(np.clip(quantized, base_radius, max_radius))

    # 一次向量化查表，得到每個 pixel 的目標半徑
    pixel_radius = label_to_radius[labels]   # shape = (H, W)

    for radius in np.unique(label_to_radius[1:]):   # 跳過 label 0（背景）
        bucket_mask = (pixel_radius == radius).astype(np.uint8) * 255
        out = cv2.inpaint(out, bucket_mask, radius, cv2.INPAINT_TELEA)

    return out.astype(np.float32) / 255.0


def remove_stars_natural(img01, mask, base_radius, feather_px=2.0, noise_strength=1.0,
                          size_factor=0.15, max_radius=None):
    """比起單純呼叫 cv2.inpaint 硬蓋上去，這裡多做三件事讓去星結果更自然：
      1. 取樣半徑依每顆星/星團實際大小自動調整(remove_stars_adaptive)
      2. 用羽化後的遮罩做邊緣柔和混合，而非二值硬邊覆蓋
      3. 依周圍背景雜訊強度，在修補區合成相近的顆粒雜訊，避免修補區「太乾淨」而顯得突兀

    效能考量：雜訊只在遮罩實際涵蓋的範圍(bounding box + 羽化外擴的緩衝)內合成，
    而不是對整張圖都生成一份隨機雜訊再模糊一次，這在大圖、遮罩只佔一小部分時能省下不少時間。
    """
    if mask is None or not np.any(mask):
        return img01.copy()

    inpainted01 = remove_stars_adaptive(img01, mask, base_radius, size_factor, max_radius)

    if noise_strength > 0:
        sigma = estimate_local_noise(img01, mask) * noise_strength
        ys, xs = np.where(mask > 0)
        pad = int(round(feather_px * 3)) + 4
        y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad + 1)
        x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad + 1)
        noise_crop = synthesize_noise_like((y1 - y0, x1 - x0), sigma) / 255.0
        inpainted01[y0:y1, x0:x1] = np.clip(inpainted01[y0:y1, x0:x1] + noise_crop[:, :, None], 0, 1)

    alpha = feather_alpha(mask, feather_px)[:, :, None]
    out = img01 * (1 - alpha) + inpainted01 * alpha
    return np.clip(out, 0, 1)


def remove_stars_multiscale(img01, star_mask, cluster_mask, radius_small, radius_large,
                             feather_px=2.0, noise_strength=1.0):
    out = img01.copy()
    if star_mask is not None and np.any(star_mask):
        out = remove_stars_natural(
            out, star_mask, base_radius=radius_small,
            feather_px=feather_px, noise_strength=noise_strength,
            size_factor=0.15, max_radius=radius_small * 4,
        )
    if cluster_mask is not None and np.any(cluster_mask):
        out = remove_stars_natural(
            out, cluster_mask, base_radius=radius_large,
            feather_px=feather_px, noise_strength=noise_strength,
            size_factor=0.08, max_radius=radius_large * 3,
        )
    return out


# 7️⃣ / 7️⃣b 分頁中，屬於「像素單位」的星點參數：
#   - 長度類(核大小/外擴像素/取樣半徑/羽化像素)：預覽時依 scale 等比例縮小
#   - 面積類(最大面積等)：屬於像素平方單位，需依 scale^2 縮小
#   其餘如門檻、長寬比、混合強度、次數、雜訊回填強度(倍率)等為「無因次」參數，不隨解析度縮放
_STAR_LINEAR_KEYS = [
    'star_kernel', 'star_dilate', 'star_shrink_kernel', 'star_inpaint_radius', 'star_feather_px',
    'cluster_kernel', 'cluster_dilate', 'cluster_inpaint_radius',
]
_STAR_AREA_KEYS = [
    'star_max_area', 'star_max_area_large', 'cluster_min_area', 'cluster_max_area',
]


def get_effective_star_params(p, scale):
    """依目前處理影像相對於原圖的縮放比例(scale)，換算出星點縮小/去星相關的
    「等效參數」。scale=1.0(全解析度匯出)時直接沿用使用者設定的原始數值；
    scale<1.0(預覽縮圖)時，長度類參數乘上 scale、面積類參數乘上 scale^2，
    這樣縮圖與全解析度圖上偵測到的星點大小/範圍才會一致，而不是直接套用
    針對原圖尺寸調校的數值到縮圖上。"""
    if scale is None or scale >= 0.999:
        return p
    q = dict(p)
    for k in _STAR_LINEAR_KEYS:
        if k in q:
            q[k] = max(1, q[k] * scale)
    for k in _STAR_AREA_KEYS:
        if k in q:
            q[k] = max(1, q[k] * (scale ** 2))
    return q


def process_stars(mode, img01, star_mask, cluster_mask, p):
    if mode == "shrink":
        return shrink_stars(img01, star_mask, p['star_shrink_kernel'], p['star_shrink_iter'], p['star_shrink_strength'])
    elif mode == "remove":
        return remove_stars_multiscale(
            img01, star_mask, cluster_mask, p['star_inpaint_radius'], p['cluster_inpaint_radius'],
            feather_px=p.get('star_feather_px', 2.0), noise_strength=p.get('star_noise_strength', 1.0),
        )
    elif mode == "external":
        # 方案 A 外部工具介接——呼叫使用者指定的外部去星 CLI（例如 StarXTerminator/StarNet CLI）。
        # 失敗時直接跳過本步驟並回傳原圖（未去星），不中斷 pipeline。
        out01, err = run_external_image_tool(img01, p.get('star_ext_path', ''), p.get('star_ext_args', ''))
        if err is not None:
            print(f"[外部去星 external star removal] 已跳過，改用原圖（未去星）: {err}")
            return img01
        return out01
    elif mode == "none":
        return img01
    return img01


def finish_pipeline(img01, p):
    """天文影像後處理管線，依科學正確的順序執行：

    1. 雙邊降噪（Denoise）
       在低增益、未銳化的線性空間先降噪，保護微弱星雲邊界不被後續銳化放大的雜訊所破壞。

    2. Clarity 局部對比 + 銳化（Clarity / Sharpen）
       在乾淨的降噪後影像上提升局部對比與邊緣清晰度，效果更精準，
       避免「先銳化放大雜訊、再降噪一起抹掉細節」的反效果。

    3. 飽和度 / 明度 / 通道增益（Saturation / Boost / Gain）
       最後才在已降噪且銳化完的影像上調整色彩，
       防止先飽和再做高斯 Clarity 在亮星周圍產生彩虹色邊（色相過飽和 + 模糊擴散）。

    v1.3.0 位元深度修正：這三步以前每一步之間都會先量化成 uint8 [0,255]
    再交給下一步（denoise 前、clarity/sharpen 中間、以及函式最終回傳都各自
    quantize 一次），疊加三次獨立的捨入誤差。實際受影響最明顯的是背景漸層
    扣除、動態範圍拉伸這類已經對原始暗部/弱訊號做過大幅拉伸的天文影像，
    拉伸後的色階斷層（banding）在這種資料上比一般攝影更容易被看到。

    現在整個函式全程維持 float32 [0,1]，只有 denoise() 內部的 "quality"
    模式（NLM 降噪）仍會有一次無法避免的 8-bit 量化，因為那是 OpenCV
    fastNlMeansDenoisingColored 本身的限制，不是這裡的問題（細節見
    denoise() 的 docstring）。函式回傳的是 float32 [0,1]，不再是 uint8——
    呼叫端（畫面預覽 / 全解析度匯出）依各自需求，在需要顯示或寫出 8-bit
    格式（如 JPEG、螢幕預覽）時才轉成 uint8；寫 16-bit TIFF 時則直接用
    這裡的 float32 結果換算，不會先損失精度再撐大格式。
    """
    # Step 1: 先降噪——保護微弱星雲邊界
    img_work = np.clip(img01, 0, 1).astype(np.float32)  # float32 [0, 1]
    if p['denoise_enable']:
        img_work = denoise(
            img_work, enable=True,
            mode=p.get('denoise_mode', 'fast'),
            d=p['denoise_d'],
            sigma_color=p['denoise_sigma_color'],
            sigma_space=p['denoise_sigma_space'],
            nlm_h=p.get('denoise_nlm_h', 10),
            nlm_h_color=p.get('denoise_nlm_h_color', 10),
            ext_path=p.get('denoise_ext_path', ''),
            ext_args=p.get('denoise_ext_args', ''),
        )

    # Step 2: Clarity + 銳化——在乾淨影像上提升局部對比
    # v1.3.3：Clarity 永遠內建；銳化子步驟依 sharpen_mode 二選一（internal/external）
    img_work = apply_clarity_and_sharpen(
        img_work,
        p['clarity_blur'], p['clarity_strength'],
        p['sharpen_blur'], p['sharpen_amount'],
        sharpen_mode=p.get('sharpen_mode', 'internal'),
        sharpen_ext_path=p.get('sharpen_ext_path', ''),
        sharpen_ext_args=p.get('sharpen_ext_args', ''),
    )

    # Step 3: 飽和度 / 明度 / 通道增益——最後調色，避免色偏擴散
    img01_out = boost_saturation(
        img_work,
        p['sat_boost'], p['bright_boost'],
        p['r_gain'], p['g_gain'], p['b_gain']
    )
    return img01_out  # float32 [0, 1]


def build_target_mask_overlay(img_uint8, mask, contour_thresh=0.35):
    """把 detect_target_regions() 算出的目標遮罩，疊成半透明色塊 + 邊界輪廓線顯示在成品圖上，
    方便使用者確認「自動局部拉伸」實際加強了哪些區域(可能同時有好幾片不相連的區域)。
    遮罩值越高，該處疊色越明顯；平坦天空背景(遮罩≈0)幾乎維持原圖不變。

    contour_thresh：把遮罩用這個門檻二值化後，用 cv2.findContours 畫出「演算法認定的邊界」——
    findContours 本來就會回傳所有互不相連的輪廓，所以有好幾片星雲/銀河結構時，
    每一片都會各自畫出一條邊界線，不會只框出其中一塊。
    """
    if mask is None or img_uint8 is None:
        return None
    h, w = img_uint8.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    overlay_color = np.array([255, 70, 70], dtype=np.float32)  # 半透明紅色，標示「有加強」的區域
    alpha = np.clip(mask, 0.0, 1.0)[:, :, None] * 0.5  # 最高疊色不透明度限制在 50%，避免整張變色蓋掉原圖
    base = img_uint8.astype(np.float32)
    out = np.clip(base * (1 - alpha) + overlay_color[None, None, :] * alpha, 0, 255).astype(np.uint8)

    # 邊界輪廓線：用門檻二值化後找輪廓，畫成亮黃色線條，讓「演算法看到的邊界」清楚可辨
    mask_u8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    _, binary = cv2.threshold(mask_u8, int(contour_thresh * 255), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    line_thickness = max(1, round(min(h, w) / 300))
    cv2.drawContours(out, contours, -1, (255, 255, 0), thickness=line_thickness)  # 亮黃色邊界線
    return out


def run_pipeline(img01, p, want_layers=False, preview_scale=1.0):
    """
    執行主流程：修正 want_layers=True 時，獨立生成真正 100% 執行 Inpaint 的去星背景層

    preview_scale: 目前傳入的 img01 相對於「原始全解析度圖」的縮放比例。
        - 全解析度匯出時固定為 1.0，星點參數直接使用使用者設定的原始數值。
        - 預覽縮圖時傳入實際縮放比例(<1.0)，星點縮小/去星相關的像素單位參數
          會先等比例換算，避免在縮圖上直接套用針對原圖尺寸調校的數值。
    """
    img = img01
    if p['bg_enable']:
        # 修正：bg_min_filter / bg_blur_sigma 為像素單位，作用在 bg_downscale 後的小圖上。
        # 縮圖預覽時，小圖遠比全解析度時更小，導致相同數值的 kernel 相對尺寸暴增
        # （例如 min_filter=9 在 48px 小圖上覆蓋 18.75%，在 240px 全圖小圖上只覆蓋 3.75%）
        # 解法：預覽時依 preview_scale 等比縮小這兩個 pixel-unit 參數，
        # 讓背景估計的「實際覆蓋範圍」與全解析度時保持一致。
        bg_min_filter = p['bg_min_filter']
        bg_blur_sigma = p['bg_blur_sigma']
        if preview_scale is not None and preview_scale < 0.99:
            bg_min_filter = max(3, int(round(bg_min_filter * preview_scale)))
            bg_blur_sigma  = max(0.5, bg_blur_sigma * preview_scale)
        img = remove_background_gradient(img, p['bg_downscale'], bg_min_filter, bg_blur_sigma, p['bg_subtract'])


    img = correct_color_cast(img, p['wb_enable'], p['wb_min'], p['wb_max'])
    img = stretch_dynamic_range(img, p['black_pct'], p['stretch_factor'], p['white_pct'])
    img, target_mask = apply_local_target_boost(
        img,
        p.get('local_target_enable', False),
        p.get('local_target_strength', 0.0),
        p.get('local_target_radius', 40),
        p.get('local_target_sensitivity', 1.0),
        p.get('manual_target_enable', False),
        p.get('manual_target_x_pct', 50.0),
        p.get('manual_target_y_pct', 50.0),
        p.get('manual_target_w_pct', 25.0),
        p.get('manual_target_h_pct', 25.0),
        p.get('manual_target_weight', 1.0),
        p.get('manual_target_feather_pct', 20.0),
        p.get('manual_target_shape', 'rectangle'),
    )
    pre_star_img = img

    # 星點縮小/去星相關的「等效參數」：預覽縮圖時已依 preview_scale 等比例縮小
    p_star = get_effective_star_params(p, preview_scale)

    need_mask = p['star_mode'] != 'none' or want_layers
    star_mask = None
    cluster_mask = None
    
    if need_mask:
        star_mask = detect_star_mask(
            pre_star_img, p_star['star_kernel'], p['star_thresh'], p_star['star_max_area'],
            p_star['star_max_area_large'], p['star_aspect'], p_star['star_dilate'], p['star_dilate_scale']
        )
        if p['multiscale_enable']:
            cluster_mask = detect_cluster_mask(
                pre_star_img, p_star['cluster_kernel'], p['cluster_thresh'], p_star['cluster_min_area'],
                p_star['cluster_max_area'], p['cluster_aspect'], p_star['cluster_dilate'], exclude_mask=star_mask
            )

    # 1. 產生主畫面需要的成品（可能是縮星、無星或無處理）
    main_processed_img = process_stars(p['star_mode'], pre_star_img, star_mask, cluster_mask, p_star)
    # v1.3.0：main_out 現在是 float32 [0,1]（之前是 uint8），保留完整精度，
    # 供全解析度匯出時寫出真正的 16-bit TIFF。畫面顯示用的 uint8 版本只在
    # 真正需要顯示的地方（例如下面的 target_mask_overlay）才轉換，呼叫端
    # （預覽/匯出/批次函式）各自決定何時轉 uint8。
    main_out = finish_pipeline(main_processed_img, p)

    # 2. 如果使用者需要額外圖層，我們獨立且強制跑一次完整的「去星(Remove)」流程
    starless_out = main_out
    if want_layers:
        # 強制進行多尺度去星填補（即使主畫面選的是 shrink 或 none）
        pure_starless_img = remove_stars_multiscale(
            pre_star_img, star_mask, cluster_mask, 
            radius_small=p_star['star_inpaint_radius'], radius_large=p_star['cluster_inpaint_radius'],
            feather_px=p_star.get('star_feather_px', 2.0), noise_strength=p.get('star_noise_strength', 1.0),
        )
        # 去星後，同樣走完最後的色彩與降噪流程，確保曝光色調跟主圖完全一致
        starless_out = finish_pipeline(pure_starless_img, p)

    # build_target_mask_overlay() 是純畫面顯示用途（半透明疊色 + 邊界線），
    # 只需要 uint8，這裡才轉換，不影響 result['main'] 保留的 float32 精度。
    main_out_uint8_for_overlay = (np.clip(main_out, 0, 1) * 255).astype(np.uint8)

    result = {
        'main': main_out,
        'mask': star_mask,
        'starless': starless_out,
        'target_mask_overlay': build_target_mask_overlay(main_out_uint8_for_overlay, target_mask),
    }
    return result


def save_image_files(img01, out_dir, name, formats=("jpg", "tif")):
    """輸出 JPEG（8-bit，供快速預覽/分享）與/或 TIFF（真 16-bit，供後續編修）。

    img01: float32 [0,1]，來自 finish_pipeline() 的全精度結果。
    formats: 要寫出的格式集合，可包含 "jpg"、"tif"（大小寫不拘）。預設沿用
        舊行為，兩種都寫。v1.3.2 新增：批次/單張匯出時如果只需要其中一種格式，
        可以省下另一種格式的匯出時間與硬碟用量（`export_fn()`／
        `batch_process_fn()` 呼叫端可透過 UI 勾選框控制）。

    回傳 (out_jpg, out_tif)：對應格式沒有被選取時該欄位為 None，呼叫端需自行
    判斷是否為 None 再使用（例如組訊息、組下載連結時跳過）。

    v1.3.0 修正：以前這裡收到的是已經被 finish_pipeline() 沿路量化過的
    uint8 結果，「16-bit TIFF」只是把 8-bit 數值撐大（`img8.astype(np.uint16)
    * 257`），容器是 16-bit，內容其實還是 8-bit。現在改成直接吃全精度的
    float32，用 round() 換算成真正的 16-bit 數值，TIFF 檔案裡的色階數量
    才會名副其實。改用 tifffile 寫 TIFF（而不是 cv2.imwrite）也跟本檔案
    其他地方寫 16-bit TIFF 的方式一致（見 run_external_image_tool()／
    run_cosmic_clarity_denoise()）——tifffile 直接吃 RGB 順序，不像 cv2
    需要先轉 BGR。
    """
    formats_norm = {str(f).strip().lower() for f in (formats or ())}
    if not formats_norm:
        # 保底：一種格式都沒選到的話，維持舊版「兩種都寫」的行為，
        # 避免因為呼叫端傳空集合而完全沒有輸出檔案。
        formats_norm = {"jpg", "tif"}
    want_jpg = bool(formats_norm & {"jpg", "jpeg"})
    want_tif = bool(formats_norm & {"tif", "tiff"})

    img01_clipped = np.clip(img01, 0, 1).astype(np.float32)

    out_jpg = None
    if want_jpg:
        out_jpg = os.path.join(out_dir, f"{name}.jpg")
        img8 = (img01_clipped * 255.0 + 0.5).astype(np.uint8)
        cv2.imwrite(out_jpg, cv2.cvtColor(img8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 96])

    out_tif = None
    if want_tif:
        out_tif = os.path.join(out_dir, f"{name}.tif")
        img16 = (img01_clipped * 65535.0 + 0.5).astype(np.uint16)
        tifffile.imwrite(out_tif, img16)

    return out_jpg, out_tif


def generate_histogram(img8):
    """① 純 NumPy/cv2 快速 Histogram 渲染。
    不使用 matplotlib，避免每次 figure 建立/繪製/銷毀的巨大開銷（50–200ms），
    改為直接在 numpy 陣列上繪製，速度降至 1–5ms。
    """
    if img8 is None:
        return None

    # ── 畫布尺寸 ─────────────────────────────────────────────
    W, H = 512, 200
    canvas = np.full((H, W, 3), (17, 17, 17), dtype=np.uint8)  # 深灰背景

    # ── 背景格線 ─────────────────────────────────────────────
    for x in range(0, W, W // 4):
        canvas[:, x] = (40, 40, 40)
    for y in [H // 4, H // 2, 3 * H // 4]:
        canvas[y, :] = (40, 40, 40)

    # ── 計算三通道直方圖並繪製 ───────────────────────────────
    _COLORS = [(70, 70, 220), (70, 190, 70), (220, 70, 70)]   # BGR: R、G、B 線條
    _ALPHA  = [0.85, 0.85, 0.85]
    for ch, (bgr_color, alpha) in enumerate(zip(_COLORS, _ALPHA)):
        hist = cv2.calcHist([img8], [ch], None, [256], [0, 256]).flatten()
        # 對數壓縮，讓低值區域也可見
        hist = np.log1p(hist)
        hist_max = hist.max()
        if hist_max < 1e-6:
            continue
        hist_norm = hist / hist_max  # [0, 1]

        # 把 256 個值插值到 W 個 x 位置
        xs = np.linspace(0, 255, W).astype(np.float32)
        ys_f = np.interp(xs, np.arange(256), hist_norm)
        # y 座標（翻轉：0 在底部）
        ys = np.clip((1.0 - ys_f) * (H - 4), 0, H - 1).astype(np.int32)

        # 填充曲線下方區域（半透明）
        fill_color = tuple(int(c * 0.25) for c in bgr_color)
        for xi in range(W):
            y_top = ys[xi]
            if y_top < H - 1:
                canvas[y_top:H - 1, xi] = np.clip(
                    canvas[y_top:H - 1, xi].astype(np.int32) + fill_color, 0, 255
                ).astype(np.uint8)

        # 繪製曲線本身（折線連接各點）
        pts = np.stack([np.arange(W), ys], axis=1).astype(np.int32)
        for xi in range(W - 1):
            cv2.line(canvas, (pts[xi, 0], pts[xi, 1]), (pts[xi+1, 0], pts[xi+1, 1]),
                     bgr_color, 1, cv2.LINE_AA)

    # ── x 軸刻度（0, 64, 128, 192, 255） ────────────────────
    for val in [0, 64, 128, 192, 255]:
        x = int(val / 255 * (W - 1))
        cv2.putText(canvas, str(val), (max(0, x - 8), H - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (120, 120, 120), 1, cv2.LINE_AA)

    # cv2 輸出是 BGR，轉回 RGB 給 Gradio Image
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)




# ============================================================
# ===================== 滑桿疊圖對比 (Lightroom 風格) =====================
# ============================================================

def img_to_data_uri(img_uint8, quality=90):
    """把 RGB uint8 numpy 陣列編碼成 JPEG data URI，供 HTML <img> 直接顯示。"""
    if img_uint8 is None:
        return ""
    bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    b64 = base64.b64encode(buf).decode('ascii')
    return f"data:image/jpeg;base64,{b64}"


def build_compare_slider_html(before_uint8, after_uint8, lang="zh", height=450):
    """產生「同一張圖左右滑動切換前後」的疊圖比較 HTML(Lightroom 風格)，
    而非左右並排兩張圖。使用 clip-path 讓上層(處理前)的圖只顯示滑桿左側的部分，
    下層(處理後)的圖維持完整顯示，拖曳中央把手即可即時比較。"""
    if before_uint8 is None or after_uint8 is None:
        msg = "⚠️ 尚未載入圖片" if lang == "zh" else "⚠️ Image not loaded"
        return f"<div style='color:#888;padding:24px;text-align:center;'>{msg}</div>"

    before_uri = img_to_data_uri(before_uint8)
    after_uri = img_to_data_uri(after_uint8)
    before_label = LANG[lang]["before_label"]
    after_label = LANG[lang]["after_label"]

    return f"""
<div style="position:relative;width:100%;height:{height}px;background:#111;border-radius:10px;
            overflow:hidden;cursor:ew-resize;user-select:none;touch-action:none;"
     onmousedown="this.dragging=true;var r=this.getBoundingClientRect();var x=Math.max(0,Math.min(100,(event.clientX-r.left)/r.width*100));this.querySelector('.ba-before').style.clipPath='inset(0 '+(100-x)+'% 0 0)';this.querySelector('.ba-handle').style.left=x+'%';"
     onmousemove="if(this.dragging){{var r=this.getBoundingClientRect();var x=Math.max(0,Math.min(100,(event.clientX-r.left)/r.width*100));this.querySelector('.ba-before').style.clipPath='inset(0 '+(100-x)+'% 0 0)';this.querySelector('.ba-handle').style.left=x+'%';}}"
     onmouseup="this.dragging=false;"
     onmouseleave="this.dragging=false;"
     ontouchstart="this.dragging=true;var r=this.getBoundingClientRect();var t=event.touches[0];var x=Math.max(0,Math.min(100,(t.clientX-r.left)/r.width*100));this.querySelector('.ba-before').style.clipPath='inset(0 '+(100-x)+'% 0 0)';this.querySelector('.ba-handle').style.left=x+'%';"
     ontouchmove="var r=this.getBoundingClientRect();var t=event.touches[0];var x=Math.max(0,Math.min(100,(t.clientX-r.left)/r.width*100));this.querySelector('.ba-before').style.clipPath='inset(0 '+(100-x)+'% 0 0)';this.querySelector('.ba-handle').style.left=x+'%';"
     ontouchend="this.dragging=false;">

  <img src="{after_uri}" draggable="false"
       style="position:absolute;inset:0;width:100%;height:100%;object-fit:contain;pointer-events:none;" />

  <img class="ba-before" src="{before_uri}" draggable="false"
       style="position:absolute;inset:0;width:100%;height:100%;object-fit:contain;pointer-events:none;
              clip-path:inset(0 50% 0 0);" />

  <div style="position:absolute;top:10px;left:10px;background:rgba(0,0,0,.55);color:#fff;
              font-size:12px;padding:3px 9px;border-radius:5px;pointer-events:none;">{before_label}</div>
  <div style="position:absolute;top:10px;right:10px;background:rgba(0,0,0,.55);color:#fff;
              font-size:12px;padding:3px 9px;border-radius:5px;pointer-events:none;">{after_label}</div>

  <div class="ba-handle" style="position:absolute;top:0;left:50%;width:0;height:100%;
              border-left:2px solid rgba(255,255,255,.9);pointer-events:none;
              box-shadow:0 0 6px rgba(0,0,0,.6);">
    <div style="position:absolute;top:50%;left:0;transform:translate(-50%,-50%);width:34px;height:34px;
                border-radius:50%;background:#fff;box-shadow:0 2px 8px rgba(0,0,0,.5);
                display:flex;align-items:center;justify-content:center;font-size:15px;color:#333;">⇔</div>
  </div>
</div>
"""


# ============================================================
# ========================= 參數對照表 =========================
# ============================================================

PARAM_NAMES = [
    'bg_enable', 'bg_downscale', 'bg_min_filter', 'bg_blur_sigma', 'bg_subtract',
    'wb_enable', 'wb_min', 'wb_max',
    'black_pct', 'stretch_factor', 'white_pct',
    'local_target_enable', 'local_target_strength', 'local_target_radius', 'local_target_sensitivity',
    'sat_boost', 'bright_boost', 'r_gain', 'g_gain', 'b_gain',
    'clarity_blur', 'clarity_strength', 'sharpen_blur', 'sharpen_amount',
    'denoise_enable', 'denoise_mode', 'denoise_d', 'denoise_sigma_color', 'denoise_sigma_space',
    'denoise_nlm_h', 'denoise_nlm_h_color',
    'star_mode', 'star_kernel', 'star_thresh', 'star_max_area', 'star_max_area_large', 'star_aspect',
    'star_dilate', 'star_dilate_scale', 'star_shrink_kernel', 'star_shrink_iter', 'star_shrink_strength', 'star_inpaint_radius',
    'multiscale_enable', 'cluster_kernel', 'cluster_thresh', 'cluster_min_area', 'cluster_max_area',
    'cluster_aspect', 'cluster_dilate', 'cluster_inpaint_radius',
    'star_feather_px', 'star_noise_strength',
    'manual_target_enable', 'manual_target_x_pct', 'manual_target_y_pct',
    'manual_target_w_pct', 'manual_target_h_pct', 'manual_target_weight', 'manual_target_feather_pct',
    'manual_target_shape',
    # 方案 A：ML 降噪 / ML 去星 外部工具介接（附加在最後，避免動到既有參數的索引位置，
    # 例如 DEFAULTS[24]/DEFAULTS[43] 這類直接以索引取值的地方）
    'denoise_ext_path', 'denoise_ext_args',
    'star_ext_path', 'star_ext_args',
    # v1.3.3：Track A——Cosmic Clarity Sharpen 外部介接（同樣附加在最後，
    # 理由同上：避免動到既有參數的索引位置）
    'sharpen_mode', 'sharpen_ext_path', 'sharpen_ext_args',
]

DEFAULTS = [
    True, 0.06, 9, 6, 0.92,
    True, 0.6, 1.8,
    0.2, 12.0, 99.7,
    False, 0.6, 40, 1.0,
    1.45, 1.03, 1.0, 1.0, 1.0,
    25, 0.35, 2, 1.25,
    True, "fast", 5, 15, 15,
    10, 10,
    "shrink", 5, 18, 250, 2500, 1.6,
    1, 0.15, 3, 1, 0.8, 5,
    True, 21, 12, 300, 15000,
    2.5, 4, 14,
    2.0, 1.0,
    False, 50.0, 50.0, 25.0, 25.0, 1.0, 20.0,
    "rectangle",
    "", "",
    "", "",
    "internal", "", "",
]

def collect_params(values):
    return dict(zip(PARAM_NAMES, values))


# ============================================================
# ========================= 新手預設集 (Presets) =================
# ============================================================
# 三組給新手的「起始參數」，都是以 DEFAULTS 為基底、只覆寫該情境下真正有感的幾個
# 參數（背景扣除強度、拉伸曲線、飽和度、RGB 增益、降噪等），而不是每個滑桿都亂動。
# 套用後仍是完整的一組參數（其餘沿用 DEFAULTS），方便新手有個「看起來對」的起點，
# 再自行微調，而不是從全部預設值(可能完全不適合該情境)開始瞎猜。
_PRESET_OVERRIDES = {
    "milky_way": {
        # 銀河模式：地景/銀河對比通常較強，背景漸層(光害/月光)明顯，飽和度可以拉高一些
        "bg_subtract": 0.95,
        "stretch_factor": 14.0,
        "sat_boost": 1.6,
        "clarity_strength": 0.4,
    },
    "nebula": {
        # 星雲模式：需要拉出更多微弱暗部細節，加強紅色(H-alpha)訊號，通常會去星以利後續疊圖
        "black_pct": 0.1,
        "stretch_factor": 20.0,
        "white_pct": 99.5,
        "sat_boost": 1.3,
        "r_gain": 1.15,
        "star_mode": "remove",
    },
    "heavy_light_pollution": {
        # 重光害：背景漸層更強更需要扣乾淨，白平衡容忍範圍加大以校正嚴重橘/黃色偏，
        # 飽和度降低避免光害色偏被放大，同時光害环境常伴隨高 ISO 雜訊，加強降噪
        "bg_subtract": 0.98,
        "bg_blur_sigma": 8.0,
        "wb_min": 0.5,
        "wb_max": 2.0,
        "g_gain": 0.9,
        "sat_boost": 1.1,
        "denoise_enable": True,
        "denoise_d": 7,
        "denoise_sigma_color": 20,
        "denoise_sigma_space": 20,
    },
}


def get_preset_values(preset_key):
    """回傳指定 preset 完整的一組參數值（依 PARAM_NAMES 順序），供 gr.update 套用。"""
    merged = dict(zip(PARAM_NAMES, DEFAULTS))
    merged.update(_PRESET_OVERRIDES[preset_key])
    return [merged[name] for name in PARAM_NAMES]


def apply_preset_fn(preset_key, lang):
    values = get_preset_values(preset_key)
    preset_label = {
        "milky_way":            ("銀河模式", "Milky Way"),
        "nebula":               ("星雲模式", "Nebula"),
        "heavy_light_pollution":("重光害",   "Heavy Light Pollution"),
    }[preset_key]
    name = preset_label[0] if lang == "zh" else preset_label[1]
    status = f"✅ 已套用「{name}」預設參數，可再自行微調" if lang == "zh" else f"✅ Applied the '{name}' preset — feel free to fine-tune further"
    return values + [status]


# ============================================================
# ========================= 參數快照 (Snapshots) ================
# ============================================================
# 讓使用者可以把「目前這組參數」暫存到 A / B / C 三個快取格，
# 快速在 2-3 組候選設定間切換比較，而不用每次都手動記/調參數。
# 快照本身（gr.State）只存在瀏覽器工作階段中，不寫入磁碟，重新整理頁面就會清空；
# 若想長期保存，可以用下面的「快照存成檔案 / 從檔案載入快照」，
# 或既有的「匯出當前參數(JSON)」功能。

def save_snapshot_fn(slot_label, lang, *param_values):
    p = collect_params(param_values)
    import datetime
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    status = f"✅ 快照 {slot_label} 已儲存目前參數（{ts}）" if lang == "zh" else f"✅ Snapshot {slot_label} saved ({ts})"
    return p, status


def load_snapshot_fn(snapshot, slot_label, lang):
    if snapshot is None:
        msg = f"⚠️ 快照 {slot_label} 是空的，請先按「儲存」" if lang == "zh" else f"⚠️ Snapshot {slot_label} is empty — save it first"
        return [gr.update() for _ in PARAM_NAMES] + [msg]
    updates = [gr.update(value=snapshot[name]) for name in PARAM_NAMES]
    msg = f"📥 已套用快照 {slot_label}" if lang == "zh" else f"📥 Snapshot {slot_label} applied"
    return updates + [msg]


def _sanitize_filename_component(name):
    """把使用者輸入的自由文字轉成安全的檔名片段：
    僅保留中日韓文字、英數字、底線、連字號、空白（空白轉底線），其餘字元一律捨棄；
    避免路徑穿越字元（/ \\ .. 等）造成寫到非預期的路徑。空字串或全部被濾掉時回傳 None，
    交由呼叫端套用預設檔名。"""
    if not name:
        return None
    import re
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff\u3040-\u30ff]", "", name)
    name = name.strip("._-")
    return name[:60] if name else None


# ============================================================
# ========================= 參數匯入/匯出 =========================
# ============================================================
# 檔名採「參數化」設計：使用者可選填一個名稱，匯出時會反映在檔名裡
# （例如 astro_config_後院光害.json），不填則沿用預設檔名 astro_config.json。
# 這是必要的前提修正——舊版固定寫死 astro_config.json，若使用者連續匯出
# 多組不同參數，後面的匯出會直接覆蓋前一個，且下面的「具名快照存檔」也需要
# 靠檔名區分彼此，才不會互相覆蓋。

def export_config_fn(cfg_name, *param_values):
    p = collect_params(param_values)
    out_dir = _default_output_base()
    os.makedirs(out_dir, exist_ok=True)
    safe_name = _sanitize_filename_component(cfg_name)
    cfg_filename = f"astro_config_{safe_name}.json" if safe_name else "astro_config.json"
    cfg_path = os.path.join(out_dir, cfg_filename)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(p, f, indent=4, ensure_ascii=False)
    return f"✅ 參數已成功匯出至：`{cfg_path}`", cfg_path

def import_config_fn(file_obj):
    if file_obj is None:
        return ["⚠️ 請選擇要匯入的 JSON 檔案"] + [gr.update() for _ in PARAM_NAMES]
    try:
        path = file_obj if isinstance(file_obj, str) else file_obj.name
        with open(path, "r", encoding="utf-8") as f:
            p = json.load(f)
        # 相容具名快照檔（{"params": {...}, "snapshot_name": ...}）與舊版純參數字典兩種格式
        if isinstance(p, dict) and "params" in p and isinstance(p["params"], dict):
            p = p["params"]

        updates = []
        for name in PARAM_NAMES:
            if name in p:
                updates.append(gr.update(value=p[name]))
            else:
                updates.append(gr.update())
        return ["✅ 參數檔載入成功！"] + updates
    except Exception as e:
        return [f"❌ 匯入失敗: {e}"] + [gr.update() for _ in PARAM_NAMES]


# ============================================================
# ==================== 快照存成檔案 / 從檔案載入快照 ================
# ============================================================
# 沿用上面 export_config_fn / import_config_fn 的 JSON 讀寫邏輯，
# 差別在於：這裡操作的對象是「已經存在 A/B/C 某一格裡的快照」（gr.State 裡的參數字典），
# 而不是目前滑桿上的即時參數；多包一層 snapshot_name/slot 中繼資料，
# 方便之後載入時能顯示這是哪一組、叫什麼名字。

def export_snapshot_to_file_fn(slot_label, snap_name, snap_a, snap_b, snap_c, lang):
    snapshot_map = {"A": snap_a, "B": snap_b, "C": snap_c}
    snapshot = snapshot_map.get(slot_label)
    if snapshot is None:
        msg = (f"⚠️ 快照 {slot_label} 是空的，請先按「儲存為 {slot_label}」" if lang == "zh"
               else f"⚠️ Snapshot {slot_label} is empty — save it first")
        return msg, None

    out_dir = _default_output_base()
    os.makedirs(out_dir, exist_ok=True)
    safe_name = _sanitize_filename_component(snap_name)
    file_stub = safe_name or f"slot_{slot_label}"
    path = os.path.join(out_dir, f"astro_snapshot_{file_stub}.json")

    import datetime
    payload = {
        "snapshot_name": snap_name or "",
        "slot": slot_label,
        "saved_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "params": snapshot,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)

    label = snap_name or slot_label
    msg = (f"✅ 快照「{label}」已存成檔案：`{path}`" if lang == "zh"
           else f"✅ Snapshot \"{label}\" saved to file: `{path}`")
    return msg, path


def import_snapshot_from_file_fn(file_obj, slot_label, lang):
    # 三個 slot 的 state / status 預設都不變動，只更新使用者選擇的目標 slot
    state_updates = [gr.update(), gr.update(), gr.update()]
    status_updates = [gr.update(), gr.update(), gr.update()]
    idx = {"A": 0, "B": 1, "C": 2}.get(slot_label, 0)

    if file_obj is None:
        msg = "⚠️ 請選擇要載入的快照 JSON 檔案" if lang == "zh" else "⚠️ Please select a snapshot JSON file"
        return (*state_updates, *status_updates, msg, gr.update())

    try:
        path = file_obj if isinstance(file_obj, str) else file_obj.name
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        # 相容「具名快照檔」與單純的參數 JSON（例如直接拿 astro_config.json 當快照用）
        if isinstance(payload, dict) and "params" in payload and isinstance(payload["params"], dict):
            params = payload["params"]
            snap_name = payload.get("snapshot_name", "")
        else:
            params = payload
            snap_name = ""

        # 只保留這個版本本來就認識的參數欄位，未知/缺漏的欄位交給現有滑桿值
        filtered = {name: params[name] for name in PARAM_NAMES if name in params}

        state_updates[idx] = filtered
        label = snap_name or slot_label
        status_updates[idx] = (f"✅ 快照「{label}」已從檔案載入到 {slot_label}（尚未套用到滑桿，請按「套用 {slot_label}」）"
                                if lang == "zh" else
                                f"✅ Snapshot \"{label}\" loaded into slot {slot_label} from file (not yet applied — click \"Apply {slot_label}\")")
        msg = status_updates[idx]
        name_update = gr.update(value=snap_name) if snap_name else gr.update()
        return (*state_updates, *status_updates, msg, name_update)
    except Exception as e:
        err_msg = f"❌ 快照載入失敗: {e}" if lang == "zh" else f"❌ Failed to load snapshot: {e}"
        return (*state_updates, *status_updates, err_msg, gr.update())



# ========================= Gradio 介面 =========================
# ============================================================

def scan_folder(folder):
    if not folder or not os.path.isdir(folder):
        return gr.update(choices=[], value=None)
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)])
    return gr.update(choices=files, value=files[0] if files else None)


def load_image_fn(folder, filename, uploaded, preview_max, lang="zh"):
    try:
        if uploaded is not None:
            path = uploaded if isinstance(uploaded, str) else uploaded.name
            base_name = os.path.splitext(os.path.basename(path))[0]
        elif filename:
            path = os.path.join(folder, filename)
            base_name = os.path.splitext(filename)[0]
        else:
            msg = "⚠️ 請先在資料夾清單選擇檔案，或直接上傳圖片" if lang == "zh" else "⚠️ Please select a file from the folder list, or upload an image directly"
            return None, None, 1.0, msg, gr.update(), None, None, None

        img = load_image_any(path)
        preview_base, preview_scale = make_preview_base(img, int(preview_max))
        h, w = img.shape[:2]
        status = f"✅ 已載入:**{os.path.basename(path)}**({w}×{h})" if lang == "zh" else f"✅ Loaded:**{os.path.basename(path)}**({w}×{h})"
        preview_uint8 = (preview_base * 255).astype(np.uint8)
        return img, preview_base, preview_scale, status, base_name, preview_uint8, preview_uint8, None
    except Exception as e:
        status = f"❌ 載入失敗: {e}" if lang == "zh" else f"❌ Load failed: {e}"
        return None, None, 1.0, status, gr.update(), None, None, None


def update_preview_fn(preview_base, preview_scale, full_img, use_full_res, live_preview_enabled, lang, *param_values):
    if not live_preview_enabled:
        # v1.2.1 主線 A：暫停即時預覽——全部輸出回傳 no-op，畫面停留在暫停前的結果，
        # 也完全不會走到下面的 run_pipeline（包含 mode="external" 的外部工具呼叫）。
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    use_full = bool(use_full_res) and full_img is not None
    img_to_use = full_img if use_full else preview_base
    scale_to_use = 1.0 if use_full else preview_scale
    if img_to_use is None:
        msg = "⚠️ 尚未載入圖片" if lang == "zh" else "⚠️ Image not loaded"
        return None, None, None, msg, build_compare_slider_html(None, None, lang)
    p = collect_params(param_values)
    try:
        t0 = time.perf_counter()
        result = run_pipeline(img_to_use, p, want_layers=False, preview_scale=scale_to_use)
        elapsed = time.perf_counter() - t0
        # v1.3.0：result['main'] 現在是 float32 [0,1]（供匯出用的全精度結果），
        # 這裡是純畫面顯示路徑，轉成 uint8 給 Gradio Image / 直方圖 / 比較滑桿用，
        # 不影響匯出時實際用到的精度。
        main_out_uint8 = (np.clip(result['main'], 0, 1) * 255).astype(np.uint8)
        hist_out = generate_histogram(main_out_uint8)
        original_uint8 = (np.clip(img_to_use, 0, 1) * 255).astype(np.uint8)
        tag = ("全解析度原圖運算" if lang == "zh" else "Full-res image processing") if use_full else ("縮圖運算" if lang == "zh" else "Thumbnail processing")
        slider_html = build_compare_slider_html(original_uint8, main_out_uint8, lang)
        status_msg = f"✅ 預覽與 RGB 曲線已更新({tag}，{elapsed:.2f}s)" if lang == "zh" else f"✅ Preview and RGB histogram updated ({tag}, {elapsed:.2f}s)"
        return main_out_uint8, original_uint8, hist_out, status_msg, slider_html
    except Exception as e:
        err_msg = f"❌ 預覽發生錯誤: {e}" if lang == "zh" else f"❌ Preview error: {e}"
        return None, None, None, err_msg, build_compare_slider_html(None, None, lang)


def resize_preview_base_fn(full_img, preview_max):
    if full_img is None:
        return None, 1.0
    return make_preview_base(full_img, int(preview_max))


def layer_preview_fn(preview_base, preview_scale, full_img, use_full_res, lang, *param_values):
    use_full = bool(use_full_res) and full_img is not None
    img_to_use = full_img if use_full else preview_base
    scale_to_use = 1.0 if use_full else preview_scale
    if img_to_use is None:
        msg = "⚠️ 尚未載入圖片" if lang == "zh" else "⚠️ Image not loaded"
        return None, None, None, msg
    p = collect_params(param_values)
    try:
        t0 = time.perf_counter()
        result = run_pipeline(img_to_use, p, want_layers=True, preview_scale=scale_to_use)
        elapsed = time.perf_counter() - t0
        mask = result.get('mask')
        # v1.3.0：result['starless'] 現在是 float32 [0,1]，這裡是畫面顯示用途，
        # 轉成 uint8 給 gr.Image；跟 update_preview_fn 一樣不影響匯出精度。
        starless_raw = result.get('starless')
        starless = None if starless_raw is None else (np.clip(starless_raw, 0, 1) * 255).astype(np.uint8)
        target_overlay = result.get('target_mask_overlay')
        tag = ("全解析度原圖" if lang == "zh" else "Full resolution image") if use_full else ("縮圖版本，星點參數已依縮圖比例等比例換算" if lang == "zh" else "Thumbnail version, star parameters scaled accordingly")
        if mask is None:
            msg = f"ℹ️ 目前參數沒有偵測到任何星點遮罩({elapsed:.2f}s)" if lang == "zh" else f"ℹ️ No star mask detected with current parameters ({elapsed:.2f}s)"
            if target_overlay is None:
                return None, None, None, msg
            # 星點遮罩雖然是空的，但自動局部拉伸遮罩仍然有效，照樣回傳讓使用者看到
            return None, starless, target_overlay, msg
        status_msg = f"✅ 圖層預覽已產生({tag}，僅供參考，{elapsed:.2f}s)" if lang == "zh" else f"✅ Layer previews generated ({tag}, for reference only, {elapsed:.2f}s)"
        return mask, starless, target_overlay, status_msg
    except Exception as e:
        err_msg = f"❌ 錯誤: {e}" if lang == "zh" else f"❌ Error: {e}"
        return None, None, None, err_msg


def local_preview_fn(full_img, x_pct, y_pct, crop_px, lang, *param_values):
    """局部預覽：從原始全解析度圖裁出一小塊，以全解析度品質跑完整 pipeline。
    
    優勢：
      - 裁切區域小 → 處理速度極快（通常 < 0.5s）
      - preview_scale=1.0 → 和最終匯出完全一致，不會有縮圖失真
    
    回傳：
      1. 處理後的裁切結果圖（全品質）
      2. 標示裁切框位置的縮略圖（紅框標示）
      3. 狀態訊息
    """
    if full_img is None:
        msg = "⚠️ 請先載入圖片" if lang == "zh" else "⚠️ Please load an image first"
        return None, None, msg

    h, w = full_img.shape[:2]
    crop_px = max(64, min(int(crop_px), min(h, w)))

    # 中心點座標（以百分比換算為像素）
    cx = int(w * x_pct / 100.0)
    cy = int(h * y_pct / 100.0)

    # 計算裁切邊界（clamp 到圖片範圍內）
    half = crop_px // 2
    x0 = max(0, min(cx - half, w - crop_px))
    y0 = max(0, min(cy - half, h - crop_px))
    x1 = x0 + crop_px
    y1 = y0 + crop_px

    crop_f32 = full_img[y0:y1, x0:x1]
    p = collect_params(param_values)

    try:
        # 以 preview_scale=1.0 跑完整 pipeline（全解析度精度）
        t0 = time.perf_counter()
        result = run_pipeline(crop_f32, p, want_layers=False, preview_scale=1.0)
        elapsed = time.perf_counter() - t0
        # v1.3.0：result['main'] 現在是 float32 [0,1]，這裡是畫面顯示用途，轉成 uint8。
        processed_uint8 = (np.clip(result['main'], 0, 1) * 255).astype(np.uint8)

        # 產生標示裁切框的縮略圖
        thumb_max = 700
        scale_t = min(1.0, thumb_max / max(h, w))
        tw, th = max(1, int(w * scale_t)), max(1, int(h * scale_t))
        thumb = cv2.resize((full_img * 255).astype(np.uint8), (tw, th), interpolation=cv2.INTER_AREA)
        # 畫紅色裁切框
        rx0, ry0 = int(x0 * scale_t), int(y0 * scale_t)
        rx1, ry1 = int(x1 * scale_t), int(y1 * scale_t)
        cv2.rectangle(thumb, (rx0, ry0), (rx1, ry1), (255, 60, 60), max(2, int(3 * scale_t)))
        # 標示尺寸
        label = f"{crop_px}×{crop_px} px"
        cv2.putText(thumb, label, (rx0 + 4, max(ry0 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 60, 60), 1, cv2.LINE_AA)

        coords_str = f"({x0},{y0})→({x1},{y1})"
        msg = (f"✅ 局部預覽完成，裁切區域 {coords_str}，全解析度品質，{elapsed:.2f}s"
               if lang == "zh" else
               f"✅ Local preview done, crop {coords_str}, full-res quality, {elapsed:.2f}s")
        return processed_uint8, thumb, msg

    except Exception as e:
        err = f"❌ 局部預覽錯誤: {e}" if lang == "zh" else f"❌ Local preview error: {e}"
        return None, None, err


def export_fn(full_img, out_dir, out_name, want_layers, lang, export_formats, *param_values):
    # v1.3.3 Track B：新增第 3 個回傳值 elapsed（原始 float 秒數，不是格式化字串），
    # 提供給呼叫端的 .then() 串接餵給 get_status_bar_html(proc_time=...)，讓底部
    # 狀態列的「⏱ 時間」欄位第一次真正顯示數值。訊息文字本身格式不變。
    if full_img is None:
        msg = "⚠️ 請先載入圖片" if lang == "zh" else "⚠️ Please load an image first"
        return msg, None, None
    if not out_name:
        out_name = "processed"
    p = collect_params(param_values)
    t0 = time.perf_counter()
    try:
        os.makedirs(out_dir, exist_ok=True)
        # 全解析度匯出：preview_scale=1.0，星點縮小/去星參數直接採用使用者設定的原始數值
        result = run_pipeline(full_img, p, want_layers=want_layers, preview_scale=1.0)
        files = []
        jpg_path, tif_path = save_image_files(result['main'], out_dir, out_name, formats=export_formats)
        files += [f for f in (jpg_path, tif_path) if f]
        if want_layers and 'mask' in result:
            mask_path = os.path.join(out_dir, f"{out_name}_starmask.png")
            cv2.imwrite(mask_path, result['mask'])
            files.append(mask_path)
            sl_jpg, sl_tif = save_image_files(result['starless'], out_dir, f"{out_name}_starless", formats=export_formats)
            files += [f for f in (sl_jpg, sl_tif) if f]
        elapsed = time.perf_counter() - t0
        msg = (f"✅ 匯出完成，共 {len(files)} 個檔案，總耗時 {elapsed:.2f}s → `{out_dir}`"
               if lang == "zh" else
               f"✅ Export completed, {len(files)} files, total {elapsed:.2f}s → `{out_dir}`")
        return msg, files, elapsed
    except Exception as e:
        elapsed = time.perf_counter() - t0
        err_msg = (f"❌ 匯出失敗（耗時 {elapsed:.2f}s）: {e}" if lang == "zh"
                   else f"❌ Export failed (after {elapsed:.2f}s): {e}")
        # 失敗仍回傳 elapsed（失敗前實際花了多久），比留白／None 更有資訊量，
        # 也讓狀態列不會因為一次失敗的匯出就整個「消失」數值。
        return err_msg, None, elapsed


def batch_process_fn(folder, out_dir, want_layers, lang, stop_state, formats, *param_values):
    """批次處理：對資料夾內每一張圖套用目前的參數集，逐一以全解析度處理並匯出。

    設計成 generator（每處理完一張就 yield 一次最新狀態文字），這樣 Gradio 前端
    可以即時顯示進度，而不用等全部檔案跑完才有畫面回饋——批次動輒數十張全解析度圖，
    单次處理可能要好幾秒到幾十秒，沒有進度回饋的話使用者會誤以為卡死。

    單張圖失敗（例如檔案損毀、記憶體不足）不會中斷整批次，只會記錄錯誤並繼續下一張，
    最後在摘要裡列出成功/失敗數量與失敗檔名，方便使用者事後排查。

    stop_state: v1.3.2 新增。一個可變的 dict（見 state_batch_stop 定義處的說明），
        由「停止批次」按鈕原地修改 stop_state["stop"] = True。這裡在每張圖片
        開始處理前檢查一次——檢查點刻意放在單張圖片的邊界，而不是圖片處理到
        一半就強制中斷，這樣不會留下寫到一半的殘缺輸出檔，語意上也比較單純
        （已經在跑的這張圖片一定會跑完）。
    formats: 匯出格式清單（例如 ["JPG", "TIFF"]），轉呼叫 save_image_files()。
    """
    if not folder or not os.path.isdir(folder):
        msg = "⚠️ 請先輸入有效的來源資料夾路徑" if lang == "zh" else "⚠️ Please enter a valid source folder path"
        yield msg
        return

    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)])
    if not files:
        msg = "⚠️ 資料夾內沒有找到可處理的圖片" if lang == "zh" else "⚠️ No processable images found in this folder"
        yield msg
        return

    if not out_dir:
        out_dir = os.path.join(_default_output_base(), "batch")
    os.makedirs(out_dir, exist_ok=True)
    p = collect_params(param_values)

    # 每次新批次開跑時，把（可能殘留自上一批次的）停止旗標重設為 False。
    # 這裡「原地修改」而不是重新 assign 一個新 dict，確保「停止」按鈕日後
    # 修改的仍是同一份物件（詳見函式開頭與 state_batch_stop 定義處的說明）。
    if isinstance(stop_state, dict):
        stop_state["stop"] = False
    else:
        stop_state = {"stop": False}

    total = len(files)
    done, failed = 0, []
    log_lines = []
    stopped_early = False

    header = (f"🚀 批次處理開始，共 {total} 張圖片 → 輸出至 `{out_dir}`"
               if lang == "zh" else
               f"🚀 Batch started, {total} image(s) → output to `{out_dir}`")
    yield header

    batch_t0 = time.perf_counter()
    for i, fname in enumerate(files, 1):
        if isinstance(stop_state, dict) and stop_state.get("stop"):
            stopped_early = True
            break

        base_name = os.path.splitext(fname)[0]
        file_t0 = time.perf_counter()
        try:
            img = load_image_any(os.path.join(folder, fname))
            result = run_pipeline(img, p, want_layers=want_layers, preview_scale=1.0)
            save_image_files(result['main'], out_dir, base_name, formats=formats)
            if want_layers and 'mask' in result:
                cv2.imwrite(os.path.join(out_dir, f"{base_name}_starmask.png"), result['mask'])
                save_image_files(result['starless'], out_dir, f"{base_name}_starless", formats=formats)
            file_elapsed = time.perf_counter() - file_t0
            done += 1
            log_lines.append(f"✅ [{i}/{total}] {fname}（{file_elapsed:.2f}s）" if lang == "zh"
                              else f"✅ [{i}/{total}] {fname} ({file_elapsed:.2f}s)")
        except Exception as e:
            file_elapsed = time.perf_counter() - file_t0
            failed.append(fname)
            log_lines.append(f"❌ [{i}/{total}] {fname}（{file_elapsed:.2f}s）: {e}" if lang == "zh"
                              else f"❌ [{i}/{total}] {fname} ({file_elapsed:.2f}s): {e}")

        # 依目前已處理張數的平均耗時，推估剩餘時間，方便使用者判斷還要等多久
        elapsed_so_far = time.perf_counter() - batch_t0
        avg_per_file = elapsed_so_far / i
        remaining = max(0.0, avg_per_file * (total - i))
        eta_line = (f"⏱️ 已耗時 {elapsed_so_far:.1f}s，預估剩餘 {remaining:.1f}s（平均每張 {avg_per_file:.2f}s）"
                    if lang == "zh" else
                    f"⏱️ Elapsed {elapsed_so_far:.1f}s, estimated {remaining:.1f}s remaining (avg {avg_per_file:.2f}s/file)")

        progress_tail = "\n".join(log_lines[-8:])  # 只顯示最近 8 行，避免訊息無限增長
        status = (f"{header}\n\n⏳ 進度: {i}/{total}（成功 {done}，失敗 {len(failed)}）\n{eta_line}\n\n{progress_tail}"
                   if lang == "zh" else
                   f"{header}\n\n⏳ Progress: {i}/{total} (success {done}, failed {len(failed)})\n{eta_line}\n\n{progress_tail}")
        yield status

    total_elapsed = time.perf_counter() - batch_t0
    if stopped_early:
        remaining_count = total - done - len(failed)
        summary = (f"⏹ 批次已依使用者要求停止：成功 {done} 張，失敗 {len(failed)} 張，"
                   f"尚未處理 {remaining_count} 張，總耗時 {total_elapsed:.1f}s → `{out_dir}`"
                   if lang == "zh" else
                   f"⏹ Batch stopped by user: {done} succeeded, {len(failed)} failed, "
                   f"{remaining_count} not processed, total {total_elapsed:.1f}s → `{out_dir}`")
        if failed:
            fail_list = "、".join(failed) if lang == "zh" else ", ".join(failed)
            summary += (f"\n\n失敗檔案：{fail_list}" if lang == "zh" else f"\n\nFailed files: {fail_list}")
    elif failed:
        fail_list = "、".join(failed) if lang == "zh" else ", ".join(failed)
        summary = (f"🏁 批次處理完成：成功 {done} 張，失敗 {len(failed)} 張，總耗時 {total_elapsed:.1f}s → `{out_dir}`\n\n失敗檔案：{fail_list}"
                    if lang == "zh" else
                    f"🏁 Batch finished: {done} succeeded, {len(failed)} failed, total {total_elapsed:.1f}s → `{out_dir}`\n\nFailed files: {fail_list}")
    else:
        summary = (f"🏁 批次處理完成，全部 {done} 張圖片皆已成功匯出，總耗時 {total_elapsed:.1f}s → `{out_dir}`"
                    if lang == "zh" else
                    f"🏁 Batch finished, all {done} image(s) exported successfully, total {total_elapsed:.1f}s → `{out_dir}`")
    yield summary


# ============================================================
# ========================= i18n 國際化 =========================
# ============================================================

LANG = {
    "zh": {
        "compare_side":   "並排顯示",
        "compare_slider": "滑桿疊圖(Lightroom 風格)",
        "before_label":   "原圖(未處理)",
        "after_label":    "即時預覽效果(處理後)",
        "layer_btn":      "🔄 產生 / 更新圖層預覽",
        "info_hist":      "📊 Histogram",
        "info_mask":      "⭐ Star Mask 預覽",
        "info_monitor":   "🖥️ 系統監控",
        "monitor_btn":    "🔄 更新監控資訊",
        "load_btn":       "📥 載入圖片",
        "scan_btn":       "🔍 掃描資料夾",
        "export_cfg_btn": "📤 匯出當前參數",
        "export_btn":     "🚀 開始高解析度匯出",
        "reset_btn":      "↩️ 重設為預設值",
    },
    "en": {
        "compare_side":   "Side by Side",
        "compare_slider": "Slider Overlay",
        "before_label":   "Before (Original)",
        "after_label":    "After (Processed)",
        "layer_btn":      "🔄 Generate Layer Preview",
        "info_hist":      "📊 Histogram",
        "info_mask":      "⭐ Star Mask Preview",
        "info_monitor":   "🖥️ System Monitor",
        "monitor_btn":    "🔄 Refresh Monitor",
        "load_btn":       "📥 Load Image",
        "scan_btn":       "🔍 Scan Folder",
        "export_cfg_btn": "📤 Export Config",
        "export_btn":     "🚀 Start Export",
        "reset_btn":      "↩️ Reset Defaults",
    },
}

UI_TRANSLATIONS = {
    "folder_box": {"zh": "本機資料夾路徑", "en": "Local Folder Path"},
    "scan_btn": {"zh": "🔍 掃描資料夾", "en": "🔍 Scan Folder"},
    "file_dropdown": {"zh": "資料夾內圖片", "en": "Images in Folder"},
    "upload_file": {"zh": "或直接上傳圖片 (tif/jpg/png/RAW)", "en": "Or Upload Image Directly (tif/jpg/png/RAW)"},
    "preview_size": {
        "zh": ("預覽解析度(長邊像素)",  "縮圖越小處理越快，不影響最終匯出解析度"),
        "en": ("Preview Resolution (Max Dimension px)", "Smaller thumbnail = faster preview; does not affect final export quality"),
    },
    "use_full_res_preview": {
        "zh": ("即時預覽改用原圖全解析度運算(較慢但最準確)", "未勾選時使用縮圖運算，速度較快；勾選後直接用原圖跑全部流程"),
        "en": ("Use Full Resolution for Live Preview (Slower but accurate)", "When unchecked, uses a downscaled thumbnail for speed; when checked, runs the full pipeline on the original image"),
    },
    "live_preview_enabled": {
        "zh": ("即時預覽（取消勾選可暫停自動重算）",
               "暫停後，調整滑桿/開關不會立即重新運算，畫面停留在暫停前的結果；"
               "適合連續調整多個參數、或使用 external 模式時避免重複呼叫外部工具。"
               "重新勾選會立刻用目前參數更新一次。"),
        "en": ("Live Preview (Uncheck to Pause Auto-Recompute)",
               "When paused, adjusting sliders/toggles won't trigger an immediate recompute — the view stays "
               "on the result from before pausing. Useful when tweaking several parameters in a row, or with "
               "external mode to avoid repeatedly calling the external tool. Re-checking refreshes immediately "
               "with the current parameters."),
    },
    "load_btn": {"zh": "📥 載入圖片", "en": "📥 Load Image"},
    "cfg_export_btn": {"zh": "📤 匯出當前參數", "en": "📤 Export Current Parameters"},
    "cfg_import_file": {"zh": "匯入參數 JSON", "en": "Import Config JSON"},
    "bg_enable": {"zh": "啟用背景漸層去除", "en": "Enable Background Gradient Removal"},
    "bg_downscale": {
        "zh": ("估算縮圖比例", "越小越快，但過小會失真"),
        "en": ("Background Downscale Ratio", "Smaller = faster, but too small causes inaccuracy"),
    },
    "bg_min_filter": {
        "zh": ("局部最暗值視窗", "星雲核心較大時應加大，避免誤判為背景"),
        "en": ("Local Minimum Filter Size", "Increase for larger nebula cores to avoid misclassifying them as background"),
    },
    "bg_blur_sigma": {
        "zh": ("背景平滑程度", "越高背景漸層過渡越平滑"),
        "en": ("Background Blur Sigma", "Higher = smoother background gradient transition"),
    },
    "bg_subtract": {
        "zh": ("扣除強度", "1.0 完全扣除，0.90-0.95 能保留自然天光"),
        "en": ("Subtraction Strength", "1.0 = full subtraction; 0.90-0.95 preserves natural sky glow"),
    },
    "wb_enable": {"zh": "啟用色偏校正(白平衡)", "en": "Enable Color Cast Correction (White Balance)"},
    "wb_min": {
        "zh": ("增益下限", "限制最大縮小幅度，防顏色死掉"),
        "en": ("Minimum Gain Limit", "Limits how much a channel can be reduced, preventing color clipping"),
    },
    "wb_max": {
        "zh": ("增益上限", "限制最大放大倍率，防特定通道雜訊爆發"),
        "en": ("Maximum Gain Limit", "Limits how much a channel can be boosted, preventing noise blowup in that channel"),
    },
    "black_pct": {
        "zh": ("黑點百分位", "通常設 0.1-0.5%"),
        "en": ("Black Point Percentile", "Typically set to 0.1-0.5%"),
    },
    "stretch_factor": {
        "zh": ("拉伸強度(arcsinh)", "越高微弱星雲越明顯"),
        "en": ("Stretch Factor (arcsinh)", "Higher = faint nebulosity becomes more visible"),
    },
    "white_pct": {
        "zh": ("白點百分位", "99.7% 代表最亮前 0.3% 飽和成純白"),
        "en": ("White Point Percentile", "99.7% means the brightest 0.3% of pixels are saturated to pure white"),
    },
    "acc_local_target": {"zh": "3️⃣b 自動局部拉伸(星雲/銀河區域)", "en": "3️⃣b Auto Localized Stretch (Nebula/Galaxy)"},
    "local_target_enable": {
        "zh": ("啟用自動局部拉伸", "自動抓出有結構的星雲/銀河區域，只加強該處對比，天空背景不受影響"),
        "en": ("Enable Auto Localized Stretch", "Automatically finds textured nebula/galaxy regions and boosts contrast only there, leaving flat sky background untouched"),
    },
    "local_target_strength": {
        "zh": ("局部拉伸強度", "0 代表關閉效果"),
        "en": ("Local Stretch Strength", "0 disables the effect"),
    },
    "local_target_radius": {
        "zh": ("偵測/加強半徑(px)", "約略對應星雲結構的尺度，太小會連星點都吃到"),
        "en": ("Detection/Boost Radius (px)", "Roughly matches the scale of nebula structures; too small will also pick up stars"),
    },
    "local_target_sensitivity": {
        "zh": ("偵測靈敏度", "越高，越多區域會被判定為『目標』而套用加強"),
        "en": ("Detection Sensitivity", "Higher = more area gets classified as 'target' and boosted"),
    },
    "manual_target_hint_md": {
        "zh": ("**手動框選加強區域**：自動偵測有時會漏掉較弱的目標、或誤判雜訊區。"
               "啟用後可框選一塊區域（矩形或橢圓/圓形），不論自動偵測結果如何，該區域一律套用上面的局部拉伸強度加強。"
               "框選區域會和自動遮罩取聯集，可在右側自動局部拉伸遮罩預覽圖中確認實際涵蓋範圍。"),
        "en": ("**Manual Boost Region**: Auto-detection can miss a faint target or misjudge a noisy patch. "
               "Enable this to select a region (rectangle or ellipse/circle) that's always boosted with the strength above, "
               "regardless of auto-detection. The region is merged with the auto mask (union), and you can "
               "confirm the actual coverage in the Auto Localized Stretch mask preview on the right."),
    },
    "manual_target_enable": {
        "zh": ("啟用手動框選區域", "即使自動局部拉伸沒有偵測到任何區域，仍可只用手動框選的區域加強"),
        "en": ("Enable Manual Boost Region", "Works even if Auto Localized Stretch detects nothing — the manual region can boost on its own"),
    },
    "manual_target_shape": {
        "zh": ("框選形狀", "矩形，或橢圓/正圓（寬高百分比相等時即為正圓）"),
        "en": ("Region Shape", "Rectangle, or ellipse/circle (set width % = height % for a perfect circle)"),
    },
    "manual_target_x_pct": {
        "zh": ("X 中心位置 (%)", "框選區域中心點的水平位置，以整張圖寬度的百分比表示"),
        "en": ("X Center (%)", "Horizontal position of the region's center, as a percentage of image width"),
    },
    "manual_target_y_pct": {
        "zh": ("Y 中心位置 (%)", "框選區域中心點的垂直位置，以整張圖高度的百分比表示"),
        "en": ("Y Center (%)", "Vertical position of the region's center, as a percentage of image height"),
    },
    "manual_target_w_pct": {
        "zh": ("框選寬度 (%)", "框選區域的寬度，以整張圖寬度的百分比表示"),
        "en": ("Region Width (%)", "Width of the boost region, as a percentage of image width"),
    },
    "manual_target_h_pct": {
        "zh": ("框選高度 (%)", "框選區域的高度，以整張圖高度的百分比表示"),
        "en": ("Region Height (%)", "Height of the boost region, as a percentage of image height"),
    },
    "manual_target_weight": {
        "zh": ("框選區域加強權重", "框選範圍內的遮罩值，1.0=與自動偵測最強處等級的加強"),
        "en": ("Region Boost Weight", "Mask value inside the region; 1.0 = as strong as the auto-detected mask's maximum"),
    },
    "manual_target_feather_pct": {
        "zh": ("邊緣羽化 (%)", "羽化程度佔框選區域短邊的百分比，越高邊界越柔和自然，0=硬邊"),
        "en": ("Edge Feather (%)", "Feather amount as a percentage of the region's shorter side; higher = softer edge, 0 = hard edge"),
    },
    "sat_boost": {
        "zh": ("飽和度倍率", "銀河/星雲通常需 1.2-1.8 倍增益"),
        "en": ("Saturation Boost Factor", "Milky Way/nebula shots typically need a 1.2-1.8x boost"),
    },
    "bright_boost": {
        "zh": ("明度倍率", "整體亮度微調，通常接近 1.0 即可，避免過曝"),
        "en": ("Brightness Boost Factor", "Fine-tunes overall brightness; keep close to 1.0 to avoid overexposure"),
    },
    "r_gain": {
        "zh": ("🔴 紅色通道增益", "加強發射星雲 H-alpha 訊號"),
        "en": ("🔴 Red Gain", "Boosts emission nebula H-alpha signal"),
    },
    "g_gain": {
        "zh": ("🟢 綠色通道增益", "通常用來壓低綠色夜空光害"),
        "en": ("🟢 Green Gain", "Typically used to reduce green light-pollution cast"),
    },
    "b_gain": {
        "zh": ("🔵 藍色通道增益", "加強反射星雲或藍色年輕恆星"),
        "en": ("🔵 Blue Gain", "Boosts reflection nebulae or blue young stars"),
    },
    "clarity_blur": {
        "zh": ("Clarity 模糊半徑", "越大越偏向中尺度結構"),
        "en": ("Clarity Blur Radius", "Larger = emphasizes mid-scale structure more"),
    },
    "clarity_strength": {
        "zh": ("Clarity 強度", "類似 Lightroom 清晰度"),
        "en": ("Clarity Strength", "Similar to Lightroom's Clarity"),
    },
    "sharpen_blur": {
        "zh": ("銳化模糊半徑", "決定銳化鎖定的細節尺度，越大銳化範圍越粗"),
        "en": ("Sharpen Blur Radius", "Sets the detail scale that sharpening targets; larger values give coarser sharpening"),
    },
    "sharpen_amount": {
        "zh": ("銳化程度", "過大會使雜訊粒子變粗"),
        "en": ("Sharpen Amount", "Too high will make noise grain coarser"),
    },
    "sharpen_mode": {
        "zh": ("銳化模式", "Clarity 永遠套用內建演算法；此處只切換「銳化」子步驟：internal=沿用上方模糊半徑/程度的 unsharp-mask; external=改呼叫外部 Cosmic Clarity Sharpen（見下方設定）"),
        "en": ("Sharpen Mode", "Clarity always uses the built-in algorithm; this only switches the sharpening sub-step: internal = the unsharp-mask above (blur radius/amount); external = calls external Cosmic Clarity Sharpen (see settings below)"),
    },
    "sharpen_ext_path": {
        "zh": ("[external]Cosmic Clarity Sharpen 執行檔路徑",
               "呼叫使用者電腦上『自行安裝、自行取得授權』的 Seti Astro Cosmic Clarity（Sharpen）；"
               "本程式不內建、不重新散布任何第三方模型，授權責任由使用者自行負責。找不到執行檔或呼叫失敗時，"
               "本步驟會自動跳過並回傳僅套用 Clarity、未銳化的影像，不會中斷處理。"),
        "en": ("[external] Cosmic Clarity Sharpen Executable Path",
               "Calls Seti Astro Cosmic Clarity (Sharpen) that the user has installed and licensed themselves. "
               "This app never bundles or redistributes third-party models — licensing responsibility stays "
               "with the user. If the executable is missing or the call fails, this step is skipped and the "
               "Clarity-only (unsharpened) image is returned; the pipeline is not interrupted."),
    },
    "sharpen_ext_profile": {
        "zh": ("[external]工具範本（自動帶入下方參數）",
               "選了範本會自動把對應語法填進「額外命令列參數」欄位，仍可手動再調整；"
               "旗標已對照 setiastro/cosmicclarity 原始碼確認，詳見優化計畫文件。"),
        "en": ("[external] Tool Preset (auto-fills the field below)",
               "Selecting a preset auto-fills the matching syntax into \"Extra Command-line Arguments\"; "
               "you can still edit it manually afterward. Flags have been verified against the "
               "setiastro/cosmicclarity source — see the optimization plan document for details."),
    },
    "sharpen_ext_args": {
        "zh": ("[external]額外命令列參數", "必須同時包含 --sharpening_mode 與 --nonstellar_strength 才會觸發 headless 模式，否則會跳過本次呼叫；其餘旗標可選"),
        "en": ("[external] Extra Command-line Arguments", "Must include both --sharpening_mode and --nonstellar_strength to trigger headless mode, otherwise the call is skipped; other flags are optional"),
    },
    "denoise_enable": {"zh": "啟用降噪", "en": "Enable Denoise"},
    "denoise_mode": {
        "zh": ("降噪模式", "fast=雙邊濾波(快); quality=Non-local Means(較乾淨但慢很多，適合最終匯出，全精度需安裝 scikit-image，未安裝則退回 8-bit 內部處理); external=呼叫外部 ML 降噪工具(見下方設定)"),
        "en": ("Denoise Mode", "fast = Bilateral Filter (quick); quality = Non-local Means (cleaner but much slower, best for final export; full precision requires scikit-image, otherwise falls back to 8-bit internal processing); external = calls an external ML denoiser (see settings below)"),
    },
    "denoise_ext_path": {
        "zh": ("[external]外部降噪工具執行檔路徑",
               "呼叫使用者電腦上『自行安裝、自行取得授權』的外部降噪工具(如 NoiseXTerminator / DeepSNR CLI)；"
               "本程式不內建、不重新散布任何第三方模型，授權責任由使用者自行負責。找不到執行檔或呼叫失敗時，"
               "本步驟會自動跳過並回傳未降噪的原圖，不會中斷處理。"),
        "en": ("[external] External Denoise Tool Path",
               "Calls an external denoising tool that the user has installed and licensed themselves "
               "(e.g. NoiseXTerminator / DeepSNR CLI). This app never bundles or redistributes third-party "
               "models — licensing responsibility stays with the user. If the executable is missing or the "
               "call fails, this step is skipped and the unprocessed image is returned; the pipeline is not interrupted."),
    },
    "denoise_ext_profile": {
        "zh": ("[external]工具範本（自動帶入下方參數）",
               "選了範本會自動把對應語法填進「額外命令列參數」欄位，仍可手動再調整；"
               "例如 DeepSNR 範本裡的 -m/-s 數值可依圖片內容自行微調。"
               "語法查證狀態請參考優化計畫文件，尚未實測的範本可能需要自行微調。"),
        "en": ("[external] Tool Preset (auto-fills the field below)",
               "Selecting a preset auto-fills the matching syntax into \"Extra Command-line Arguments\"; "
               "you can still edit it manually afterward — e.g. the DeepSNR preset's -m/-s values may need "
               "tuning depending on the image. See the optimization plan document for each preset's "
               "verification status; presets not yet field-tested may need adjustment."),
    },
    "denoise_ext_args": {
        "zh": ("[external]額外命令列參數", "可用 {input} / {output} / {output_noext} 佔位符；留空則預設為「執行檔 輸入檔 輸出檔」"),
        "en": ("[external] Extra Command-line Arguments", "Use {input} / {output} / {output_noext} placeholders; if left blank, defaults to \"executable input_file output_file\""),
    },
    "denoise_d": {
        "zh": ("[fast]濾波視窗", "較大數值降噪範圍廣但耗時"),
        "en": ("[fast] Bilateral Filter Diameter (d)", "Larger values denoise a wider area but take longer"),
    },
    "denoise_sigma_color": {
        "zh": ("[fast]顏色 Sigma", "越高越能融合差異較大的顏色，但可能糊掉色彩邊界"),
        "en": ("[fast] Denoise Sigma Color", "Higher values blend more dissimilar colors together, but may blur color boundaries"),
    },
    "denoise_sigma_space": {
        "zh": ("[fast]空間 Sigma", "越高影響範圍越大的鄰近像素，降噪更平滑但更慢"),
        "en": ("[fast] Denoise Sigma Space", "Higher values affect a wider neighborhood of pixels — smoother denoising but slower"),
    },
    "denoise_nlm_h": {
        "zh": ("[quality]亮度降噪強度 h", "越高越乾淨，但可能抹掉細節"),
        "en": ("[quality] Luminance Denoise Strength (h)", "Higher = cleaner but may erase detail"),
    },
    "denoise_nlm_h_color": {
        "zh": ("[quality]色彩降噪強度 hColor", "越高色彩雜訊越乾淨，但可能造成色塊化"),
        "en": ("[quality] Color Denoise Strength (hColor)", "Higher values give cleaner color noise removal, but may cause color blotching"),
    },
    "star_mode": {
        "zh": ("模式", "external=呼叫外部 ML 去星工具(見下方設定)"),
        "en": ("Star Processing Mode", "external = calls an external ML star-removal tool (see settings below)"),
    },
    "star_ext_path": {
        "zh": ("[external]外部去星工具執行檔路徑",
               "呼叫使用者電腦上『自行安裝、自行取得授權』的外部去星工具(如 StarXTerminator / StarNet CLI)；"
               "本程式不內建、不重新散布任何第三方模型，授權責任由使用者自行負責。找不到執行檔或呼叫失敗時，"
               "本步驟會自動跳過並回傳未去星的原圖，不會中斷處理。"),
        "en": ("[external] External Star Removal Tool Path",
               "Calls an external star-removal tool that the user has installed and licensed themselves "
               "(e.g. StarXTerminator / StarNet CLI). This app never bundles or redistributes third-party "
               "models — licensing responsibility stays with the user. If the executable is missing or the "
               "call fails, this step is skipped and the unprocessed image is returned; the pipeline is not interrupted."),
    },
    "star_ext_profile": {
        "zh": ("[external]工具範本（自動帶入下方參數）",
               "選了範本會自動把對應語法填進「額外命令列參數」欄位，仍可手動再調整；"
               "語法查證狀態請參考優化計畫文件，尚未實測的範本可能需要自行微調。"),
        "en": ("[external] Tool Preset (auto-fills the field below)",
               "Selecting a preset auto-fills the matching syntax into \"Extra Command-line Arguments\"; "
               "you can still edit it manually afterward. See the optimization plan document for each "
               "preset's verification status; presets not yet field-tested may need adjustment."),
    },
    "star_ext_args": {
        "zh": ("[external]額外命令列參數", "可用 {input} / {output} / {output_noext} 佔位符；留空則預設為「執行檔 輸入檔 輸出檔」"),
        "en": ("[external] Extra Command-line Arguments", "Use {input} / {output} / {output_noext} placeholders; if left blank, defaults to \"executable input_file output_file\""),
    },
    "star_kernel": {
        "zh": ("偵測核大小(≈星點直徑px)", "應略大於想抓取的中小型星點直徑"),
        "en": ("Star Detection Kernel Size (px)", "Should be slightly larger than the diameter of the small/medium stars you want to detect"),
    },
    "star_thresh": {
        "zh": ("偵測門檻", "越低暗星越多，過低會誤抓背景熱雜訊"),
        "en": ("Star Detection Threshold", "Lower = more faint stars detected; too low will pick up background hot-pixel noise"),
    },
    "star_max_area": {
        "zh": ("星點最大面積", "超過此面積的斑塊不視為一般星點，避免誤抓星雲亮核"),
        "en": ("Maximum Star Area (px²)", "Blobs larger than this area are not treated as ordinary stars, to avoid mistakenly capturing bright nebula cores"),
    },
    "star_max_area_large": {
        "zh": ("亮星暈光面積上限", "超過此面積直接排除，避免大範圍暈光被誤判為星點"),
        "en": ("Maximum Bright Star Halo Area (px²)", "Blobs larger than this are excluded outright, to avoid large halos being misidentified as stars"),
    },
    "star_aspect": {
        "zh": ("圓度門檻(長寬比)", "排除長條形星雲結構"),
        "en": ("Star Aspect Ratio Threshold", "Excludes elongated nebula structures"),
    },
    "star_dilate": {
        "zh": ("遮罩外擴基本像素", "每個星點遮罩固定外擴的像素數，確保完整覆蓋星點邊緣"),
        "en": ("Mask Dilation Base px", "Fixed number of pixels each star mask is expanded by, to fully cover star edges"),
    },
    "star_dilate_scale": {
        "zh": ("依星點大小外擴比例", "星點越大外擴越多，避免大星周圍殘留光暈"),
        "en": ("Mask Dilation Size-dependent Scale", "Larger stars are expanded proportionally more, to avoid leftover halos around bright stars"),
    },
    "star_shrink_kernel": {
        "zh": ("[縮星]侵蝕核大小", "每次侵蝕使用的核心大小，越大縮星效果越明顯"),
        "en": ("[Shrink] Erosion Kernel Size", "Kernel size used per erosion pass; larger values give a stronger shrink effect"),
    },
    "star_shrink_iter": {
        "zh": ("[縮星]侵蝕次數", "重複侵蝕的次數，越多星點縮得越小"),
        "en": ("[Shrink] Erosion Iterations", "Number of erosion passes; more passes shrink stars further"),
    },
    "star_shrink_strength": {
        "zh": ("[縮星]套用強度", "0 為不縮星，1 為完全套用侵蝕結果"),
        "en": ("[Shrink] Apply Strength", "0 = no shrinking, 1 = fully apply the erosion result"),
    },
    "star_inpaint_radius": {
        "zh": ("[去星]單星取樣半徑", "從星點周圍多遠的範圍取樣來填補去星後的背景"),
        "en": ("[Remove] Inpaint Radius (px)", "How far around each star to sample from when filling in the background after removal"),
    },
    "star_feather_px": {
        "zh": ("[去星]邊緣羽化程度", "去星邊界的柔化寬度，越高過渡越自然但越模糊"),
        "en": ("[Remove] Edge Feathering px", "Softening width at the star-removal boundary; higher values blend more naturally but look blurrier"),
    },
    "star_noise_strength": {
        "zh": ("[去星]雜訊回填強度", "0 為不回填"),
        "en": ("[Remove] Noise Infill Strength", "0 = no noise infill"),
    },
    "multiscale_enable": {"zh": "啟用大範圍偵測", "en": "Enable Multi-scale Star Detection"},
    "cluster_kernel": {
        "zh": ("偵測核大小", "用來偵測大範圍密集星團的局部視窗，應大於一般星點"),
        "en": ("Cluster Detection Kernel Size", "Local window used to detect large, dense star clusters; should be larger than a typical star"),
    },
    "cluster_thresh": {
        "zh": ("偵測門檻", "越低越容易把稀疏星群也判定為星團"),
        "en": ("Cluster Detection Threshold", "Lower values make sparse star groups more likely to be classified as clusters"),
    },
    "cluster_min_area": {
        "zh": ("最小面積", "小於此面積不視為星團，避免與一般星點混淆"),
        "en": ("Minimum Cluster Area (px²)", "Blobs smaller than this area are not treated as clusters, to avoid confusion with ordinary stars"),
    },
    "cluster_max_area": {
        "zh": ("最大面積", "超過此面積可能是星雲亮區而非星團，會被排除"),
        "en": ("Maximum Cluster Area (px²)", "Blobs larger than this may be bright nebula regions rather than clusters, and are excluded"),
    },
    "cluster_aspect": {
        "zh": ("長寬比門檻", "排除過於狹長的區域，避免誤抓塵埃帶或條紋雜訊"),
        "en": ("Cluster Aspect Ratio Threshold", "Excludes overly elongated regions, to avoid mistakenly capturing dust lanes or streak noise"),
    },
    "cluster_dilate": {
        "zh": ("遮罩外擴像素", "星團遮罩外擴的像素數，確保完整覆蓋週邊暗星"),
        "en": ("Cluster Mask Dilation px", "Number of pixels the cluster mask is expanded by, to fully cover surrounding faint stars"),
    },
    "cluster_inpaint_radius": {
        "zh": ("[去星]星團取樣半徑", "去除星團時，從多遠範圍取樣填補背景"),
        "en": ("[Remove] Cluster Inpaint Radius", "How far around each cluster to sample from when filling in the background during removal"),
    },
    "reset_btn": {"zh": "↩️ 重設為預設值", "en": "↩️ Reset to Default Values"},
    "original_preview_image": {"zh": "原圖(未處理)", "en": "Original (Unprocessed)"},
    "preview_image": {"zh": "即時預覽效果(處理後)", "en": "Live Preview (Processed)"},
    "layer_preview_btn": {"zh": "🔄 產生 / 更新圖層預覽", "en": "🔄 Generate / Update Layer Preview"},
    "mask_image": {"zh": "星點遮罩 (Star Mask)", "en": "Star Mask"},
    "starless_image": {"zh": "去星背景層 (Starless)", "en": "Starless Sky Layer"},
    "target_mask_overlay_image": {
        "zh": ("自動局部拉伸偵測區域 (Auto Local Target)", "紅色疊色越明顯代表加強力道越大；黃色線為演算法判定的區域邊界"),
        "en": ("Auto Local Target Detection", "Stronger red overlay = more boost applied; yellow line marks the detected region boundary"),
    },
    "output_dir": {"zh": "輸出資料夾", "en": "Output Directory"},
    "output_name": {"zh": "輸出檔名(不含副檔名)", "en": "Output Filename (without extension)"},
    "save_layers": {"zh": "額外輸出星點遮罩 + 去星背景層(可供後續人工疊圖疊加)", "en": "Export Star Mask and Starless Layers (for manual stacking)"},
    "export_formats": {
        "zh": ("匯出格式", "只需要其中一種格式時可取消勾選，省下匯出時間與硬碟空間"),
        "en": ("Export Formats", "Uncheck a format you don't need to save export time and disk space"),
    },
    "export_btn": {"zh": "🚀 開始高解析度跑圖與匯出", "en": "🚀 Start High-Res Processing & Export"},
    "export_files": {"zh": "下載生成的結果檔案", "en": "Download Generated Results"},
    "preview_hist": {"zh": "RGB 通道分佈曲線", "en": "RGB Channel Histogram"},
    "mask_mini": {"zh": "星點遮罩(產生圖層後自動更新)", "en": "Star Mask (Auto updates after generating layers)"},
    "monitor_refresh_btn": {"zh": "🔄 更新監控資訊", "en": "🔄 Update System Monitor Info"},
    "monitor_auto_refresh": {
        "zh": ("自動刷新(每 3 秒)", "關閉後僅能用左邊按鈕手動刷新，適合想完全避免背景輪詢的情境（如共用電腦、省電）。"),
        "en": ("Auto-refresh (every 3s)", "When off, only the button on the left will refresh the stats — useful when you want to avoid background polling entirely (e.g. shared computers, power saving)."),
    },
    "tab_load": {"zh": "📂 選圖 & 設定", "en": "📂 Image & Config"},
    "tab_param": {"zh": "⚙️ 參數調整", "en": "⚙️ Parameters"},
    "acc_bg": {"zh": "1️⃣ 背景漸層去除(去光害/朦朧)", "en": "1️⃣ Background Gradient Removal"},
    "acc_wb": {"zh": "2️⃣ 色偏校正(白平衡)", "en": "2️⃣ White Balance (Color Cast)"},
    "acc_stretch": {"zh": "3️⃣ 非線性拉伸(拉出暗部細節)", "en": "3️⃣ Non-linear Stretch (Details)"},
    "acc_sat": {"zh": "4️⃣ 飽和度 / 明度 / RGB通道", "en": "4️⃣ Saturation / Brightness / RGB"},
    "acc_clarity": {"zh": "5️⃣ Clarity(局部對比) / 銳化", "en": "5️⃣ Clarity & Sharpening"},
    "acc_denoise": {"zh": "6️⃣ 降噪", "en": "6️⃣ Denoise"},
    "acc_star": {"zh": "7️⃣ 星點縮小 / 去星", "en": "7️⃣ Star Reduction / Removal"},
    "acc_cluster": {"zh": "7️⃣b 大範圍偵測(密集星團/大片暈光)", "en": "7️⃣b Cluster & Halo Detection"},
    "tab_preview": {"zh": "🖼️ 即時預覽", "en": "🖼️ Live Preview"},
    "tab_layer": {"zh": "🔍 遮罩 / 去星層", "en": "🔍 Mask & Starless Layers"},
    "tab_export": {"zh": "💾 匯出全解析度", "en": "💾 Full-Res Export"},
    "acc_hist": {"zh": "📊 Histogram", "en": "📊 Histogram"},
    "acc_mask_mini": {"zh": "⭐ Star Mask 預覽", "en": "⭐ Star Mask Preview"},
    "acc_monitor": {"zh": "🖥️ 系統監控", "en": "🖥️ System Monitor"},
    "cfg_header_md": {"zh": "### 💾 備份與還原參數 (.json)", "en": "### 💾 Backup & Restore Config (.json)"},
    "cfg_download": {"zh": "點擊下載匯出的參數檔案", "en": "Click to download exported config file"},
    "layer_hint_md": {"zh": "⚠️ *縮圖版本，僅供調整參數時參考；勾選「全解析度」後以全圖運算。*", "en": "⚠️ *Thumbnail version, for tuning parameters only; check 'Full Resolution' to process on the full image.*"},
    "focus_btn": {"zh": "⛶ 專注預覽", "en": "⛶ Focus Preview"},
    "compare_mode": {
        "zh": ("原圖對比模式", "「滑桿疊圖」：同一張圖拖曳把手切換前後；「並排顯示」：左右兩張圖分開顯示"),
        "en": ("Compare Mode", "'Slider Overlay': drag handle to compare before/after on same view; 'Side by Side': show both images"),
    },
    "tab_local": {"zh": "⚡ 局部預覽", "en": "⚡ Local Preview"},
    "local_x_pct": {
        "zh": ("X 中心位置 (%)", "裁切區域中心點的水平位置，以整張圖寬度的百分比表示"),
        "en": ("X Center (%)", "Horizontal position of the crop region's center, as a percentage of the full image width"),
    },
    "local_y_pct": {
        "zh": ("Y 中心位置 (%)", "裁切區域中心點的垂直位置，以整張圖高度的百分比表示"),
        "en": ("Y Center (%)", "Vertical position of the crop region's center, as a percentage of the full image height"),
    },
    "local_crop_px": {
        "zh": ("裁切大小 (px)", "裁切區域的邊長，越小處理越快但看到的範圍也越小"),
        "en": ("Crop Size (px)", "Side length of the crop region; smaller = faster processing but a smaller visible area"),
    },
    "local_preview_btn": {"zh": "⚡ 更新局部預覽", "en": "⚡ Update Local Preview"},
    "local_overview_img": {"zh": "裁切位置概覽（紅框=裁切區）", "en": "Crop Location Overview (Red box = crop region)"},
    "local_result_img": {"zh": "局部處理結果（全解析度品質）", "en": "Local Processed Result (Full-Res Quality)"},
    "local_hint_md": {
        "zh": "**局部預覽**：從原圖裁一小塊，以全解析度品質跑完整流程，結果和最終匯出完全一致。\n\n調整下方 X/Y 位置滑桿選取感興趣的區域，點「更新」即可。",
        "en": "**Local Preview**: Crop a small region from the original image and run the full pipeline at 100% resolution. The result matches the final export exactly.\n\nAdjust X/Y sliders to select your region of interest, and click 'Update' to refresh."
    },
    "tab_batch": {"zh": "🗂️ 批次處理", "en": "🗂️ Batch Processing"},
    "batch_hint_md": {
        "zh": "**批次處理**：把目前調好的參數，套用到「📂 選圖 & 設定」分頁中來源資料夾裡的每一張圖片，逐一以全解析度處理並輸出。\n\n⚠️ 建議先用單張圖片調好參數、確認效果滿意後，再執行批次處理。",
        "en": "**Batch Processing**: Applies the current parameter set to every image in the source folder from the '📂 Image & Config' tab, processing and exporting each one at full resolution.\n\n⚠️ Tip: tune parameters on a single image first, then run the batch once you're happy with the result.",
    },
    "batch_out_dir": {"zh": "批次輸出資料夾", "en": "Batch Output Folder"},
    "batch_want_layers": {"zh": "每張圖同時輸出星點遮罩 + 去星背景層", "en": "Also export star mask + starless layer for each image"},
    "batch_formats": {
        "zh": ("匯出格式", "只需要其中一種格式時可取消勾選，批次量大時能省下不少時間與硬碟空間"),
        "en": ("Export Formats", "Uncheck a format you don't need — saves significant time/disk on large batches"),
    },
    "batch_stop_btn": {"zh": "⏹ 停止批次", "en": "⏹ Stop Batch"},
    "batch_btn": {"zh": "🚀 開始批次處理整個資料夾", "en": "🚀 Start Batch Processing"},
    "close_btn": {"zh": "✕ 關閉", "en": "✕ Close"},
    "acc_presets": {"zh": "0️⃣a 🎛️ 新手預設集", "en": "0️⃣a 🎛️ Beginner Presets"},
    "presets_hint_md": {
        "zh": "先選一個接近你拍攝情境的起點，再自行微調。",
        "en": "Pick a starting point close to your shooting scenario, then fine-tune from there.",
    },
    "preset_milky_way_btn": {"zh": "🌌 銀河模式", "en": "🌌 Milky Way"},
    "preset_nebula_btn": {"zh": "🌫️ 星雲模式", "en": "🌫️ Nebula"},
    "preset_light_pollution_btn": {"zh": "🏙️ 重光害", "en": "🏙️ Heavy Light Pollution"},
    "cfg_export_name": {
        "zh": ("匯出檔名（選填）", "例如：後院光害設定（留空則用預設檔名）"),
        "en": ("Export Filename (optional)", "e.g. 'Backyard Light Pollution' (leave blank to use the default filename)"),
    },
    "acc_snapshots": {"zh": "0️⃣b 📌 參數快照", "en": "0️⃣b 📌 Parameter Snapshots"},
    "snapshot_hint_md": {
        "zh": "先「儲存」目前這組參數到 A/B/C，之後可以隨時「套用」快速切回比較，不用重新調整滑桿。快照只暫存在這次瀏覽器工作階段中，重新整理頁面會清空。若想長期保留，可在下方為快照命名並「存成檔案」，之後隨時能「從檔案載入」回 A/B/C。",
        "en": "'Save' the current parameter set to slot A/B/C, then 'Apply' any time to switch back instantly for comparison — no need to re-adjust sliders. Snapshots only live in this browser session and are cleared on page refresh. To keep one long-term, name it below and 'Save to File' — you can 'Load from File' back into A/B/C any time.",
    },
    "snap_a_save_btn": {"zh": "💾 儲存為 A", "en": "💾 Save as A"},
    "snap_a_load_btn": {"zh": "📥 套用 A", "en": "📥 Apply A"},
    "snap_b_save_btn": {"zh": "💾 儲存為 B", "en": "💾 Save as B"},
    "snap_b_load_btn": {"zh": "📥 套用 B", "en": "📥 Apply B"},
    "snap_c_save_btn": {"zh": "💾 儲存為 C", "en": "💾 Save as C"},
    "snap_c_load_btn": {"zh": "📥 套用 C", "en": "📥 Apply C"},
    "snap_file_hint_md": {
        "zh": "**💾 快照命名與存檔**：選擇上面 A/B/C 其中一格，取個名字後「存成檔案」，或「從檔案載入」回填到指定的 A/B/C（載入後仍需按「套用」才會實際套用到滑桿）。",
        "en": "**💾 Named Snapshot Save/Load**: Pick one of the A/B/C slots above, give it a name, and 'Save to File' — or 'Load from File' back into a chosen slot (still needs 'Apply' afterward to actually update the sliders).",
    },
    "snap_file_target": {"zh": "目標快照", "en": "Target Slot"},
    "snap_file_name": {
        "zh": ("快照名稱（選填）", "例如：後院光害"),
        "en": ("Snapshot Name (optional)", "e.g. 'Backyard Light Pollution'"),
    },
    "snap_save_file_btn": {"zh": "💾 快照存成檔案", "en": "💾 Save Snapshot to File"},
    "snap_load_file": {"zh": "📂 從檔案載入快照", "en": "📂 Load Snapshot from File"},
    "snap_file_download": {"zh": "點擊下載已存檔的快照", "en": "Click to download the saved snapshot file"},
}


# 以下是「初次載入畫面時」的預設佔位訊息（尚未做任何操作前顯示）。
# 因為這些欄位平常會被 load_image_fn / export_config_fn / update_preview_fn 等函式
# 覆寫成真正的處理結果訊息，所以切換語言時「不能」無條件覆寫——否則會把使用者
# 已經看到的真實狀態（例如「已載入 xxx.tif」）洗成空白佔位字，而是要先比對目前顯示
# 的文字是否仍然是「初始佔位字」，是的話才跟著語言切換，其餘一律保留原文字。
PLACEHOLDER_TEXTS = {
    "load_status":         {"zh": "尚未載入圖片",       "en": "No image loaded yet"},
    "cfg_status":          {"zh": "尚未執行匯入/匯出",   "en": "No import/export performed yet"},
    "preview_status":      {"zh": "等待圖片載入...",     "en": "Waiting for image to load..."},
    "compare_slider_html": {
        "zh": "<div style='color:#888;padding:40px;text-align:center;'>⚠️ 尚未載入圖片</div>",
        "en": "<div style='color:#888;padding:40px;text-align:center;'>⚠️ No image loaded yet</div>",
    },
    "snap_a_status": {"zh": "🔲 快照 A：尚未儲存", "en": "🔲 Snapshot A: not saved yet"},
    "snap_b_status": {"zh": "🔲 快照 B：尚未儲存", "en": "🔲 Snapshot B: not saved yet"},
    "snap_c_status": {"zh": "🔲 快照 C：尚未儲存", "en": "🔲 Snapshot C: not saved yet"},
}

# ============================================================
# ===================== 系統監控輔助函式 ========================
# ============================================================

try:
    import psutil as _psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    _psutil = None

try:
    import GPUtil as _GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False
    _GPUtil = None


def _bar(pct, cls=""):
    """產生一個帶百分比填滿的 HTML 進度條。"""
    return (
        f'<div class="mon-bar-wrap">'
        f'<div class="mon-bar {cls}" style="width:{max(2, min(100, pct)):.0f}%"></div>'
        f'</div>'
    )


def get_system_stats_html():
    """回傳系統 RAM / CPU / GPU / VRAM 監控的 HTML 片段。"""
    rows = []

    if HAS_PSUTIL:
        mem = _psutil.virtual_memory()
        ram_used  = mem.used  / (1024 ** 3)
        ram_total = mem.total / (1024 ** 3)
        cpu_pct   = _psutil.cpu_percent(interval=None)
        rows.append(
            f'<div class="mon-row">'
            f'<span class="mon-label">RAM</span>'
            f'{_bar(mem.percent)}'
            f'<span class="mon-val">{ram_used:.1f}/{ram_total:.1f} GB</span>'
            f'</div>'
        )
        rows.append(
            f'<div class="mon-row">'
            f'<span class="mon-label">CPU</span>'
            f'{_bar(cpu_pct, "cpu-bar")}'
            f'<span class="mon-val">{cpu_pct:.0f}%</span>'
            f'</div>'
        )
    else:
        rows.append("<div class='mon-hint'>🔧 pip install psutil</div>")

    # v1.2.1 主線 B：GPU 區塊重寫，讓 DirectML(AMD/Intel) 使用者至少看到一句解釋，
    # 而不是完全空白（原本 HAS_GPUTIL 為 True 但 getGPUs() 找不到 NVIDIA 顯卡時，
    # 迴圈跑零次、什麼提示都不會顯示，看起來像程式壞掉）。
    gpu_shown = False
    if HAS_GPUTIL:
        try:
            gpus = _GPUtil.getGPUs()
        except Exception:
            gpus = []
        for gpu in gpus:
            gpu_pct   = gpu.load * 100
            vram_used  = gpu.memoryUsed  / 1024
            vram_total = gpu.memoryTotal / 1024
            vram_pct   = (gpu.memoryUsed / max(gpu.memoryTotal, 1)) * 100
            rows.append(
                f'<div class="mon-row">'
                f'<span class="mon-label">GPU</span>'
                f'{_bar(gpu_pct, "gpu-bar")}'
                f'<span class="mon-val">{gpu_pct:.0f}%</span>'
                f'</div>'
            )
            rows.append(
                f'<div class="mon-row">'
                f'<span class="mon-label">VRAM</span>'
                f'{_bar(vram_pct, "vram-bar")}'
                f'<span class="mon-val">{vram_used:.1f}/{vram_total:.1f} GB</span>'
                f'</div>'
            )
            gpu_shown = True

    if not gpu_shown and USE_GPU and _IS_DIRECTML:
        # GPUtil 底層包 nvidia-smi，只找得到 NVIDIA；DirectML(AMD/Intel/Windows GPU)
        # 目前沒有跨廠牌的即時使用率/VRAM 讀取方式，先給一句誠實的說明取代空白區塊。
        rows.append(
            "<div class='mon-hint'>⚡ DirectML 加速中（此廠牌暫無法讀取即時使用率/VRAM）</div>"
        )
        gpu_shown = True

    if not gpu_shown and USE_GPU and HAS_TORCH:
        try:
            import torch
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / (1024 ** 3)
                total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                pct = alloc / max(total_mem, 1e-6) * 100
                rows.append(
                    f'<div class="mon-row">'
                    f'<span class="mon-label">VRAM</span>'
                    f'{_bar(pct, "vram-bar")}'
                    f'<span class="mon-val">{alloc:.1f}/{total_mem:.1f} GB</span>'
                    f'</div>'
                )
                gpu_shown = True
        except Exception:
            pass

    if not gpu_shown:
        rows.append("<div class='mon-hint'>🔧 pip install GPUtil</div>")

    return f'<div class="mon-panel">{"".join(rows)}</div>'


def get_status_bar_html(proc_time=None, lang="zh"):
    """回傳底部狀態列 HTML。"""
    if isinstance(proc_time, str) and proc_time in ["zh", "en"]:
        lang = proc_time
        proc_time = None

    time_str = f"{proc_time:.2f}s" if proc_time is not None else "--"
    ram_str, gpu_str = "N/A", "N/A"

    if HAS_PSUTIL:
        try:
            mem = _psutil.virtual_memory()
            ram_str = f"{mem.used / (1024**3):.1f} GB"
        except Exception:
            pass

    # v1.3.2：補上 get_system_stats_html() 在 v1.2.1 就有的 DirectML fallback —
    # 之前這裡 GPUtil 找不到 NVIDIA 顯卡時會直接落到 "N/A"，AMD/Intel 使用者
    # 在狀態列完全看不出 GPU 有沒有在跑，跟監控面板的行為不一致。
    gpu_shown = False
    if HAS_GPUTIL:
        try:
            gpus = _GPUtil.getGPUs()
        except Exception:
            gpus = []
        if gpus:
            g = gpus[0]
            gpu_str = f"{g.load * 100:.0f}% | {g.memoryUsed / 1024:.1f} GB"
            gpu_shown = True

    if not gpu_shown and USE_GPU and _IS_DIRECTML:
        gpu_str = "DirectML 加速中" if lang == "zh" else "DirectML active"
        gpu_shown = True

    if not gpu_shown and USE_GPU and HAS_TORCH:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_str = f"{torch.cuda.memory_allocated() / (1024**3):.1f} GB alloc"
                gpu_shown = True
        except Exception:
            pass

    t_lbl = "⏱ 時間:" if lang == "zh" else "⏱ Time:"
    r_lbl = "💾 記憶體:" if lang == "zh" else "💾 RAM:"
    g_lbl = "🖥️ 顯卡:" if lang == "zh" else "🖥️ GPU:"

    return (
        f'<div id="status-bar">'
        f'<div class="st-dot"></div>'
        f'<div class="st-item">{t_lbl} <b>{time_str}</b></div>'
        f'<div class="st-item">{r_lbl} <b>{ram_str}</b></div>'
        f'<div class="st-item">{g_lbl} <b>{gpu_str}</b></div>'
        f'<div class="st-backend">⚡ {_TORCH_BACKEND_NAME}</div>'
        f'</div>'
    )


# ============================================================
# ========================= PRO CSS ==========================
# ============================================================

PRO_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Design Tokens ── */
html {
    --bg-main:    #090913;
    --bg-panel:   #0e0e1c;
    --bg-card:    #13132a;
    --bg-hover:   #1a1a38;
    --accent:     #7c6fff;
    --accent2:    #00d9ff;
    --accent-dim: rgba(124,111,255,0.14);
    --txt:        #ddddf0;
    --txt-sub:    #7878a8;
    --txt-dim:    #38384e;
    --border:     #18182e;
    --border-lt:  #24243e;
    --success:    #00cc96;
    --warn:       #ffc857;
    --bar-bg:     #1c1c38;
    --radius:     10px;
}

html.light-theme {
    --bg-main:    #f0f0f8;
    --bg-panel:   #ffffff;
    --bg-card:    #ebebf8;
    --bg-hover:   #dcdcfa;
    --accent:     #5c4fff;
    --accent2:    #00a8cc;
    --accent-dim: rgba(92,79,255,0.14);
    --txt:        #101026;
    --txt-sub:    #50507a;
    --txt-dim:    #9090b0;
    --border:     #d0d0e5;
    --border-lt:  #b5b5d8;
    --success:    #00a378;
    --warn:       #e0a520;
    --bar-bg:     #d5d5ee;
}

body, .gradio-container, html {
    background-color: var(--bg-main) !important;
    color: var(--txt) !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    transition: background-color 0.25s, color 0.25s;
}

.gradio-container {
    --background-fill-primary: var(--bg-main) !important;
    --background-fill-secondary: var(--bg-panel) !important;
    --block-background-fill: var(--bg-card) !important;
    --block-border-color: var(--border) !important;
    --border-color-primary: var(--border) !important;
    --border-color-secondary: var(--border-lt) !important;
    --body-text-color: var(--txt) !important;
    --body-text-color-subdued: var(--txt-sub) !important;
    --button-secondary-background-fill: var(--bg-card) !important;
    --button-secondary-background-fill-hover: var(--bg-hover) !important;
    --button-secondary-border-color: var(--border-lt) !important;
    --button-secondary-text-color: var(--txt) !important;
    --block-label-text-color: var(--txt-sub) !important;
    --input-background-fill: var(--bg-panel) !important;
    --input-border-color: var(--border) !important;
    --slider-color: var(--accent) !important;
}

/* ── Toolbar (Gradio Row) ── */
#pro-toolbar-row {
    align-items: center !important;
    gap: 8px !important;
    padding: 8px 20px !important;
    background: linear-gradient(135deg, var(--bg-panel) 0%, var(--bg-hover) 100%) !important;
    border-bottom: 1px solid var(--border-lt) !important;
    border-radius: var(--radius) var(--radius) 0 0 !important;
    flex-wrap: nowrap !important;
}

/* Title stretches to fill all remaining space → controls stay on right */
#pro-title {
    flex: 1 1 auto !important;
    font-size: 19px;
    font-weight: 700;
    background: linear-gradient(135deg, #a89eff 0%, #00d9ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.3px;
    white-space: nowrap;
    padding: 4px 0;
}

/* The first child wrapper (HTML with title) should stretch */
#pro-toolbar-row > div:first-child {
    flex: 1 1 auto !important;
    min-width: 0 !important;
}

/* Every other Gradio wrapper in the toolbar should NOT grow */
#pro-toolbar-row > div:not(:first-child) {
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: 0 !important;
}

/* Language radio in toolbar */
.toolbar-radio {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    gap: 0 !important;
    flex: 0 0 auto !important;
    width: fit-content !important;
    min-width: 0 !important;
}

.toolbar-radio .wrap {
    gap: 0 !important;
    display: flex !important;
    flex-direction: row !important;
    border: 1px solid var(--border-lt) !important;
    border-radius: 6px !important;
    overflow: hidden !important;
    width: fit-content !important;
}

.toolbar-radio input[type="radio"] {
    display: none !important;
}

.toolbar-radio .wrap label {
    background: transparent;
    border: none !important;
    padding: 4px 14px !important;
    font-size: 12px !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--txt-sub) !important;
    cursor: pointer;
    transition: all 0.18s;
    letter-spacing: 0.2px;
    margin: 0 !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    flex: 0 0 auto !important;
    white-space: nowrap !important;
    display: flex !important;
    align-items: center !important;
}

.toolbar-radio .wrap label:hover {
    background: var(--accent-dim) !important;
    color: var(--accent) !important;
}

.toolbar-radio .wrap label.selected,
.toolbar-radio .wrap label:has(input:checked),
.toolbar-radio .wrap label:has(+ input:checked),
.toolbar-radio .wrap span[data-testid="radio-label"].selected {
    background: var(--accent) !important;
    color: #fff !important;
}

/* Gradio 5+ uses this pattern for selected radio */
.toolbar-radio .wrap > label.svelte-1aq2jjk.selected,
.toolbar-radio .wrap > label[aria-checked="true"] {
    background: var(--accent) !important;
    color: #fff !important;
}

/* Theme checkbox in toolbar */
.toolbar-checkbox {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-lt) !important;
    border-radius: 6px !important;
    padding: 0 !important;
    flex: 0 0 auto !important;
    width: fit-content !important;
    min-width: 0 !important;
}

.toolbar-checkbox label {
    padding: 4px 12px !important;
    font-size: 12px !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--txt-sub) !important;
    cursor: pointer !important;
    white-space: nowrap !important;
    display: flex !important;
    align-items: center !important;
    gap: 4px !important;
}

.toolbar-checkbox input[type="checkbox"] {
    display: none !important;
}

.toolbar-checkbox:hover {
    border-color: var(--warn) !important;
}

.toolbar-checkbox:hover label {
    color: var(--warn) !important;
}

/* GPU badge */
#gpu-badge {
    background: linear-gradient(135deg,
        rgba(124,111,255,0.12) 0%, rgba(0,217,255,0.08) 100%);
    border: 1px solid var(--border-lt);
    border-radius: 6px;
    padding: 5px 12px;
    font-size: 11px;
    color: var(--accent2);
    font-family: monospace;
    letter-spacing: 0.2px;
    white-space: nowrap;
    max-width: 320px;
    overflow: hidden;
    text-overflow: ellipsis;
    flex: 0 0 auto;
}

/* ── Main 3-column Layout ── */
#main-row {
    align-items: stretch !important;
    gap: 0 !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-bottom: none !important;
    min-height: 76vh;
}

#left-col {
    max-height: 80vh;
    overflow-y: auto;
    border-right: 1px solid var(--border-lt) !important;
    background: var(--bg-panel) !important;
    padding: 8px 6px 8px 8px !important;
}

#center-col {
    max-height: 80vh;
    overflow-y: auto;
    background: var(--bg-main) !important;
    padding: 10px !important;
}

#right-col {
    max-height: 80vh;
    overflow-y: auto;
    border-left: 1px solid var(--border-lt) !important;
    background: var(--bg-panel) !important;
    padding: 8px 8px 8px 6px !important;
}

/* Thin scrollbar */
#left-col::-webkit-scrollbar,
#right-col::-webkit-scrollbar,
#center-col::-webkit-scrollbar { width: 3px; }
#left-col::-webkit-scrollbar-thumb,
#right-col::-webkit-scrollbar-thumb,
#center-col::-webkit-scrollbar-thumb {
    background: var(--border-lt); border-radius: 2px;
}

/* ── Status Bar ── */
#status-bar {
    display: flex;
    align-items: center;
    gap: 20px;
    padding: 8px 20px;
    background: linear-gradient(135deg, var(--bg-panel) 0%, var(--bg-hover) 100%);
    border-top: 1px solid var(--border-lt);
    border-radius: 0 0 var(--radius) var(--radius);
    font-size: 12px;
    color: var(--txt-sub);
    font-family: 'Inter', monospace;
}

.st-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--accent2);
    box-shadow: 0 0 7px var(--accent2);
    animation: pulse-dot 2.5s ease-in-out infinite;
    flex-shrink: 0;
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; box-shadow: 0 0 7px var(--accent2); }
    50%       { opacity: 0.3; box-shadow: none; }
}

.st-item { display: flex; align-items: center; gap: 5px; }
.st-item b { color: var(--txt); }
.st-backend { margin-left: auto; font-size: 11px; color: var(--txt-dim); }

/* ── Monitor Panel ── */
.mon-panel { padding: 4px 0; }

.mon-row {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 9px;
}

.mon-label {
    width: 36px;
    color: var(--txt-sub);
    font-size: 10px;
    font-family: monospace;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    flex-shrink: 0;
}

.mon-bar-wrap {
    flex: 1;
    height: 5px;
    background: var(--bar-bg);
    border-radius: 3px;
    overflow: hidden;
}

.mon-bar {
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    border-radius: 3px;
    transition: width 0.6s cubic-bezier(.4,0,.2,1);
    min-width: 2px;
}

.cpu-bar  { background: linear-gradient(90deg, #00cc96, #00d9ff); }
.gpu-bar  { background: linear-gradient(90deg, #7c6fff, #cc6fff); }
.vram-bar { background: linear-gradient(90deg, #ffc857, #ff8c57); }

.mon-val {
    color: var(--txt);
    font-size: 11px;
    font-family: monospace;
    min-width: 88px;
    text-align: right;
    flex-shrink: 0;
}

.mon-hint {
    color: var(--txt-dim);
    font-size: 11px;
    font-style: italic;
    padding: 6px 0;
    text-align: center;
}

/* Focus mode button */
#focus-btn {
    background: transparent !important;
    border: 1px solid var(--border-lt) !important;
    border-radius: 6px !important;
    padding: 4px 12px !important;
    font-size: 12px !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--txt-sub) !important;
    cursor: pointer !important;
    transition: all 0.18s !important;
    flex-shrink: 0 !important;
    white-space: nowrap !important;
    min-width: auto !important;
    box-shadow: none !important;
}

#focus-btn:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    box-shadow: 0 0 8px rgba(124,111,255,0.3) !important;
}

#close-btn {
    background: transparent !important;
    border: 1px solid var(--border-lt) !important;
    border-radius: 6px !important;
    padding: 4px 12px !important;
    font-size: 12px !important;
    font-family: 'Inter', sans-serif !important;
    color: #ff6b6b !important;
    cursor: pointer !important;
    transition: all 0.18s !important;
    flex-shrink: 0 !important;
    white-space: nowrap !important;
    min-width: auto !important;
    box-shadow: none !important;
}

#close-btn:hover {
    border-color: #ff4d4d !important;
    color: #ffffff !important;
    background: #ff4d4d !important;
    box-shadow: 0 0 8px rgba(255,77,77,0.4) !important;
}
"""

# ============================================================
# ========================= Gradio UI =========================
# ============================================================

custom_js = """
() => {
    /* Force Gradio default dark mode wrapper on initial load */
    var html = document.querySelector('html');
    var body = document.querySelector('body');
    if (html && !html.classList.contains('light-theme')) {
        html.classList.add('dark');
    }
    if (body && !body.classList.contains('light-theme')) {
        body.classList.add('dark');
    }
}
"""

def _sharpen_mode_visibility(mode_val):
    """v1.3.3 追加：mode == "internal" 時隱藏 external 三欄；mode == "external"
    時反過來隱藏 internal 用的 sharpen_blur/sharpen_amount 兩顆滑桿——兩組欄位
    互斥且沒有交叉依賴（sharpen 只有這兩種模式，internal 滑桿只在
    apply_clarity_and_sharpen() 的 internal 分支被讀取，external 分支完全不會
    用到），可以放心兩邊都隱藏。
    """
    is_ext = (mode_val == "external")
    return (
        gr.update(visible=not is_ext),  # sharpen_blur
        gr.update(visible=not is_ext),  # sharpen_amount
        gr.update(visible=is_ext),      # sharpen_ext_path
        gr.update(visible=is_ext),      # sharpen_ext_profile
        gr.update(visible=is_ext),      # sharpen_ext_args
    )


def _denoise_mode_visibility(mode_val):
    """v1.3.3 追加：denoise() 讀過原始碼確認過 d/sigma_color/sigma_space 只在
    mode=="fast" 分支用到、nlm_h/nlm_h_color 只在 mode=="quality" 分支用到，
    彼此不共用、也沒有其他函式跨模式讀取這幾個欄位，三組（fast/quality/
    external）可以互斥顯示，不會藏到其實還有作用的欄位。
    """
    return (
        gr.update(visible=mode_val == "fast"),      # denoise_d
        gr.update(visible=mode_val == "fast"),      # denoise_sigma_color
        gr.update(visible=mode_val == "fast"),      # denoise_sigma_space
        gr.update(visible=mode_val == "quality"),   # denoise_nlm_h
        gr.update(visible=mode_val == "quality"),   # denoise_nlm_h_color
        gr.update(visible=mode_val == "external"),  # denoise_ext_path
        gr.update(visible=mode_val == "external"),  # denoise_ext_profile
        gr.update(visible=mode_val == "external"),  # denoise_ext_args
    )


def _star_mode_visibility(mode_val):
    """v1.3.3 追加：只隱藏 star_shrink_kernel/star_shrink_iter/star_shrink_strength
    這三個「[縮星]」欄位（`process_stars()` 讀過確認只有 mode=="shrink" 分支
    會用到，其他分支完全不讀），mode!="shrink" 時收起來。

    刻意*不*對星點偵測欄位（star_kernel/star_thresh/...）和「[去星]」欄位
    （star_inpaint_radius/star_feather_px/star_noise_strength）做同樣的事，
    即使乍看也是「只有 remove 模式才用得到」：讀過 finish_pipeline() 才發現
    `need_mask = star_mode != 'none' or want_layers`，且只要 want_layers=True
    （按「產生圖層」預覽、或匯出/批次勾選「同時輸出去星圖層」），不管主畫面
    選的是 shrink／none／external，都會強制跑一次完整的多尺度去星去產生
    starless 圖層——這時候用的正是偵測欄位跟這三個「[去星]」欄位。如果照
    mode 隱藏，會讓使用者在 shrink 模式下調不到其實仍在運作的圖層去星參數，
    比目前「全部顯示」更容易誤導，所以維持不動。
    """
    show_shrink = (mode_val == "shrink")
    show_ext = (mode_val == "external")
    return (
        gr.update(visible=show_shrink),  # star_shrink_kernel
        gr.update(visible=show_shrink),  # star_shrink_iter
        gr.update(visible=show_shrink),  # star_shrink_strength
        gr.update(visible=show_ext),     # star_ext_path
        gr.update(visible=show_ext),     # star_ext_profile
        gr.update(visible=show_ext),     # star_ext_args
    )


with gr.Blocks(
    title="🌌 Astro Processor Pro",
) as demo:

    # ── Global state ──────────────────────────────────────────
    state_full          = gr.State(None)
    state_preview_base  = gr.State(None)
    state_preview_scale = gr.State(1.0)
    state_lang          = gr.State("zh")
    focus_mode          = gr.State(False)
    state_snap_a        = gr.State(None)
    state_snap_b        = gr.State(None)
    state_snap_c        = gr.State(None)
    # v1.3.2：批次處理停止旗標。刻意用「內容可變的 dict」而不是單純的 bool，
    # 是因為 batch_process_fn() 在批次開始當下就把這個物件當一般參數收下、
    # 存成區域變數，之後不會再重新跟 Gradio session 要一次新的值；如果用不可變的
    # bool，「停止」按鈕之後才寫回 session 的新值，正在跑的那個 generator 根本看
    # 不到。改成傳同一個 dict 物件、用「原地修改」（stop_state["stop"] = True）
    # 而不是整個替換掉，批次迴圈裡每次檢查的才會是同一份、被按鈕即時改過的資料。
    state_batch_stop    = gr.State({"stop": False})

    # v1.3.3 Track B：狀態列「⏱ 時間」欄位過去從未被接上過任何實際數值
    # （get_status_bar_html 的 proc_time 參數一直收到 None）。決定讓狀態列
    # 反映「最近一次匯出」的耗時——export_fn 已經算過 elapsed，只是沒有把
    # 原始 float 一併回傳；現在額外用這個 State 存起來，load_btn / export_btn /
    # lang_radio 三個既有呼叫點都改成把它一起傳給 get_status_bar_html()，
    # 這樣不管哪個觸發，狀態列顯示的都是同一個「最近一次匯出耗時多久」。
    state_last_export_time = gr.State(None)

    # ── Top Toolbar (Gradio Row) ──────────────────────────────
    with gr.Row(elem_id="pro-toolbar-row", equal_height=True):
        gr.HTML(f'<div id="pro-title">🌌 Astro Processor Pro</div>')
        lang_radio = gr.Radio(
            choices=["中文", "EN"],
            value="中文",
            label="",
            show_label=False,
            interactive=True,
            elem_id="lang-radio",
            elem_classes=["toolbar-radio"],
            min_width=0,
        )
        theme_checkbox = gr.Checkbox(
            value=False,
            label="☀ Light Mode",
            interactive=True,
            elem_id="theme-checkbox",
            elem_classes=["toolbar-checkbox"],
            min_width=0,
        )
        focus_btn = gr.Button(
            "⛶ 專注預覽",
            elem_id="focus-btn",
            elem_classes=["toolbar-focus-btn"],
            size="sm",
            min_width=0,
        )
        gr.HTML(f'<div id="gpu-badge">⚡ {_TORCH_BACKEND_NAME}</div>')
        close_btn = gr.Button(
            "✕ 關閉",
            elem_id="close-btn",
            elem_classes=["toolbar-close-btn"],
            size="sm",
            min_width=0,
        )

    # ── 3-Column Main Layout ──────────────────────────────────
    with gr.Row(elem_id="main-row"):

        # ════════════════ LEFT COLUMN — Parameters ════════════
        with gr.Column(scale=1, elem_id="left-col") as left_col:
            with gr.Tabs():

                # ── Tab: Load & Config ────────────────────────
                with gr.Tab("📂 選圖 & 設定") as tab_load:
                    with gr.Group():
                        folder_box = gr.Textbox(
                            label="本機資料夾路徑", value=".",
                            placeholder="例如 D:/astro_photos"
                        )
                        with gr.Row():
                            scan_btn = gr.Button("🔍 掃描資料夾", size="sm")
                            file_dropdown = gr.Dropdown(
                                label="資料夾內圖片", choices=[], interactive=True
                            )
                        upload_file = gr.File(
                            label="或直接上傳圖片 (tif/jpg/png/RAW)",
                            file_count="single", type="filepath"
                        )
                        preview_size = gr.Slider(
                            400, 2000, value=900, step=50,
                            label="預覽解析度(長邊像素)",
                            info="縮圖越小處理越快，不影響最終匯出解析度"
                        )
                        use_full_res_preview = gr.Checkbox(
                            label="即時預覽改用原圖全解析度運算(較慢但最準確)",
                            value=False,
                            info="未勾選時使用縮圖運算，速度較快；勾選後直接用原圖跑全部流程"
                        )
                        load_btn = gr.Button("📥 載入圖片", variant="primary")
                        load_status = gr.Markdown("尚未載入圖片")

                    gr.Markdown("---")
                    with gr.Group():
                        cfg_header_md = gr.Markdown("### 💾 備份與還原參數 (.json)")
                        cfg_export_name = gr.Textbox(
                            label="匯出檔名（選填）", placeholder="例如：後院光害設定（留空則用預設檔名）",
                        )
                        with gr.Row():
                            cfg_export_btn = gr.Button("📤 匯出當前參數", size="sm")
                            cfg_import_file = gr.File(
                                label="匯入參數 JSON",
                                file_count="single", type="filepath"
                            )
                        cfg_status   = gr.Markdown("尚未執行匯入/匯出")
                        cfg_download = gr.File(label="點擊下載匯出的參數檔案")

                # ── Tab: All Parameters ───────────────────────
                with gr.Tab("⚙️ 參數調整") as tab_param:

                    # ── Presets：給新手的起始參數 ──────────────
                    with gr.Accordion("0️⃣a 🎛️ 新手預設集", open=False, elem_id="presets-group") as acc_presets:
                        presets_hint_md = gr.Markdown("先選一個接近你拍攝情境的起點，再自行微調。")
                        with gr.Row():
                            preset_milky_way_btn = gr.Button("🌌 銀河模式", size="sm")
                            preset_nebula_btn    = gr.Button("🌫️ 星雲模式", size="sm")
                            preset_light_pollution_btn = gr.Button("🏙️ 重光害", size="sm")
                        preset_status = gr.Markdown("")

                    # ── 參數快照：A / B / C 三組快速比較 ─────────
                    with gr.Accordion("0️⃣b 📌 參數快照", open=False, elem_id="snapshots-group") as acc_snapshots:
                        snapshot_hint_md = gr.Markdown(
                            "先「儲存」目前這組參數到 A/B/C，之後可以隨時「套用」快速切回比較，"
                            "不用重新調整滑桿。快照只暫存在這次瀏覽器工作階段中，重新整理頁面會清空。"
                            "若想長期保留，可在下方為快照命名並「存成檔案」，之後隨時能「從檔案載入」回 A/B/C。"
                        )
                        with gr.Row():
                            with gr.Column(min_width=0):
                                snap_a_save_btn = gr.Button("💾 儲存為 A", size="sm")
                                snap_a_load_btn = gr.Button("📥 套用 A", size="sm")
                                snap_a_status = gr.Markdown("🔲 快照 A：尚未儲存")
                            with gr.Column(min_width=0):
                                snap_b_save_btn = gr.Button("💾 儲存為 B", size="sm")
                                snap_b_load_btn = gr.Button("📥 套用 B", size="sm")
                                snap_b_status = gr.Markdown("🔲 快照 B：尚未儲存")
                            with gr.Column(min_width=0):
                                snap_c_save_btn = gr.Button("💾 儲存為 C", size="sm")
                                snap_c_load_btn = gr.Button("📥 套用 C", size="sm")
                                snap_c_status = gr.Markdown("🔲 快照 C：尚未儲存")

                        gr.Markdown("---")
                        snap_file_hint_md = gr.Markdown(
                            "**💾 快照命名與存檔**：選擇上面 A/B/C 其中一格，取個名字後「存成檔案」，"
                            "或「從檔案載入」回填到指定的 A/B/C（載入後仍需按「套用」才會實際套用到滑桿）。"
                        )
                        with gr.Row():
                            snap_file_target = gr.Radio(["A", "B", "C"], value="A", label="目標快照", min_width=0)
                            snap_file_name = gr.Textbox(label="快照名稱（選填）", placeholder="例如：後院光害", min_width=0)
                        with gr.Row():
                            snap_save_file_btn = gr.Button("💾 快照存成檔案", size="sm")
                            snap_load_file = gr.File(label="📂 從檔案載入快照", file_count="single", type="filepath")
                        snap_file_status = gr.Markdown("")
                        snap_file_download = gr.File(label="點擊下載已存檔的快照")

                    with gr.Group():

                        with gr.Accordion("1️⃣ 背景漸層去除(去光害/朦朧)", open=False) as acc_bg:
                            bg_enable     = gr.Checkbox(label="啟用背景漸層去除", value=DEFAULTS[0])
                            bg_downscale  = gr.Slider(0.02, 0.2,  value=0.06, step=0.01,  label="估算縮圖比例", info="越小越快，但過小會失真")
                            bg_min_filter = gr.Slider(3,    25,   value=9,    step=2,     label="局部最暗值視窗", info="星雲核心較大時應加大，避免誤判為背景")
                            bg_blur_sigma = gr.Slider(1,    20,   value=6,    step=0.5,   label="背景平滑程度", info="越高背景漸層過渡越平滑")
                            bg_subtract   = gr.Slider(0,    1,    value=0.92, step=0.01,  label="扣除強度", info="1.0 完全扣除，0.90-0.95 能保留自然天光")

                        with gr.Accordion("2️⃣ 色偏校正(白平衡)", open=False) as acc_wb:
                            wb_enable = gr.Checkbox(label="啟用色偏校正(白平衡)", value=DEFAULTS[5])
                            wb_min    = gr.Slider(0.1, 1, value=0.6, step=0.05, label="增益下限", info="限制最大縮小幅度，防顏色死掉")
                            wb_max    = gr.Slider(1,   3, value=1.8, step=0.05, label="增益上限", info="限制最大放大倍率，防特定通道雜訊爆發")

                        with gr.Accordion("3️⃣ 非線性拉伸(拉出暗部細節)", open=True) as acc_stretch:
                            black_pct      = gr.Slider(0,  5,   value=0.2,  step=0.1,  label="黑點百分位", info="通常設 0.1-0.5%")
                            stretch_factor = gr.Slider(1,  200, value=12,   step=0.5,  label="拉伸強度(arcsinh)", info="越高微弱星雲越明顯")
                            white_pct      = gr.Slider(90, 100, value=99.7, step=0.1,  label="白點百分位", info="99.7% 代表最亮前 0.3% 飽和成純白")

                        with gr.Accordion("3️⃣b 自動局部拉伸(星雲/銀河區域)", open=False) as acc_local_target:
                            local_target_enable      = gr.Checkbox(label="啟用自動局部拉伸", value=DEFAULTS[11], info="自動抓出有結構的星雲/銀河區域，只加強該處對比，天空背景不受影響")
                            local_target_strength    = gr.Slider(0,   2,   value=0.6,  step=0.05, label="局部拉伸強度", info="0 代表關閉效果")
                            local_target_radius      = gr.Slider(10,  120, value=40,   step=5,    label="偵測/加強半徑(px)", info="約略對應星雲結構的尺度，太小會連星點都吃到")
                            local_target_sensitivity = gr.Slider(0.3, 3,   value=1.0,  step=0.1,  label="偵測靈敏度", info="越高，越多區域會被判定為『目標』而套用加強")

                            gr.Markdown("---")
                            manual_target_hint_md = gr.Markdown(
                                "**手動框選加強區域**：自動偵測有時會漏掉較弱的目標、或誤判雜訊區。"
                                "啟用後可框選一塊區域（矩形或橢圓/圓形），不論自動偵測結果如何，該區域一律套用上面的局部拉伸強度加強。"
                                "框選區域會和自動遮罩取聯集，可在右側自動局部拉伸遮罩預覽圖中確認實際涵蓋範圍。"
                            )
                            manual_target_enable = gr.Checkbox(
                                label="啟用手動框選區域", value=DEFAULTS[53],
                                info="即使自動局部拉伸沒有偵測到任何區域，仍可只用手動框選的區域加強"
                            )
                            manual_target_shape = gr.Radio(
                                ["rectangle", "ellipse"], value="rectangle",
                                label="框選形狀", info="矩形，或橢圓/正圓（寬高百分比相等時即為正圓）"
                            )
                            with gr.Row():
                                manual_target_x_pct = gr.Slider(0, 100, value=50, step=0.5,
                                                                 label="X 中心位置 (%)", info="框選區域中心點的水平位置，以整張圖寬度的百分比表示")
                                manual_target_y_pct = gr.Slider(0, 100, value=50, step=0.5,
                                                                 label="Y 中心位置 (%)", info="框選區域中心點的垂直位置，以整張圖高度的百分比表示")
                            with gr.Row():
                                manual_target_w_pct = gr.Slider(2, 100, value=25, step=0.5,
                                                                 label="框選寬度 (%)", info="框選區域的寬度，以整張圖寬度的百分比表示")
                                manual_target_h_pct = gr.Slider(2, 100, value=25, step=0.5,
                                                                 label="框選高度 (%)", info="框選區域的高度，以整張圖高度的百分比表示")
                            with gr.Row():
                                manual_target_weight = gr.Slider(0, 1, value=1.0, step=0.05,
                                                                  label="框選區域加強權重", info="框選範圍內的遮罩值，1.0=與自動偵測最強處等級的加強")
                                manual_target_feather_pct = gr.Slider(0, 50, value=20, step=1,
                                                                       label="邊緣羽化 (%)", info="羽化程度佔框選區域短邊的百分比，越高邊界越柔和自然，0=硬邊")

                        with gr.Accordion("4️⃣ 飽和度 / 明度 / RGB通道", open=True) as acc_sat:
                            sat_boost    = gr.Slider(0.5, 3,   value=1.45, step=0.05, label="飽和度倍率", info="銀河/星雲通常需 1.2-1.8 倍增益")
                            bright_boost = gr.Slider(0.5, 2,   value=1.03, step=0.01, label="明度倍率", info="整體亮度微調，通常接近 1.0 即可，避免過曝")
                            r_gain       = gr.Slider(0.5, 1.5, value=1.0,  step=0.01, label="🔴 紅色通道增益", info="加強發射星雲 H-alpha 訊號")
                            g_gain       = gr.Slider(0.5, 1.5, value=1.0,  step=0.01, label="🟢 綠色通道增益", info="通常用來壓低綠色夜空光害")
                            b_gain       = gr.Slider(0.5, 1.5, value=1.0,  step=0.01, label="🔵 藍色通道增益", info="加強反射星雲或藍色年輕恆星")

                        with gr.Accordion("5️⃣ Clarity(局部對比) / 銳化", open=False) as acc_clarity:
                            clarity_blur     = gr.Slider(1,   60,  value=25,   step=1,    label="Clarity 模糊半徑", info="越大越偏向中尺度結構")
                            clarity_strength = gr.Slider(0,   1,   value=0.35, step=0.01, label="Clarity 強度", info="類似 Lightroom 清晰度")
                            sharpen_mode     = gr.Radio(
                                ["internal", "external"], value="internal",
                                label=UI_TRANSLATIONS["sharpen_mode"]["zh"][0],
                                info=UI_TRANSLATIONS["sharpen_mode"]["zh"][1],
                            )
                            sharpen_blur     = gr.Slider(0.5, 10,  value=2,    step=0.1,  label="[internal]銳化模糊半徑", info="決定銳化鎖定的細節尺度，越大銳化範圍越粗")
                            sharpen_amount   = gr.Slider(0.5, 3,   value=1.25, step=0.01, label="[internal]銳化程度", info="過大會使雜訊粒子變粗")
                            sharpen_ext_path    = gr.Textbox(
                                label=UI_TRANSLATIONS["sharpen_ext_path"]["zh"][0], value="",
                                placeholder="例如：C:\\Tools\\CosmicClaritySharpen\\SetiAstroCosmicClarity.exe",
                                info=UI_TRANSLATIONS["sharpen_ext_path"]["zh"][1],
                                visible=False,
                            )
                            sharpen_ext_profile = gr.Dropdown(
                                choices=list(EXTERNAL_TOOL_PROFILES_SHARPEN.keys()),
                                value="自訂 / Custom",
                                label=UI_TRANSLATIONS["sharpen_ext_profile"]["zh"][0],
                                info=UI_TRANSLATIONS["sharpen_ext_profile"]["zh"][1],
                                visible=False,
                            )
                            sharpen_ext_args    = gr.Textbox(
                                label=UI_TRANSLATIONS["sharpen_ext_args"]["zh"][0], value="",
                                placeholder="例如：--sharpening_mode Both --nonstellar_strength 3 --stellar_amount 0.5 --nonstellar_amount 0.5",
                                info=UI_TRANSLATIONS["sharpen_ext_args"]["zh"][1],
                                visible=False,
                            )
                            sharpen_ext_profile.change(
                                fn=lambda name: EXTERNAL_TOOL_PROFILES_SHARPEN.get(name, ""),
                                inputs=[sharpen_ext_profile], outputs=[sharpen_ext_args],
                            )
                            # v1.3.3：mode 不是 external 時，external 三個欄位跟使用者無關；
                            # mode 是 external 時，internal 專用的兩顆滑桿反過來跟使用者無關。
                            # 兩邊都做，面板才不會一直混著一堆用不到的控制項。
                            sharpen_mode.change(
                                fn=_sharpen_mode_visibility,
                                inputs=[sharpen_mode],
                                outputs=[sharpen_blur, sharpen_amount,
                                         sharpen_ext_path, sharpen_ext_profile, sharpen_ext_args],
                            )

                        with gr.Accordion("6️⃣ 降噪", open=False) as acc_denoise:
                            denoise_enable      = gr.Checkbox(label="啟用降噪", value=DEFAULTS[24])
                            denoise_mode        = gr.Radio(["fast", "quality", "external"], value="fast", label="降噪模式", info="fast=雙邊濾波(快); quality=Non-local Means(較乾淨但慢很多，適合最終匯出); external=呼叫外部 ML 降噪工具(見下方設定)")
                            denoise_d           = gr.Slider(1, 15, value=5,  step=1, label="[fast]濾波視窗", info="較大數值降噪範圍廣但耗時")
                            denoise_sigma_color = gr.Slider(1, 50, value=15, step=1, label="[fast]顏色 Sigma", info="越高越能融合差異較大的顏色，但可能糊掉色彩邊界")
                            denoise_sigma_space = gr.Slider(1, 50, value=15, step=1, label="[fast]空間 Sigma", info="越高影響範圍越大的鄰近像素，降噪更平滑但更慢")
                            denoise_nlm_h       = gr.Slider(1, 30, value=10, step=1, label="[quality]亮度降噪強度 h", info="越高越乾淨，但可能抹掉細節", visible=False)
                            denoise_nlm_h_color = gr.Slider(1, 30, value=10, step=1, label="[quality]色彩降噪強度 hColor", info="越高色彩雜訊越乾淨，但可能造成色塊化", visible=False)
                            denoise_ext_path    = gr.Textbox(
                                label="[external]外部降噪工具執行檔路徑", value="",
                                placeholder="例如：C:\\Tools\\NoiseXTerminator\\nxt_cli.exe",
                                info="呼叫使用者電腦上『自行安裝、自行取得授權』的外部降噪工具(如 NoiseXTerminator / DeepSNR CLI)；"
                                     "本程式不內建、不重新散布任何第三方模型，授權責任由使用者自行負責。找不到執行檔或呼叫失敗時，"
                                     "本步驟會自動跳過並回傳未降噪的原圖，不會中斷處理。",
                                visible=False,
                            )
                            denoise_ext_profile = gr.Dropdown(
                                choices=list(EXTERNAL_TOOL_PROFILES_DENOISE.keys()),
                                value="自訂 / Custom",
                                label=UI_TRANSLATIONS["denoise_ext_profile"]["zh"][0],
                                info=UI_TRANSLATIONS["denoise_ext_profile"]["zh"][1],
                                visible=False,
                            )
                            denoise_ext_args    = gr.Textbox(
                                label="[external]額外命令列參數", value="",
                                placeholder="可用 {input} / {output} / {output_noext} 佔位符；留空則預設為「執行檔 輸入檔 輸出檔」",
                                info="若外部工具的命令列語法不是單純的「輸入檔 輸出檔」，可在此用 {input}/{output} 自訂順序與其他參數；"
                                     "若工具會自己補副檔名（如 GraXpert），改用 {output_noext}。",
                                visible=False,
                            )
                            denoise_ext_profile.change(
                                fn=lambda name: EXTERNAL_TOOL_PROFILES_DENOISE.get(name, ""),
                                inputs=[denoise_ext_profile], outputs=[denoise_ext_args],
                            )
                            denoise_mode.change(
                                fn=_denoise_mode_visibility,
                                inputs=[denoise_mode],
                                outputs=[denoise_d, denoise_sigma_color, denoise_sigma_space,
                                         denoise_nlm_h, denoise_nlm_h_color,
                                         denoise_ext_path, denoise_ext_profile, denoise_ext_args],
                            )

                        with gr.Accordion("7️⃣ 星點縮小 / 去星", open=False) as acc_star:
                            star_mode           = gr.Radio(["none","shrink","remove","external"], value="shrink", label="模式", info="external=呼叫外部 ML 去星工具(見下方設定)")
                            star_kernel         = gr.Slider(3,  15,   value=5,    step=1,    label="偵測核大小(≈星點直徑px)", info="應略大於想抓取的中小型星點直徑")
                            star_thresh         = gr.Slider(1,  60,   value=18,   step=1,    label="偵測門檻", info="越低暗星越多，過低會誤抓背景熱雜訊")
                            star_max_area       = gr.Slider(20, 1000, value=250,  step=10,   label="星點最大面積", info="超過此面積的斑塊不視為一般星點，避免誤抓星雲亮核")
                            star_max_area_large = gr.Slider(500,5000, value=2500, step=50,   label="亮星暈光面積上限", info="超過此面積直接排除，避免大範圍暈光被誤判為星點")
                            star_aspect         = gr.Slider(1,  3,    value=1.6,  step=0.05, label="圓度門檻(長寬比)", info="排除長條形星雲結構")
                            star_dilate         = gr.Slider(0,  10,   value=1,    step=1,    label="遮罩外擴基本像素", info="每個星點遮罩固定外擴的像素數，確保完整覆蓋星點邊緣")
                            star_dilate_scale   = gr.Slider(0,  1,    value=0.15, step=0.01, label="依星點大小外擴比例", info="星點越大外擴越多，避免大星周圍殘留光暈")
                            star_shrink_kernel  = gr.Slider(1,  9,    value=3,    step=1,    label="[縮星]侵蝕核大小", info="每次侵蝕使用的核心大小，越大縮星效果越明顯")
                            star_shrink_iter    = gr.Slider(1,  5,    value=1,    step=1,    label="[縮星]侵蝕次數", info="重複侵蝕的次數，越多星點縮得越小")
                            star_shrink_strength= gr.Slider(0,  1,    value=0.8,  step=0.01, label="[縮星]套用強度", info="0 為不縮星，1 為完全套用侵蝕結果")
                            star_inpaint_radius = gr.Slider(1,  20,   value=5,    step=1,    label="[去星]單星取樣半徑", info="從星點周圍多遠的範圍取樣來填補去星後的背景")
                            star_feather_px     = gr.Slider(0,  8,    value=2.0,  step=0.25, label="[去星]邊緣羽化程度", info="去星邊界的柔化寬度，越高過渡越自然但越模糊")
                            star_noise_strength = gr.Slider(0,  2,    value=1.0,  step=0.05, label="[去星]雜訊回填強度", info="0 為不回填")
                            star_ext_path       = gr.Textbox(
                                label="[external]外部去星工具執行檔路徑", value="",
                                placeholder="例如：C:\\Tools\\StarXTerminator\\sxt_cli.exe",
                                info="呼叫使用者電腦上『自行安裝、自行取得授權』的外部去星工具(如 StarXTerminator / StarNet CLI)；"
                                     "本程式不內建、不重新散布任何第三方模型，授權責任由使用者自行負責。找不到執行檔或呼叫失敗時，"
                                     "本步驟會自動跳過並回傳未去星的原圖，不會中斷處理。",
                                visible=False,
                            )
                            star_ext_profile    = gr.Dropdown(
                                choices=list(EXTERNAL_TOOL_PROFILES_STAR.keys()),
                                value="自訂 / Custom",
                                label=UI_TRANSLATIONS["star_ext_profile"]["zh"][0],
                                info=UI_TRANSLATIONS["star_ext_profile"]["zh"][1],
                                visible=False,
                            )
                            star_ext_args       = gr.Textbox(
                                label="[external]額外命令列參數", value="",
                                placeholder="可用 {input} / {output} / {output_noext} 佔位符；留空則預設為「執行檔 輸入檔 輸出檔」",
                                info="若外部工具的命令列語法不是單純的「輸入檔 輸出檔」，可在此用 {input}/{output} 自訂順序與其他參數；"
                                     "若工具會自己補副檔名，改用 {output_noext}。",
                                visible=False,
                            )
                            star_ext_profile.change(
                                fn=lambda name: EXTERNAL_TOOL_PROFILES_STAR.get(name, ""),
                                inputs=[star_ext_profile], outputs=[star_ext_args],
                            )
                            star_mode.change(
                                fn=_star_mode_visibility,
                                inputs=[star_mode],
                                outputs=[star_shrink_kernel, star_shrink_iter, star_shrink_strength,
                                         star_ext_path, star_ext_profile, star_ext_args],
                            )

                        with gr.Accordion("7️⃣b 大範圍偵測(密集星團/大片暈光)", open=False) as acc_cluster:
                            multiscale_enable      = gr.Checkbox(label="啟用大範圍偵測", value=DEFAULTS[43])
                            cluster_kernel         = gr.Slider(5,    50,    value=21,    step=1,   label="偵測核大小", info="用來偵測大範圍密集星團的局部視窗，應大於一般星點")
                            cluster_thresh         = gr.Slider(1,    60,    value=12,    step=1,   label="偵測門檻", info="越低越容易把稀疏星群也判定為星團")
                            cluster_min_area       = gr.Slider(50,   2000,  value=300,   step=10,  label="最小面積", info="小於此面積不視為星團，避免與一般星點混淆")
                            cluster_max_area       = gr.Slider(2000, 30000, value=15000, step=100, label="最大面積", info="超過此面積可能是星雲亮區而非星團，會被排除")
                            cluster_aspect         = gr.Slider(1,    5,     value=2.5,   step=0.1, label="長寬比門檻", info="排除過於狹長的區域，避免誤抓塵埃帶或條紋雜訊")
                            cluster_dilate         = gr.Slider(0,    15,    value=4,     step=1,   label="遮罩外擴像素", info="星團遮罩外擴的像素數，確保完整覆蓋週邊暗星")
                            cluster_inpaint_radius = gr.Slider(1,    30,    value=14,    step=1,   label="[去星]星團取樣半徑", info="去除星團時，從多遠範圍取樣填補背景")

                    reset_btn = gr.Button("↩️ 重設為預設值", variant="secondary")

        # ════════════════ CENTER COLUMN — Preview ════════════
        with gr.Column(scale=3, elem_id="center-col"):
            with gr.Tabs():

                # ── Tab: Live Preview ─────────────────────────
                with gr.Tab("🖼️ 即時預覽") as tab_preview:
                    live_preview_enabled = gr.Checkbox(
                        value=True,
                        label=UI_TRANSLATIONS["live_preview_enabled"]["zh"][0],
                        info=UI_TRANSLATIONS["live_preview_enabled"]["zh"][1],
                    )
                    compare_mode = gr.Radio(
                        choices=["並排顯示", "滑桿疊圖"],
                        value="並排顯示",
                        label="原圖對比模式",
                        info="「滑桿疊圖」：同一張圖拖曳把手切換前後；「並排顯示」：左右兩張圖分開顯示",
                        elem_id="compare-mode-radio",
                    )
                    with gr.Row(elem_id="side-by-side-row", visible=True) as side_by_side_row:
                        original_preview_image = gr.Image(
                            label="原圖(未處理)", type="numpy", height=580
                        )
                        preview_image = gr.Image(
                            label="即時預覽效果(處理後)", type="numpy", height=580
                        )
                    compare_slider_html = gr.HTML(
                        value="<div style='color:#888;padding:40px;text-align:center;'>⚠️ 尚未載入圖片</div>",
                        visible=False,
                        elem_id="compare-slider-html",
                    )
                    preview_status = gr.Markdown("等待圖片載入...")



                # ── Tab: Layer Preview ────────────────────────
                with gr.Tab("🔍 遮罩 / 去星層") as tab_layer:
                    layer_hint_md = gr.Markdown(
                        "⚠️ *縮圖版本，僅供調整參數時參考；勾選「全解析度」後以全圖運算。*"
                    )
                    layer_preview_btn = gr.Button(
                        "🔄 產生 / 更新圖層預覽", variant="secondary"
                    )
                    with gr.Row():
                        mask_image     = gr.Image(label="星點遮罩 (Star Mask)",    type="numpy")
                        starless_image = gr.Image(label="去星背景層 (Starless)",   type="numpy")
                        target_mask_overlay_image = gr.Image(label="自動局部拉伸偵測區域 (Auto Local Target)", type="numpy")

                # ── Tab: Export ───────────────────────────────
                with gr.Tab("💾 匯出全解析度") as tab_export:
                    output_dir  = gr.Textbox(label="輸出資料夾",         value=_default_output_base())
                    output_name = gr.Textbox(label="輸出檔名(不含副檔名)", value="processed")
                    save_layers = gr.Checkbox(
                        label="額外輸出星點遮罩 + 去星背景層(可供後續人工疊圖疊加)",
                        value=True
                    )
                    export_formats = gr.CheckboxGroup(
                        choices=["JPG", "TIFF"],
                        value=["JPG", "TIFF"],
                        label="匯出格式",
                        info="只需要其中一種格式時可取消勾選，省下匯出時間與硬碟空間",
                    )
                    export_btn    = gr.Button("🚀 開始高解析度跑圖與匯出", variant="primary")
                    export_status = gr.Markdown("")
                    export_files  = gr.File(
                        label="下載生成的結果檔案", file_count="multiple"
                    )

                # ── Tab: Batch Processing ──────────────────────
                with gr.Tab("🗂️ 批次處理") as tab_batch:
                    batch_hint_md = gr.Markdown(
                        "**批次處理**：把目前調好的參數，套用到「📂 選圖 & 設定」分頁中"
                        "來源資料夾裡的每一張圖片，逐一以全解析度處理並輸出。\n\n"
                        "⚠️ 建議先用單張圖片調好參數、確認效果滿意後，再執行批次處理。"
                    )
                    batch_out_dir = gr.Textbox(label="批次輸出資料夾", value=os.path.join(_default_output_base(), "batch"))
                    batch_want_layers = gr.Checkbox(
                        label="每張圖同時輸出星點遮罩 + 去星背景層", value=False
                    )
                    batch_formats = gr.CheckboxGroup(
                        choices=["JPG", "TIFF"],
                        value=["JPG", "TIFF"],
                        label="匯出格式",
                        info="只需要其中一種格式時可取消勾選，批次量大時能省下不少時間與硬碟空間",
                    )
                    with gr.Row():
                        batch_btn = gr.Button("🚀 開始批次處理整個資料夾", variant="primary")
                        batch_stop_btn = gr.Button("⏹ 停止批次", variant="stop")
                    # v1.3.3 Track B：停止旗標是逐張圖片邊界檢查，這是設計本身
                    # （不會留下寫到一半的殘檔），但如果降噪／去星／銳化目前是
                    # external 模式，正在跑的那張圖要等外部工具的 subprocess
                    # 結束或撞到逾時（預設 180 秒）才會走到下一次檢查點——決定
                    # 先把這個既有行為寫清楚，而不是實作提早中止 subprocess
                    # （那需要把 Popen+poll 換掉現在會卡住的 subprocess.run）。
                    batch_stop_hint = gr.Markdown(
                        "ℹ️ 停止會在**目前這張圖片處理完後**才生效；"
                        "若降噪／去星／銳化正使用 external 模式呼叫外部工具，"
                        "最長可能要等到該外部工具逾時（預設 180 秒）才會真正停下。",
                        elem_id="batch-stop-hint",
                    )
                    batch_status = gr.Markdown("")

                # ── Tab: Local ROI Preview ────────────────────
                with gr.Tab("⚡ 局部預覽") as tab_local:
                    local_hint_md = gr.Markdown(
                        "**局部預覽**：從原圖裁一小塊，以全解析度品質跑完整流程，結果和最終匯出完全一致。\n\n"
                        "調整下方 X/Y 位置滑桿選取感興趣的區域，點「更新」即可。"
                    )
                    with gr.Row():
                        local_x_pct  = gr.Slider(0, 100, value=50, step=0.5,
                                                  label="X 中心位置 (%)", elem_id="local-x-pct",
                                                  info="裁切區域中心點的水平位置，以整張圖寬度的百分比表示")
                        local_y_pct  = gr.Slider(0, 100, value=50, step=0.5,
                                                  label="Y 中心位置 (%)", elem_id="local-y-pct",
                                                  info="裁切區域中心點的垂直位置，以整張圖高度的百分比表示")
                        local_crop_px = gr.Slider(128, 2000, value=512, step=64,
                                                   label="裁切大小 (px)", elem_id="local-crop-px",
                                                   info="裁切區域的邊長，越小處理越快但看到的範圍也越小")
                    local_preview_btn = gr.Button("⚡ 更新局部預覽", variant="primary",
                                                   elem_id="local-preview-btn")
                    local_status = gr.Markdown("")
                    with gr.Row():
                        local_overview_img = gr.Image(
                            label="裁切位置概覽（紅框=裁切區）", type="numpy", height=360,
                            elem_id="local-overview-img"
                        )
                        local_result_img = gr.Image(
                            label="局部處理結果（全解析度品質）", type="numpy", height=360,
                            elem_id="local-result-img"
                        )

        # ════════════════ RIGHT COLUMN — Info Panel ══════════

        with gr.Column(scale=1, elem_id="right-col") as right_col:


            with gr.Accordion("📊 Histogram", open=True) as acc_hist:
                preview_hist = gr.Image(
                    label="RGB 通道分佈曲線", type="numpy", height=190
                )

            with gr.Accordion("⭐ Star Mask 預覽", open=False) as acc_mask_mini:
                mask_mini = gr.Image(
                    label="星點遮罩(產生圖層後自動更新)", type="numpy", height=170
                )

            with gr.Accordion("🖥️ 系統監控", open=True) as acc_monitor:
                monitor_html        = gr.HTML(get_system_stats_html())
                with gr.Row():
                    monitor_refresh_btn = gr.Button("🔄 更新監控資訊", size="sm")
                    monitor_auto_refresh = gr.Checkbox(
                        value=True, label=UI_TRANSLATIONS["monitor_auto_refresh"]["zh"][0], scale=0,
                        info=UI_TRANSLATIONS["monitor_auto_refresh"]["zh"][1],
                    )
                monitor_timer = gr.Timer(3, active=True)

    # ── Bottom Status Bar ─────────────────────────────────────
    status_bar_out = gr.HTML(get_status_bar_html())

    # ── Event Bindings ────────────────────────────────────────
    PARAM_COMPONENTS = [
        bg_enable, bg_downscale, bg_min_filter, bg_blur_sigma, bg_subtract,
        wb_enable, wb_min, wb_max,
        black_pct, stretch_factor, white_pct,
        local_target_enable, local_target_strength, local_target_radius, local_target_sensitivity,
        sat_boost, bright_boost, r_gain, g_gain, b_gain,
        clarity_blur, clarity_strength, sharpen_blur, sharpen_amount,
        denoise_enable, denoise_mode, denoise_d, denoise_sigma_color, denoise_sigma_space,
        denoise_nlm_h, denoise_nlm_h_color,
        star_mode, star_kernel, star_thresh, star_max_area, star_max_area_large, star_aspect,
        star_dilate, star_dilate_scale, star_shrink_kernel, star_shrink_iter,
        star_shrink_strength, star_inpaint_radius,
        multiscale_enable, cluster_kernel, cluster_thresh, cluster_min_area, cluster_max_area,
        cluster_aspect, cluster_dilate, cluster_inpaint_radius,
        star_feather_px, star_noise_strength,
        manual_target_enable, manual_target_x_pct, manual_target_y_pct,
        manual_target_w_pct, manual_target_h_pct, manual_target_weight, manual_target_feather_pct,
        manual_target_shape,
        denoise_ext_path, denoise_ext_args,
        star_ext_path, star_ext_args,
        sharpen_mode, sharpen_ext_path, sharpen_ext_args,
    ]

    sliders_for_release = [
        bg_downscale, bg_min_filter, bg_blur_sigma, bg_subtract, wb_min, wb_max,
        black_pct, stretch_factor, white_pct,
        local_target_strength, local_target_radius, local_target_sensitivity,
        sat_boost, bright_boost,
        r_gain, g_gain, b_gain,
        clarity_blur, clarity_strength, sharpen_blur, sharpen_amount,
        denoise_d, denoise_sigma_color, denoise_sigma_space, denoise_nlm_h, denoise_nlm_h_color,
        star_kernel, star_thresh, star_max_area, star_max_area_large, star_aspect,
        star_dilate, star_dilate_scale, star_shrink_kernel, star_shrink_iter,
        star_shrink_strength, star_inpaint_radius,
        cluster_kernel, cluster_thresh, cluster_min_area, cluster_max_area,
        cluster_aspect, cluster_dilate, cluster_inpaint_radius,
        star_feather_px, star_noise_strength,
        manual_target_x_pct, manual_target_y_pct, manual_target_w_pct, manual_target_h_pct,
        manual_target_weight, manual_target_feather_pct,
    ]
    toggles_for_change = [bg_enable, wb_enable, local_target_enable, manual_target_enable, manual_target_shape, denoise_enable, denoise_mode, star_mode, multiscale_enable, sharpen_mode]

    # Preview common inputs / outputs
    _PREV_IN  = [state_preview_base, state_preview_scale, state_full,
                 use_full_res_preview, live_preview_enabled, state_lang] + PARAM_COMPONENTS
    _PREV_OUT = [preview_image, original_preview_image, preview_hist, preview_status, compare_slider_html]


    # ── Language Translation Handlers ──────────────────────────
    TRANSLATED_COMPONENTS_MAP = [
        (folder_box, "folder_box"),
        (scan_btn, "scan_btn"),
        (file_dropdown, "file_dropdown"),
        (upload_file, "upload_file"),
        (preview_size, "preview_size"),
        (use_full_res_preview, "use_full_res_preview"),
        (live_preview_enabled, "live_preview_enabled"),
        (load_btn, "load_btn"),
        (cfg_export_btn, "cfg_export_btn"),
        (cfg_import_file, "cfg_import_file"),
        (bg_enable, "bg_enable"),
        (bg_downscale, "bg_downscale"),
        (bg_min_filter, "bg_min_filter"),
        (bg_blur_sigma, "bg_blur_sigma"),
        (bg_subtract, "bg_subtract"),
        (wb_enable, "wb_enable"),
        (wb_min, "wb_min"),
        (wb_max, "wb_max"),
        (black_pct, "black_pct"),
        (stretch_factor, "stretch_factor"),
        (white_pct, "white_pct"),
        (acc_local_target, "acc_local_target"),
        (local_target_enable, "local_target_enable"),
        (local_target_strength, "local_target_strength"),
        (local_target_radius, "local_target_radius"),
        (local_target_sensitivity, "local_target_sensitivity"),
        (manual_target_hint_md, "manual_target_hint_md"),
        (manual_target_enable, "manual_target_enable"),
        (manual_target_shape, "manual_target_shape"),
        (manual_target_x_pct, "manual_target_x_pct"),
        (manual_target_y_pct, "manual_target_y_pct"),
        (manual_target_w_pct, "manual_target_w_pct"),
        (manual_target_h_pct, "manual_target_h_pct"),
        (manual_target_weight, "manual_target_weight"),
        (manual_target_feather_pct, "manual_target_feather_pct"),
        (sat_boost, "sat_boost"),
        (bright_boost, "bright_boost"),
        (r_gain, "r_gain"),
        (g_gain, "g_gain"),
        (b_gain, "b_gain"),
        (clarity_blur, "clarity_blur"),
        (clarity_strength, "clarity_strength"),
        (sharpen_mode, "sharpen_mode"),
        (sharpen_blur, "sharpen_blur"),
        (sharpen_amount, "sharpen_amount"),
        (sharpen_ext_path, "sharpen_ext_path"),
        (sharpen_ext_profile, "sharpen_ext_profile"),
        (sharpen_ext_args, "sharpen_ext_args"),
        (denoise_enable, "denoise_enable"),
        (denoise_mode, "denoise_mode"),
        (denoise_d, "denoise_d"),
        (denoise_sigma_color, "denoise_sigma_color"),
        (denoise_sigma_space, "denoise_sigma_space"),
        (denoise_nlm_h, "denoise_nlm_h"),
        (denoise_nlm_h_color, "denoise_nlm_h_color"),
        (denoise_ext_path, "denoise_ext_path"),
        (denoise_ext_profile, "denoise_ext_profile"),
        (denoise_ext_args, "denoise_ext_args"),
        (star_mode, "star_mode"),
        (star_kernel, "star_kernel"),
        (star_thresh, "star_thresh"),
        (star_max_area, "star_max_area"),
        (star_max_area_large, "star_max_area_large"),
        (star_aspect, "star_aspect"),
        (star_dilate, "star_dilate"),
        (star_dilate_scale, "star_dilate_scale"),
        (star_shrink_kernel, "star_shrink_kernel"),
        (star_shrink_iter, "star_shrink_iter"),
        (star_shrink_strength, "star_shrink_strength"),
        (star_inpaint_radius, "star_inpaint_radius"),
        (star_feather_px, "star_feather_px"),
        (star_noise_strength, "star_noise_strength"),
        (star_ext_path, "star_ext_path"),
        (star_ext_profile, "star_ext_profile"),
        (star_ext_args, "star_ext_args"),
        (multiscale_enable, "multiscale_enable"),
        (cluster_kernel, "cluster_kernel"),
        (cluster_thresh, "cluster_thresh"),
        (cluster_min_area, "cluster_min_area"),
        (cluster_max_area, "cluster_max_area"),
        (cluster_aspect, "cluster_aspect"),
        (cluster_dilate, "cluster_dilate"),
        (cluster_inpaint_radius, "cluster_inpaint_radius"),
        (reset_btn, "reset_btn"),
        (original_preview_image, "original_preview_image"),
        (preview_image, "preview_image"),
        (layer_preview_btn, "layer_preview_btn"),
        (mask_image, "mask_image"),
        (starless_image, "starless_image"),
        (target_mask_overlay_image, "target_mask_overlay_image"),
        (output_dir, "output_dir"),
        (output_name, "output_name"),
        (save_layers, "save_layers"),
        (export_formats, "export_formats"),
        (export_btn, "export_btn"),
        (export_files, "export_files"),
        (preview_hist, "preview_hist"),
        (mask_mini, "mask_mini"),
        (monitor_refresh_btn, "monitor_refresh_btn"),
        (monitor_auto_refresh, "monitor_auto_refresh"),
        (tab_load, "tab_load"),
        (tab_param, "tab_param"),
        (acc_bg, "acc_bg"),
        (acc_wb, "acc_wb"),
        (acc_stretch, "acc_stretch"),
        (acc_sat, "acc_sat"),
        (acc_clarity, "acc_clarity"),
        (acc_denoise, "acc_denoise"),
        (acc_star, "acc_star"),
        (acc_cluster, "acc_cluster"),
        (tab_preview, "tab_preview"),
        (tab_layer, "tab_layer"),
        (tab_export, "tab_export"),
        (acc_hist, "acc_hist"),
        (acc_mask_mini, "acc_mask_mini"),
        (acc_monitor, "acc_monitor"),
        (cfg_header_md, "cfg_header_md"),
        (cfg_download, "cfg_download"),
        (layer_hint_md, "layer_hint_md"),
        (focus_btn, "focus_btn"),
        (compare_mode, "compare_mode"),
        (tab_local, "tab_local"),
        (local_x_pct, "local_x_pct"),
        (local_y_pct, "local_y_pct"),
        (local_crop_px, "local_crop_px"),
        (local_preview_btn, "local_preview_btn"),
        (local_overview_img, "local_overview_img"),
        (local_result_img, "local_result_img"),
        (local_hint_md, "local_hint_md"),
        (tab_batch, "tab_batch"),
        (batch_hint_md, "batch_hint_md"),
        (batch_out_dir, "batch_out_dir"),
        (batch_want_layers, "batch_want_layers"),
        (batch_formats, "batch_formats"),
        (batch_stop_btn, "batch_stop_btn"),
        (batch_btn, "batch_btn"),
        (close_btn, "close_btn"),
        (presets_hint_md, "presets_hint_md"),
        (acc_presets, "acc_presets"),
        (preset_milky_way_btn, "preset_milky_way_btn"),
        (preset_nebula_btn, "preset_nebula_btn"),
        (preset_light_pollution_btn, "preset_light_pollution_btn"),
        (snapshot_hint_md, "snapshot_hint_md"),
        (acc_snapshots, "acc_snapshots"),
        (snap_a_save_btn, "snap_a_save_btn"),
        (snap_a_load_btn, "snap_a_load_btn"),
        (snap_b_save_btn, "snap_b_save_btn"),
        (snap_b_load_btn, "snap_b_load_btn"),
        (snap_c_save_btn, "snap_c_save_btn"),
        (snap_c_load_btn, "snap_c_load_btn"),
        (cfg_export_name, "cfg_export_name"),
        (snap_file_hint_md, "snap_file_hint_md"),
        (snap_file_target, "snap_file_target"),
        (snap_file_name, "snap_file_name"),
        (snap_save_file_btn, "snap_save_file_btn"),
        (snap_load_file, "snap_load_file"),
        (snap_file_download, "snap_file_download"),
    ]

    def make_change_lang(lang):
        def handler(is_focused, load_status_val, cfg_status_val, preview_status_val, compare_html_val,
                    snap_a_status_val, snap_b_status_val, snap_c_status_val):
            updates = []
            for comp, key in TRANSLATED_COMPONENTS_MAP:
                entry = UI_TRANSLATIONS[key]
                val = entry[lang]                 # str or (label, info) tuple
                label_txt  = val[0] if isinstance(val, tuple) else val
                info_txt   = val[1] if isinstance(val, tuple) else None

                if key == "focus_btn":
                    if is_focused:
                        btn_label = "⬅ 展開面板" if lang == "zh" else "⬅ Show Panels"
                    else:
                        btn_label = "⛶ 專注預覽" if lang == "zh" else "⛶ Focus Preview"
                    updates.append(gr.update(value=btn_label))
                elif key in ["scan_btn", "load_btn", "cfg_export_btn", "reset_btn",
                             "layer_preview_btn", "export_btn", "monitor_refresh_btn",
                             "local_preview_btn", "batch_btn", "batch_stop_btn", "close_btn",
                             "preset_milky_way_btn", "preset_nebula_btn", "preset_light_pollution_btn",
                             "snap_a_save_btn", "snap_a_load_btn", "snap_b_save_btn", "snap_b_load_btn",
                             "snap_c_save_btn", "snap_c_load_btn", "snap_save_file_btn"]:
                    updates.append(gr.update(value=label_txt))
                elif key in ["cfg_header_md", "layer_hint_md", "local_hint_md", "batch_hint_md",
                             "presets_hint_md", "snapshot_hint_md", "snap_file_hint_md",
                             "manual_target_hint_md"]:
                    updates.append(gr.update(value=label_txt))
                elif key == "compare_mode":
                    choices_zh = ["並排顯示", "滑桿疊圖"]
                    choices_en = ["Side by Side", "Slider Overlay"]
                    kw = dict(label=label_txt, choices=choices_zh if lang == "zh" else choices_en)
                    if info_txt:
                        kw["info"] = info_txt
                    updates.append(gr.update(**kw))
                elif info_txt is not None:
                    # 有 info 文字的欄位（如 preview_size）：同時更新 label 和 info
                    # gr.Image 這類元件在目前版本的 Gradio 不支援 info 參數，
                    # 傳入會直接丟例外導致切換語言失敗，因此改成併入 label 文字顯示
                    if isinstance(comp, gr.Image):
                        updates.append(gr.update(label=f"{label_txt}　{info_txt}"))
                    else:
                        updates.append(gr.update(label=label_txt, info=info_txt))
                else:
                    updates.append(gr.update(label=label_txt))

            # ── 狀態/佔位文字：只有「目前顯示的內容仍是初始佔位字」時才跟著語言切換，
            #    避免洗掉使用者已經看到的真實處理結果訊息（例如「已載入 xxx.tif」）。
            def _swap_if_placeholder(current_val, ph_key):
                zh_default, en_default = PLACEHOLDER_TEXTS[ph_key]["zh"], PLACEHOLDER_TEXTS[ph_key]["en"]
                if current_val in (zh_default, en_default):
                    return gr.update(value=PLACEHOLDER_TEXTS[ph_key][lang])
                return gr.update()

            status_updates = [
                _swap_if_placeholder(load_status_val, "load_status"),
                _swap_if_placeholder(cfg_status_val, "cfg_status"),
                _swap_if_placeholder(preview_status_val, "preview_status"),
                _swap_if_placeholder(compare_html_val, "compare_slider_html"),
                _swap_if_placeholder(snap_a_status_val, "snap_a_status"),
                _swap_if_placeholder(snap_b_status_val, "snap_b_status"),
                _swap_if_placeholder(snap_c_status_val, "snap_c_status"),
            ]

            return [lang] + updates + status_updates
        return handler


    # ── Scan folder ───────────────────────────────────────────
    scan_btn.click(fn=scan_folder, inputs=[folder_box], outputs=[file_dropdown])

    # ── Load image ────────────────────────────────────────────
    load_btn.click(
        fn=load_image_fn,
        inputs=[folder_box, file_dropdown, upload_file, preview_size, state_lang],
        outputs=[state_full, state_preview_base, state_preview_scale,
                 load_status, output_name,
                 preview_image, original_preview_image, preview_hist],
    ).then(
        fn=update_preview_fn,
        inputs=_PREV_IN,
        outputs=_PREV_OUT,
    ).then(
        fn=get_status_bar_html,
        inputs=[state_last_export_time, state_lang],
        outputs=[status_bar_out],
    )

    # ── Preview size change ───────────────────────────────────
    preview_size.release(
        fn=resize_preview_base_fn,
        inputs=[state_full, preview_size],
        outputs=[state_preview_base, state_preview_scale],
    ).then(
        fn=update_preview_fn, inputs=_PREV_IN, outputs=_PREV_OUT,
    )

    # ── Full-res toggle ───────────────────────────────────────
    use_full_res_preview.change(
        fn=update_preview_fn, inputs=_PREV_IN, outputs=_PREV_OUT,
    )

    # ── Live preview pause/resume toggle（v1.2.1 主線 A）──────
    # 關閉時 update_preview_fn 自己會 no-op；重新開啟時這裡會立刻補跑一次，
    # 避免恢復後畫面停在暫停前的舊結果。
    live_preview_enabled.change(
        fn=update_preview_fn, inputs=_PREV_IN, outputs=_PREV_OUT,
    )

    # ── Sliders (on release) ──────────────────────────────────
    for comp in sliders_for_release:
        comp.release(fn=update_preview_fn, inputs=_PREV_IN, outputs=_PREV_OUT)

    # ── Toggles (on change) ───────────────────────────────────
    for comp in toggles_for_change:
        comp.change(fn=update_preview_fn, inputs=_PREV_IN, outputs=_PREV_OUT)

    # ── Reset defaults ────────────────────────────────────────
    reset_btn.click(
        fn=lambda: list(DEFAULTS), outputs=PARAM_COMPONENTS,
    ).then(
        fn=update_preview_fn, inputs=_PREV_IN, outputs=_PREV_OUT,
    )

    # ── Presets（新手起始參數）──────────────────────────────
    def _make_preset_handler(preset_key):
        def handler(lang):
            *values, status = apply_preset_fn(preset_key, lang)
            return values + [status]
        return handler

    for btn, key in [
        (preset_milky_way_btn, "milky_way"),
        (preset_nebula_btn, "nebula"),
        (preset_light_pollution_btn, "heavy_light_pollution"),
    ]:
        btn.click(
            fn=_make_preset_handler(key),
            inputs=[state_lang],
            outputs=PARAM_COMPONENTS + [preset_status],
        ).then(
            fn=update_preview_fn, inputs=_PREV_IN, outputs=_PREV_OUT,
        )

    # ── 參數快照 A / B / C ───────────────────────────────────
    for save_btn, load_btn_snap, state_snap, status_comp, slot in [
        (snap_a_save_btn, snap_a_load_btn, state_snap_a, snap_a_status, "A"),
        (snap_b_save_btn, snap_b_load_btn, state_snap_b, snap_b_status, "B"),
        (snap_c_save_btn, snap_c_load_btn, state_snap_c, snap_c_status, "C"),
    ]:
        save_btn.click(
            fn=lambda lang, *pv, _slot=slot: save_snapshot_fn(_slot, lang, *pv),
            inputs=[state_lang] + PARAM_COMPONENTS,
            outputs=[state_snap, status_comp],
        )
        load_btn_snap.click(
            fn=lambda snap, lang, _slot=slot: load_snapshot_fn(snap, _slot, lang),
            inputs=[state_snap, state_lang],
            outputs=PARAM_COMPONENTS + [status_comp],
        ).then(
            fn=update_preview_fn, inputs=_PREV_IN, outputs=_PREV_OUT,
        )

    # ── Layer preview (also updates Star Mask mini in right panel) ──
    def _layer_with_mini(*args):
        """Wrapper: calls layer_preview_fn and duplicates mask to mask_mini."""
        mask, starless, target_overlay, status = layer_preview_fn(*args)
        return mask, starless, target_overlay, status, mask   # 5th = mask_mini

    layer_preview_btn.click(
        fn=_layer_with_mini,
        inputs=[state_preview_base, state_preview_scale, state_full,
                use_full_res_preview, state_lang] + PARAM_COMPONENTS,
        outputs=[mask_image, starless_image, target_mask_overlay_image, preview_status, mask_mini],
    )

    # ── Full-res export ───────────────────────────────────────
    export_btn.click(
        fn=export_fn,
        inputs=[state_full, output_dir, output_name, save_layers, state_lang, export_formats] + PARAM_COMPONENTS,
        outputs=[export_status, export_files, state_last_export_time],
    ).then(
        fn=get_status_bar_html,
        inputs=[state_last_export_time, state_lang],
        outputs=[status_bar_out],
    )

    # ── Batch processing (reuses folder_box from the load tab as source) ──
    batch_event = batch_btn.click(
        fn=batch_process_fn,
        inputs=[folder_box, batch_out_dir, batch_want_layers, state_lang,
                state_batch_stop, batch_formats] + PARAM_COMPONENTS,
        outputs=[batch_status],
    )

    # v1.3.2：批次停止按鈕。除了原地修改 state_batch_stop（讓正在跑的
    # batch_process_fn() 在下一張圖片開始前的檢查點看到旗標並提早結束、
    # 印出正常的「已停止」摘要訊息），另外也用 Gradio 內建的 cancels 機制
    # 保底：萬一某次呼叫卡在檢查點之間很久沒有 yield（例如單張圖片處理
    # 特別慢），至少 Gradio 佇列本身也會嘗試中止這個事件。
    def request_batch_stop_fn(stop_state, lang, denoise_mode_val, star_mode_val, sharpen_mode_val):
        if isinstance(stop_state, dict):
            stop_state["stop"] = True
        else:
            stop_state = {"stop": True}
        msg = ("⏹ 已送出停止要求，將在目前這張圖片處理完後停止批次…" if lang == "zh"
               else "⏹ Stop requested — batch will halt after the current file finishes…")
        # v1.3.3 Track B：如果目前是 external 模式，額外提醒使用者「這張」可能
        # 要等外部工具的 subprocess 結束或逾時才會真的停下，避免使用者以為
        # 按了沒反應（見 batch_stop_hint 的固定說明，這裡是按下當下的即時提醒）。
        if "external" in (denoise_mode_val, star_mode_val, sharpen_mode_val):
            msg += (
                "（目前有步驟使用 external 模式，這張圖可能要再等外部工具"
                "結束或逾時才會真正停止）" if lang == "zh" else
                " (an external-tool step is active, so the current file may take "
                "until that tool finishes or times out before it truly stops)"
            )
        return stop_state, msg

    batch_stop_btn.click(
        fn=request_batch_stop_fn,
        inputs=[state_batch_stop, state_lang, denoise_mode, star_mode, sharpen_mode],
        outputs=[state_batch_stop, batch_status],
    ).then(
        fn=None, inputs=None, outputs=None, cancels=[batch_event], queue=False,
    )

    # ── Local ROI preview ─────────────────────────────────────
    local_preview_btn.click(
        fn=local_preview_fn,
        inputs=[state_full, local_x_pct, local_y_pct, local_crop_px,
                state_lang] + PARAM_COMPONENTS,
        outputs=[local_result_img, local_overview_img, local_status],
    )

    # 滑桿移動時即時更新概覽圖（只畫框，不跑 pipeline，極快）
    def _update_overview_only(full_img, x_pct, y_pct, crop_px):
        if full_img is None:
            return None
        h, w = full_img.shape[:2]
        crop_px = max(64, min(int(crop_px), min(h, w)))
        cx = int(w * x_pct / 100.0)
        cy = int(h * y_pct / 100.0)
        half = crop_px // 2
        x0 = max(0, min(cx - half, w - crop_px))
        y0 = max(0, min(cy - half, h - crop_px))
        x1, y1 = x0 + crop_px, y0 + crop_px
        thumb_max = 700
        scale_t = min(1.0, thumb_max / max(h, w))
        tw, th = max(1, int(w * scale_t)), max(1, int(h * scale_t))
        thumb = cv2.resize((full_img * 255).astype(np.uint8), (tw, th), interpolation=cv2.INTER_AREA)
        rx0, ry0 = int(x0 * scale_t), int(y0 * scale_t)
        rx1, ry1 = int(x1 * scale_t), int(y1 * scale_t)
        cv2.rectangle(thumb, (rx0, ry0), (rx1, ry1), (255, 60, 60), max(2, int(3 * scale_t)))
        label = f"{crop_px}×{crop_px} px"
        cv2.putText(thumb, label, (rx0 + 4, max(ry0 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 60, 60), 1, cv2.LINE_AA)
        return thumb

    for _slider in [local_x_pct, local_y_pct, local_crop_px]:
        _slider.release(
            fn=_update_overview_only,
            inputs=[state_full, local_x_pct, local_y_pct, local_crop_px],
            outputs=[local_overview_img],
        )


    # ── Config export / import ────────────────────────────────
    cfg_export_btn.click(
        fn=export_config_fn,
        inputs=[cfg_export_name] + PARAM_COMPONENTS,
        outputs=[cfg_status, cfg_download],
    )

    cfg_import_file.change(
        fn=import_config_fn,
        inputs=[cfg_import_file],
        outputs=[cfg_status] + PARAM_COMPONENTS,
    ).then(
        fn=update_preview_fn, inputs=_PREV_IN, outputs=_PREV_OUT,
    )

    # ── Named snapshot save-to-file / load-from-file ──────────
    snap_save_file_btn.click(
        fn=export_snapshot_to_file_fn,
        inputs=[snap_file_target, snap_file_name, state_snap_a, state_snap_b, state_snap_c, state_lang],
        outputs=[snap_file_status, snap_file_download],
    )

    snap_load_file.change(
        fn=import_snapshot_from_file_fn,
        inputs=[snap_load_file, snap_file_target, state_lang],
        outputs=[state_snap_a, state_snap_b, state_snap_c,
                 snap_a_status, snap_b_status, snap_c_status,
                 snap_file_status, snap_file_name],
    )

    # ── System monitor refresh ────────────────────────────────
    monitor_refresh_btn.click(
        fn=get_system_stats_html,
        outputs=[monitor_html],
    )

    # ── System monitor auto-refresh（v1.2.1 主線 B）───────────
    # Timer 週期性 tick 呼叫既有的 get_system_stats_html，不新增運算；
    # 勾選框關閉時回傳 active=False 的 Timer 更新，暫停輪詢，
    # 手動按鈕在暫停狀態下仍可正常使用。
    monitor_timer.tick(fn=get_system_stats_html, outputs=[monitor_html])

    # v1.3.3 Track B：status_bar_out 過去完全沒有定時刷新，只在
    # load_btn/export_btn/lang_radio 三個離散觸發點才重算一次，RAM/GPU
    # 數字在觸發點之間會停在舊值、看起來像卡住。做法比照 monitor_html
    # 已經驗證過的模式：重用同一顆 monitor_timer，多掛一個 tick 目標即可，
    # 不需要新的計時器或新的運算邏輯。勾選框關閉 monitor_timer 時，這裡
    # 也會一併暫停，行為跟監控面板保持一致。
    monitor_timer.tick(
        fn=get_status_bar_html,
        inputs=[state_last_export_time, state_lang],
        outputs=[status_bar_out],
    )
    monitor_auto_refresh.change(
        fn=lambda enabled: gr.Timer(value=3, active=bool(enabled)),
        inputs=[monitor_auto_refresh],
        outputs=[monitor_timer],
    )

    # ── Language radio ──────────────────────────────────────────
    def lang_radio_change(radio_val, is_focused, load_status_val, cfg_status_val, preview_status_val, compare_html_val,
                           snap_a_status_val, snap_b_status_val, snap_c_status_val):
        lang = "zh" if radio_val == "中文" else "en"
        return make_change_lang(lang)(is_focused, load_status_val, cfg_status_val, preview_status_val, compare_html_val,
                                       snap_a_status_val, snap_b_status_val, snap_c_status_val)

    lang_radio.change(
        fn=lang_radio_change,
        inputs=[lang_radio, focus_mode, load_status, cfg_status, preview_status, compare_slider_html,
                snap_a_status, snap_b_status, snap_c_status],
        outputs=[state_lang] + [comp for comp, _ in TRANSLATED_COMPONENTS_MAP]
                + [load_status, cfg_status, preview_status, compare_slider_html,
                   snap_a_status, snap_b_status, snap_c_status]
    ).then(
        fn=get_status_bar_html,
        inputs=[state_last_export_time, state_lang],
        outputs=[status_bar_out]
    )

    # ── Theme checkbox (Light/Dark) ────────────────────────────
    def toggle_theme_fn(is_light):
        label = "🌙 Dark Mode" if is_light else "☀ Light Mode"
        return gr.update(label=label)

    theme_checkbox.change(
        fn=toggle_theme_fn,
        inputs=[theme_checkbox],
        outputs=[theme_checkbox],
        js="""(v) => {
            var h = document.querySelector('html');
            var b = document.querySelector('body');
            if (v) {
                if (h) { h.classList.add('light-theme'); h.classList.remove('dark'); }
                if (b) { b.classList.add('light-theme'); b.classList.remove('dark'); }
            } else {
                if (h) { h.classList.remove('light-theme'); h.classList.add('dark'); }
                if (b) { b.classList.remove('light-theme'); b.classList.add('dark'); }
            }
            return v;
        }"""
    )


    # ── Compare mode toggle (並排 ↔ 滑桿疊圖) ────────────────
    def toggle_compare_mode(mode):
        # 支援中英文選項：並排顯示 / Side by Side → 並排；其餘 → 滑桿
        is_slider = mode not in ("並排顯示", "Side by Side")
        return (
            gr.update(visible=not is_slider),   # side_by_side_row
            gr.update(visible=is_slider),         # compare_slider_html
        )

    compare_mode.change(
        fn=toggle_compare_mode,
        inputs=[compare_mode],
        outputs=[side_by_side_row, compare_slider_html],
    )

    # ── Focus mode toggle ─────────────────────────────────────

    def toggle_focus(is_focused, lang):
        new_focused = not is_focused
        if new_focused:
            btn_label = "⬅ 展開面板" if lang == "zh" else "⬅ Show Panels"
        else:
            btn_label = "⛶ 專注預覽" if lang == "zh" else "⛶ Focus Preview"
        return (
            gr.update(visible=not new_focused),  # left_col
            gr.update(visible=not new_focused),  # right_col
            new_focused,                          # focus_mode state
            gr.update(value=btn_label),           # focus_btn label
        )

    focus_btn.click(
        fn=toggle_focus,
        inputs=[focus_mode, state_lang],
        outputs=[left_col, right_col, focus_mode, focus_btn],
    )

    # ── Close button (關閉程式) ───────────────────────────────
    def _shutdown():
        """關閉整個程序（含 Gradio/uvicorn 伺服器）。
        延遲 0.8 秒讓 Gradio 有時間把回應送回瀏覽器後再結束。"""
        def _do_exit():
            import time
            time.sleep(0.8)
            os._exit(0)
        t = threading.Thread(target=_do_exit, daemon=True)
        t.start()
        return gr.update(value="⏳ 正在關閉…", interactive=False)

    close_btn.click(
        fn=_shutdown,
        inputs=[],
        outputs=[close_btn],
        js="""() => {
            setTimeout(() => {
                document.body.innerHTML = `
                    <div style="
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: center;
                        height: 100vh;
                        background: radial-gradient(circle at center, #13132a 0%, #090913 100%);
                        color: #ddddf0;
                        font-family: 'Inter', system-ui, sans-serif;
                        text-align: center;
                    ">
                        <div style="
                            font-size: 64px;
                            margin-bottom: 20px;
                            animation: pulse 2s infinite ease-in-out;
                        ">🌌</div>
                        <h1 style="
                            font-size: 28px;
                            font-weight: 700;
                            margin-bottom: 12px;
                            background: linear-gradient(135deg, #a89eff 0%, #00d9ff 100%);
                            -webkit-background-clip: text;
                            -webkit-text-fill-color: transparent;
                            background-clip: text;
                        ">Astro Processor Pro</h1>
                        <p style="font-size: 16px; color: #7878a8; margin-bottom: 8px;">
                            程式已安全關閉 / Program safely shut down.
                        </p>
                        <p style="font-size: 13px; color: #38384e;">
                            您可以安全地關閉此瀏覽器分頁 / You can now safely close this browser tab.
                        </p>
                        <style>
                            @keyframes pulse {
                                0%, 100% { transform: scale(1); opacity: 0.9; }
                                50% { transform: scale(1.05); opacity: 1; }
                            }
                        </style>
                    </div>
                `;
                try {
                    window.close();
                } catch (e) {
                    console.log('window.close被瀏覽器安全性原則阻擋:', e);
                }
            }, 600);
        }"""
    )


if __name__ == "__main__":
    allowed_dir = _default_output_base()
    os.makedirs(allowed_dir, exist_ok=True)
    demo.launch(
        inbrowser=True,
        allowed_paths=[allowed_dir],
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.purple,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=PRO_CSS,
        js=custom_js,
    )
