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


def apply_local_target_boost(img01, enable, strength, radius, sensitivity):
    """在自動偵測到的目標區域(星雲/銀河等)內加強局部對比，天空背景則幾乎不受影響。

    做法：對亮度做一次大半徑的 unsharp-mask(clarity 概念)，但套用強度依 detect_target_regions()
    算出的遮罩逐像素加權——遮罩值高(有結構的區域)才吃到明顯的對比增強，
    遮罩值接近 0 的平坦天空背景幾乎不變，藉此取代「整張圖統一拉伸」的做法。
    色彩維持方式與 remove_background_gradient() 相同：只算亮度的增減比例，
    再乘回三個色版，避免整體色偏。
    """
    if not enable or strength <= 0:
        return img01, None

    mask = detect_target_regions(img01, radius=radius, sensitivity=sensitivity)
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
    return (np.clip(rgb01, 0, 1) * 255).astype(np.uint8)



def apply_clarity_and_sharpen(img8, clarity_blur, clarity_strength, sharpen_blur, sharpen_amount):
    img_f = img8.astype(np.float32)
    blur = cv2.GaussianBlur(img_f, (0, 0), sigmaX=clarity_blur)
    clarity = img_f + (img_f - blur) * clarity_strength
    clarity = np.clip(clarity, 0, 255).astype(np.uint8)

    blur2 = cv2.GaussianBlur(clarity, (0, 0), sigmaX=sharpen_blur)
    sharpened = cv2.addWeighted(clarity, sharpen_amount, blur2, 1 - sharpen_amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def denoise(img8, enable, mode, d, sigma_color, sigma_space, nlm_h, nlm_h_color):
    """降噪。

    mode = "fast"（預設）：雙邊濾波(bilateralFilter)，速度快，適合即時預覽與一般使用。
    mode = "quality"：cv2.fastNlMeansDenoisingColored，屬於 Non-local Means 演算法，
        會在整張圖搜尋相似的小區塊來平均，降噪效果通常比雙邊濾波乾淨、更能保留細節邊緣，
        但運算量遠高於雙邊濾波（一般會慢上數倍到十倍以上），較適合最終高解析度匯出而非即時預覽。
    """
    if not enable:
        return img8
    if mode == "quality":
        return cv2.fastNlMeansDenoisingColored(
            img8, None,
            h=float(nlm_h), hColor=float(nlm_h_color),
            templateWindowSize=7, searchWindowSize=21,
        )
    return cv2.bilateralFilter(img8, d=int(round(d)), sigmaColor=sigma_color, sigmaSpace=sigma_space)


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
    """
    # Step 1: 先降噪——保護微弱星雲邊界
    img_work = img01  # float32 [0, 1]
    if p['denoise_enable']:
        img8_tmp = (np.clip(img_work, 0, 1) * 255).astype(np.uint8)
        img8_tmp = denoise(
            img8_tmp, enable=True,
            mode=p.get('denoise_mode', 'fast'),
            d=p['denoise_d'],
            sigma_color=p['denoise_sigma_color'],
            sigma_space=p['denoise_sigma_space'],
            nlm_h=p.get('denoise_nlm_h', 10),
            nlm_h_color=p.get('denoise_nlm_h_color', 10),
        )
        img_work = img8_tmp.astype(np.float32) / 255.0

    # Step 2: Clarity + 銳化——在乾淨影像上提升局部對比
    img8_cs = (np.clip(img_work, 0, 1) * 255).astype(np.uint8)
    img8_cs = apply_clarity_and_sharpen(
        img8_cs,
        p['clarity_blur'], p['clarity_strength'],
        p['sharpen_blur'], p['sharpen_amount']
    )
    img_work = img8_cs.astype(np.float32) / 255.0

    # Step 3: 飽和度 / 明度 / 通道增益——最後調色，避免色偏擴散
    img8 = boost_saturation(
        img_work,
        p['sat_boost'], p['bright_boost'],
        p['r_gain'], p['g_gain'], p['b_gain']
    )
    return img8


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
    
    result = {
        'main': main_out,
        'mask': star_mask,
        'starless': starless_out,
        'target_mask_overlay': build_target_mask_overlay(main_out, target_mask),
    }
    return result


def save_image_files(img8, out_dir, name):
    out_jpg = os.path.join(out_dir, f"{name}.jpg")
    out_tif = os.path.join(out_dir, f"{name}.tif")
    cv2.imwrite(out_jpg, cv2.cvtColor(img8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 96])
    img16 = (img8.astype(np.uint16) * 257)
    cv2.imwrite(out_tif, cv2.cvtColor(img16, cv2.COLOR_RGB2BGR))
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
# 快照只存在瀏覽器工作階段(gr.State)中，不寫入磁碟，重新整理頁面就會清空——
# 若要長期保存，仍建議使用既有的「匯出當前參數(JSON)」功能。

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

# ============================================================
# ========================= 參數匯入/匯出 =========================
# ============================================================

def export_config_fn(*param_values):
    p = collect_params(param_values)
    os.makedirs("outputs", exist_ok=True)
    cfg_path = os.path.join("outputs", "astro_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(p, f, indent=4, ensure_ascii=False)
    return f"✅ 參數已成功匯出至：`outputs/astro_config.json`", cfg_path

def import_config_fn(file_obj):
    if file_obj is None:
        return ["⚠️ 請選擇要匯入的 JSON 檔案"] + [gr.update() for _ in PARAM_NAMES]
    try:
        path = file_obj if isinstance(file_obj, str) else file_obj.name
        with open(path, "r", encoding="utf-8") as f:
            p = json.load(f)
        
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


def update_preview_fn(preview_base, preview_scale, full_img, use_full_res, lang, *param_values):
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
        main_out = result['main']
        hist_out = generate_histogram(main_out)
        original_uint8 = (np.clip(img_to_use, 0, 1) * 255).astype(np.uint8)
        tag = ("全解析度原圖運算" if lang == "zh" else "Full-res image processing") if use_full else ("縮圖運算" if lang == "zh" else "Thumbnail processing")
        slider_html = build_compare_slider_html(original_uint8, main_out, lang)
        status_msg = f"✅ 預覽與 RGB 曲線已更新({tag}，{elapsed:.2f}s)" if lang == "zh" else f"✅ Preview and RGB histogram updated ({tag}, {elapsed:.2f}s)"
        return main_out, original_uint8, hist_out, status_msg, slider_html
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
        starless = result.get('starless')
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
        processed_uint8 = result['main']

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


def export_fn(full_img, out_dir, out_name, want_layers, lang, *param_values):

    if full_img is None:
        msg = "⚠️ 請先載入圖片" if lang == "zh" else "⚠️ Please load an image first"
        return msg, None
    if not out_name:
        out_name = "processed"
    p = collect_params(param_values)
    try:
        os.makedirs(out_dir, exist_ok=True)
        # 全解析度匯出：preview_scale=1.0，星點縮小/去星參數直接採用使用者設定的原始數值
        result = run_pipeline(full_img, p, want_layers=want_layers, preview_scale=1.0)
        files = []
        jpg_path, tif_path = save_image_files(result['main'], out_dir, out_name)
        files += [jpg_path, tif_path]
        if want_layers and 'mask' in result:
            mask_path = os.path.join(out_dir, f"{out_name}_starmask.png")
            cv2.imwrite(mask_path, result['mask'])
            files.append(mask_path)
            sl_jpg, sl_tif = save_image_files(result['starless'], out_dir, f"{out_name}_starless")
            files += [sl_jpg, sl_tif]
        msg = f"✅ 匯出完成，共 {len(files)} 個檔案 → `{out_dir}`" if lang == "zh" else f"✅ Export completed, {len(files)} files → `{out_dir}`"
        return msg, files
    except Exception as e:
        err_msg = f"❌ 匯出失敗: {e}" if lang == "zh" else f"❌ Export failed: {e}"
        return err_msg, None


def batch_process_fn(folder, out_dir, want_layers, lang, *param_values):
    """批次處理：對資料夾內每一張圖套用目前的參數集，逐一以全解析度處理並匯出。

    設計成 generator（每處理完一張就 yield 一次最新狀態文字），這樣 Gradio 前端
    可以即時顯示進度，而不用等全部檔案跑完才有畫面回饋——批次動輒數十張全解析度圖，
    单次處理可能要好幾秒到幾十秒，沒有進度回饋的話使用者會誤以為卡死。

    單張圖失敗（例如檔案損毀、記憶體不足）不會中斷整批次，只會記錄錯誤並繼續下一張，
    最後在摘要裡列出成功/失敗數量與失敗檔名，方便使用者事後排查。
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
        out_dir = "outputs/batch"
    os.makedirs(out_dir, exist_ok=True)
    p = collect_params(param_values)

    total = len(files)
    done, failed = 0, []
    log_lines = []

    header = (f"🚀 批次處理開始，共 {total} 張圖片 → 輸出至 `{out_dir}`"
               if lang == "zh" else
               f"🚀 Batch started, {total} image(s) → output to `{out_dir}`")
    yield header

    for i, fname in enumerate(files, 1):
        base_name = os.path.splitext(fname)[0]
        try:
            img = load_image_any(os.path.join(folder, fname))
            result = run_pipeline(img, p, want_layers=want_layers, preview_scale=1.0)
            save_image_files(result['main'], out_dir, base_name)
            if want_layers and 'mask' in result:
                cv2.imwrite(os.path.join(out_dir, f"{base_name}_starmask.png"), result['mask'])
                save_image_files(result['starless'], out_dir, f"{base_name}_starless")
            done += 1
            log_lines.append(f"✅ [{i}/{total}] {fname}")
        except Exception as e:
            failed.append(fname)
            log_lines.append(f"❌ [{i}/{total}] {fname}: {e}")

        progress_tail = "\n".join(log_lines[-8:])  # 只顯示最近 8 行，避免訊息無限增長
        status = (f"{header}\n\n⏳ 進度: {i}/{total}（成功 {done}，失敗 {len(failed)}）\n\n{progress_tail}"
                   if lang == "zh" else
                   f"{header}\n\n⏳ Progress: {i}/{total} (success {done}, failed {len(failed)})\n\n{progress_tail}")
        yield status

    if failed:
        fail_list = "、".join(failed) if lang == "zh" else ", ".join(failed)
        summary = (f"🏁 批次處理完成：成功 {done} 張，失敗 {len(failed)} 張 → `{out_dir}`\n\n失敗檔案：{fail_list}"
                    if lang == "zh" else
                    f"🏁 Batch finished: {done} succeeded, {len(failed)} failed → `{out_dir}`\n\nFailed files: {fail_list}")
    else:
        summary = (f"🏁 批次處理完成，全部 {done} 張圖片皆已成功匯出 → `{out_dir}`"
                    if lang == "zh" else
                    f"🏁 Batch finished, all {done} image(s) exported successfully → `{out_dir}`")
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
    "sat_boost": {
        "zh": ("飽和度倍率", "銀河/星雲通常需 1.2-1.8 倍增益"),
        "en": ("Saturation Boost Factor", "Milky Way/nebula shots typically need a 1.2-1.8x boost"),
    },
    "bright_boost": {"zh": "明度倍率", "en": "Brightness Boost Factor"},
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
    "sharpen_blur": {"zh": "銳化模糊半徑", "en": "Sharpen Blur Radius"},
    "sharpen_amount": {
        "zh": ("銳化程度", "過大會使雜訊粒子變粗"),
        "en": ("Sharpen Amount", "Too high will make noise grain coarser"),
    },
    "denoise_enable": {"zh": "啟用降噪", "en": "Enable Denoise"},
    "denoise_mode": {
        "zh": ("降噪模式", "fast=雙邊濾波(快); quality=Non-local Means(較乾淨但慢很多，適合最終匯出)"),
        "en": ("Denoise Mode", "fast = Bilateral Filter (quick); quality = Non-local Means (cleaner but much slower, best for final export)"),
    },
    "denoise_d": {
        "zh": ("[fast]濾波視窗", "較大數值降噪範圍廣但耗時"),
        "en": ("[fast] Bilateral Filter Diameter (d)", "Larger values denoise a wider area but take longer"),
    },
    "denoise_sigma_color": {"zh": "[fast]顏色 Sigma", "en": "[fast] Denoise Sigma Color"},
    "denoise_sigma_space": {"zh": "[fast]空間 Sigma", "en": "[fast] Denoise Sigma Space"},
    "denoise_nlm_h": {
        "zh": ("[quality]亮度降噪強度 h", "越高越乾淨，但可能抹掉細節"),
        "en": ("[quality] Luminance Denoise Strength (h)", "Higher = cleaner but may erase detail"),
    },
    "denoise_nlm_h_color": {"zh": "[quality]色彩降噪強度 hColor", "en": "[quality] Color Denoise Strength (hColor)"},
    "star_mode": {"zh": "模式", "en": "Star Processing Mode"},
    "star_kernel": {
        "zh": ("偵測核大小(≈星點直徑px)", "應略大於想抓取的中小型星點直徑"),
        "en": ("Star Detection Kernel Size (px)", "Should be slightly larger than the diameter of the small/medium stars you want to detect"),
    },
    "star_thresh": {
        "zh": ("偵測門檻", "越低暗星越多，過低會誤抓背景熱雜訊"),
        "en": ("Star Detection Threshold", "Lower = more faint stars detected; too low will pick up background hot-pixel noise"),
    },
    "star_max_area": {"zh": "星點最大面積", "en": "Maximum Star Area (px²)"},
    "star_max_area_large": {"zh": "亮星暈光面積上限", "en": "Maximum Bright Star Halo Area (px²)"},
    "star_aspect": {
        "zh": ("圓度門檻(長寬比)", "排除長條形星雲結構"),
        "en": ("Star Aspect Ratio Threshold", "Excludes elongated nebula structures"),
    },
    "star_dilate": {"zh": "遮罩外擴基本像素", "en": "Mask Dilation Base px"},
    "star_dilate_scale": {"zh": "依星點大小外擴比例", "en": "Mask Dilation Size-dependent Scale"},
    "star_shrink_kernel": {"zh": "[縮星]侵蝕核大小", "en": "[Shrink] Erosion Kernel Size"},
    "star_shrink_iter": {"zh": "[縮星]侵蝕次數", "en": "[Shrink] Erosion Iterations"},
    "star_shrink_strength": {"zh": "[縮星]套用強度", "en": "[Shrink] Apply Strength"},
    "star_inpaint_radius": {"zh": "[去星]單星取樣半徑", "en": "[Remove] Inpaint Radius (px)"},
    "star_feather_px": {"zh": "[去星]邊緣羽化程度", "en": "[Remove] Edge Feathering px"},
    "star_noise_strength": {
        "zh": ("[去星]雜訊回填強度", "0 為不回填"),
        "en": ("[Remove] Noise Infill Strength", "0 = no noise infill"),
    },
    "multiscale_enable": {"zh": "啟用大範圍偵測", "en": "Enable Multi-scale Star Detection"},
    "cluster_kernel": {"zh": "偵測核大小", "en": "Cluster Detection Kernel Size"},
    "cluster_thresh": {"zh": "偵測門檻", "en": "Cluster Detection Threshold"},
    "cluster_min_area": {"zh": "最小面積", "en": "Minimum Cluster Area (px²)"},
    "cluster_max_area": {"zh": "最大面積", "en": "Maximum Cluster Area (px²)"},
    "cluster_aspect": {"zh": "長寬比門檻", "en": "Cluster Aspect Ratio Threshold"},
    "cluster_dilate": {"zh": "遮罩外擴像素", "en": "Cluster Mask Dilation px"},
    "cluster_inpaint_radius": {"zh": "[去星]星團取樣半徑", "en": "[Remove] Cluster Inpaint Radius"},
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
    "export_btn": {"zh": "🚀 開始高解析度跑圖與匯出", "en": "🚀 Start High-Res Processing & Export"},
    "export_files": {"zh": "下載生成的結果檔案", "en": "Download Generated Results"},
    "preview_hist": {"zh": "RGB 通道分佈曲線", "en": "RGB Channel Histogram"},
    "mask_mini": {"zh": "星點遮罩(產生圖層後自動更新)", "en": "Star Mask (Auto updates after generating layers)"},
    "monitor_refresh_btn": {"zh": "🔄 更新監控資訊", "en": "🔄 Update System Monitor Info"},
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
    "local_x_pct": {"zh": "X 中心位置 (%)", "en": "X Center (%)"},
    "local_y_pct": {"zh": "Y 中心位置 (%)", "en": "Y Center (%)"},
    "local_crop_px": {"zh": "裁切大小 (px)", "en": "Crop Size (px)"},
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
    "batch_btn": {"zh": "🚀 開始批次處理整個資料夾", "en": "🚀 Start Batch Processing"},
    "close_btn": {"zh": "✕ 關閉", "en": "✕ Close"},
    "presets_hint_md": {
        "zh": "**🎛️ 新手預設集**：先選一個接近你拍攝情境的起點，再自行微調。",
        "en": "**🎛️ Beginner Presets**: Pick a starting point close to your shooting scenario, then fine-tune from there.",
    },
    "preset_milky_way_btn": {"zh": "🌌 銀河模式", "en": "🌌 Milky Way"},
    "preset_nebula_btn": {"zh": "🌫️ 星雲模式", "en": "🌫️ Nebula"},
    "preset_light_pollution_btn": {"zh": "🏙️ 重光害", "en": "🏙️ Heavy Light Pollution"},
    "snapshot_hint_md": {
        "zh": "**📌 參數快照**：先「儲存」目前這組參數到 A/B/C，之後可以隨時「套用」快速切回比較，不用重新調整滑桿。快照只暫存在這次瀏覽器工作階段中，重新整理頁面會清空。",
        "en": "**📌 Parameter Snapshots**: 'Save' the current parameter set to slot A/B/C, then 'Apply' any time to switch back instantly for comparison — no need to re-adjust sliders. Snapshots only live in this browser session and are cleared on page refresh.",
    },
    "snap_a_save_btn": {"zh": "💾 儲存為 A", "en": "💾 Save as A"},
    "snap_a_load_btn": {"zh": "📥 套用 A", "en": "📥 Apply A"},
    "snap_b_save_btn": {"zh": "💾 儲存為 B", "en": "💾 Save as B"},
    "snap_b_load_btn": {"zh": "📥 套用 B", "en": "📥 Apply B"},
    "snap_c_save_btn": {"zh": "💾 儲存為 C", "en": "💾 Save as C"},
    "snap_c_load_btn": {"zh": "📥 套用 C", "en": "📥 Apply C"},
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

    if HAS_GPUTIL:
        try:
            for gpu in _GPUtil.getGPUs():
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
        except Exception:
            pass
    elif USE_GPU and HAS_TORCH:
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
        except Exception:
            pass
    else:
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

    if HAS_GPUTIL:
        try:
            gpus = _GPUtil.getGPUs()
            if gpus:
                g = gpus[0]
                gpu_str = f"{g.load * 100:.0f}% | {g.memoryUsed / 1024:.1f} GB"
        except Exception:
            pass
    elif USE_GPU and HAS_TORCH:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_str = f"{torch.cuda.memory_allocated() / (1024**3):.1f} GB alloc"
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
                    with gr.Group(elem_id="presets-group"):
                        presets_hint_md = gr.Markdown("**🎛️ 新手預設集**：先選一個接近你拍攝情境的起點，再自行微調。")
                        with gr.Row():
                            preset_milky_way_btn = gr.Button("🌌 銀河模式", size="sm")
                            preset_nebula_btn    = gr.Button("🌫️ 星雲模式", size="sm")
                            preset_light_pollution_btn = gr.Button("🏙️ 重光害", size="sm")
                        preset_status = gr.Markdown("")

                    # ── 參數快照：A / B / C 三組快速比較 ─────────
                    with gr.Group(elem_id="snapshots-group"):
                        snapshot_hint_md = gr.Markdown(
                            "**📌 參數快照**：先「儲存」目前這組參數到 A/B/C，之後可以隨時「套用」快速切回比較，"
                            "不用重新調整滑桿。快照只暫存在這次瀏覽器工作階段中，重新整理頁面會清空。"
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

                        with gr.Accordion("4️⃣ 飽和度 / 明度 / RGB通道", open=True) as acc_sat:
                            sat_boost    = gr.Slider(0.5, 3,   value=1.45, step=0.05, label="飽和度倍率", info="銀河/星雲通常需 1.2-1.8 倍增益")
                            bright_boost = gr.Slider(0.5, 2,   value=1.03, step=0.01, label="明度倍率", info="整體亮度微調，通常接近 1.0 即可，避免過曝")
                            r_gain       = gr.Slider(0.5, 1.5, value=1.0,  step=0.01, label="🔴 紅色通道增益", info="加強發射星雲 H-alpha 訊號")
                            g_gain       = gr.Slider(0.5, 1.5, value=1.0,  step=0.01, label="🟢 綠色通道增益", info="通常用來壓低綠色夜空光害")
                            b_gain       = gr.Slider(0.5, 1.5, value=1.0,  step=0.01, label="🔵 藍色通道增益", info="加強反射星雲或藍色年輕恆星")

                        with gr.Accordion("5️⃣ Clarity(局部對比) / 銳化", open=False) as acc_clarity:
                            clarity_blur     = gr.Slider(1,   60,  value=25,   step=1,    label="Clarity 模糊半徑", info="越大越偏向中尺度結構")
                            clarity_strength = gr.Slider(0,   1,   value=0.35, step=0.01, label="Clarity 強度", info="類似 Lightroom 清晰度")
                            sharpen_blur     = gr.Slider(0.5, 10,  value=2,    step=0.1,  label="銳化模糊半徑", info="決定銳化鎖定的細節尺度，越大銳化範圍越粗")
                            sharpen_amount   = gr.Slider(0.5, 3,   value=1.25, step=0.01, label="銳化程度", info="過大會使雜訊粒子變粗")

                        with gr.Accordion("6️⃣ 降噪", open=False) as acc_denoise:
                            denoise_enable      = gr.Checkbox(label="啟用降噪", value=DEFAULTS[24])
                            denoise_mode        = gr.Radio(["fast", "quality"], value="fast", label="降噪模式", info="fast=雙邊濾波(快); quality=Non-local Means(較乾淨但慢很多，適合最終匯出)")
                            denoise_d           = gr.Slider(1, 15, value=5,  step=1, label="[fast]濾波視窗", info="較大數值降噪範圍廣但耗時")
                            denoise_sigma_color = gr.Slider(1, 50, value=15, step=1, label="[fast]顏色 Sigma", info="越高越能融合差異較大的顏色，但可能糊掉色彩邊界")
                            denoise_sigma_space = gr.Slider(1, 50, value=15, step=1, label="[fast]空間 Sigma", info="越高影響範圍越大的鄰近像素，降噪更平滑但更慢")
                            denoise_nlm_h       = gr.Slider(1, 30, value=10, step=1, label="[quality]亮度降噪強度 h", info="越高越乾淨，但可能抹掉細節")
                            denoise_nlm_h_color = gr.Slider(1, 30, value=10, step=1, label="[quality]色彩降噪強度 hColor", info="越高色彩雜訊越乾淨，但可能造成色塊化")

                        with gr.Accordion("7️⃣ 星點縮小 / 去星", open=False) as acc_star:
                            star_mode           = gr.Radio(["none","shrink","remove"], value="shrink", label="模式")
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
                    output_dir  = gr.Textbox(label="輸出資料夾",         value="outputs")
                    output_name = gr.Textbox(label="輸出檔名(不含副檔名)", value="processed")
                    save_layers = gr.Checkbox(
                        label="額外輸出星點遮罩 + 去星背景層(可供後續人工疊圖疊加)",
                        value=True
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
                    batch_out_dir = gr.Textbox(label="批次輸出資料夾", value="outputs/batch")
                    batch_want_layers = gr.Checkbox(
                        label="每張圖同時輸出星點遮罩 + 去星背景層", value=False
                    )
                    batch_btn = gr.Button("🚀 開始批次處理整個資料夾", variant="primary")
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
                monitor_refresh_btn = gr.Button("🔄 更新監控資訊", size="sm")

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
    ]
    toggles_for_change = [bg_enable, wb_enable, local_target_enable, denoise_enable, denoise_mode, star_mode, multiscale_enable]

    # Preview common inputs / outputs
    _PREV_IN  = [state_preview_base, state_preview_scale, state_full,
                 use_full_res_preview, state_lang] + PARAM_COMPONENTS
    _PREV_OUT = [preview_image, original_preview_image, preview_hist, preview_status, compare_slider_html]


    # ── Language Translation Handlers ──────────────────────────
    TRANSLATED_COMPONENTS_MAP = [
        (folder_box, "folder_box"),
        (scan_btn, "scan_btn"),
        (file_dropdown, "file_dropdown"),
        (upload_file, "upload_file"),
        (preview_size, "preview_size"),
        (use_full_res_preview, "use_full_res_preview"),
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
        (sat_boost, "sat_boost"),
        (bright_boost, "bright_boost"),
        (r_gain, "r_gain"),
        (g_gain, "g_gain"),
        (b_gain, "b_gain"),
        (clarity_blur, "clarity_blur"),
        (clarity_strength, "clarity_strength"),
        (sharpen_blur, "sharpen_blur"),
        (sharpen_amount, "sharpen_amount"),
        (denoise_enable, "denoise_enable"),
        (denoise_mode, "denoise_mode"),
        (denoise_d, "denoise_d"),
        (denoise_sigma_color, "denoise_sigma_color"),
        (denoise_sigma_space, "denoise_sigma_space"),
        (denoise_nlm_h, "denoise_nlm_h"),
        (denoise_nlm_h_color, "denoise_nlm_h_color"),
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
        (export_btn, "export_btn"),
        (export_files, "export_files"),
        (preview_hist, "preview_hist"),
        (mask_mini, "mask_mini"),
        (monitor_refresh_btn, "monitor_refresh_btn"),
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
        (batch_btn, "batch_btn"),
        (close_btn, "close_btn"),
        (presets_hint_md, "presets_hint_md"),
        (preset_milky_way_btn, "preset_milky_way_btn"),
        (preset_nebula_btn, "preset_nebula_btn"),
        (preset_light_pollution_btn, "preset_light_pollution_btn"),
        (snapshot_hint_md, "snapshot_hint_md"),
        (snap_a_save_btn, "snap_a_save_btn"),
        (snap_a_load_btn, "snap_a_load_btn"),
        (snap_b_save_btn, "snap_b_save_btn"),
        (snap_b_load_btn, "snap_b_load_btn"),
        (snap_c_save_btn, "snap_c_save_btn"),
        (snap_c_load_btn, "snap_c_load_btn"),
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
                             "local_preview_btn", "batch_btn", "close_btn",
                             "preset_milky_way_btn", "preset_nebula_btn", "preset_light_pollution_btn",
                             "snap_a_save_btn", "snap_a_load_btn", "snap_b_save_btn", "snap_b_load_btn",
                             "snap_c_save_btn", "snap_c_load_btn"]:
                    updates.append(gr.update(value=label_txt))
                elif key in ["cfg_header_md", "layer_hint_md", "local_hint_md", "batch_hint_md",
                             "presets_hint_md", "snapshot_hint_md"]:
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
        inputs=[state_lang],
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
        inputs=[state_full, output_dir, output_name, save_layers, state_lang] + PARAM_COMPONENTS,
        outputs=[export_status, export_files],
    ).then(
        fn=get_status_bar_html,
        inputs=[state_lang],
        outputs=[status_bar_out],
    )

    # ── Batch processing (reuses folder_box from the load tab as source) ──
    batch_btn.click(
        fn=batch_process_fn,
        inputs=[folder_box, batch_out_dir, batch_want_layers, state_lang] + PARAM_COMPONENTS,
        outputs=[batch_status],
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
        inputs=PARAM_COMPONENTS,
        outputs=[cfg_status, cfg_download],
    )

    cfg_import_file.change(
        fn=import_config_fn,
        inputs=[cfg_import_file],
        outputs=[cfg_status] + PARAM_COMPONENTS,
    ).then(
        fn=update_preview_fn, inputs=_PREV_IN, outputs=_PREV_OUT,
    )

    # ── System monitor refresh ────────────────────────────────
    monitor_refresh_btn.click(
        fn=get_system_stats_html,
        outputs=[monitor_html],
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
        inputs=[state_lang],
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
    allowed_dir = os.path.abspath("outputs")
    os.makedirs(allowed_dir, exist_ok=True)
    demo.launch(
        inbrowser=True,
        allowed_paths=[allowed_dir, "outputs"],
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.purple,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=PRO_CSS,
        js=custom_js,
    )
