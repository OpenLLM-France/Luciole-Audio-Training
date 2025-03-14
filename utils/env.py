import os
import sys
import tempfile
import py3nvml.py3nvml as pynvml  # For GPU (VRAM) monitoring
import psutil  # For RAM monitoring
import multiprocessing

# Global flags and variables for GPU configuration
DISABLE_GPU = False
REQUIRED_GPU = []
ALL_GPU_INDICES = None  # Global placeholder

# Set GPU ordering and PyTorch settings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_num_gpus(ignore_errors=False):
    """
    Returns the number of GPUs available.
    """
    global ALL_GPU_INDICES
    if ALL_GPU_INDICES is not None:
        return len(ALL_GPU_INDICES)
    try:
        pynvml.nvmlInit()  # May throw if the driver is not loaded
    except pynvml.NVMLError_DriverNotLoaded:
        import torch
        if torch.cuda.is_available():
            raise RuntimeError("CUDA is available but pynvml reports the driver is not loaded. "
                               "If you are using a conda environment, try installing nvidia-smi there.")
        return 0
    except Exception as unexpected_error:
        if ignore_errors:
            return 0
        raise unexpected_error
    num_gpus = pynvml.nvmlDeviceGetCount()
    ALL_GPU_INDICES = list(range(num_gpus))
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        ALL_GPU_INDICES = [int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    return len(ALL_GPU_INDICES)

def vram_free(index=0):
    """
    Returns the free GPU memory (in MiB) for the GPU at the given index.
    """
    handle = _get_gpu_handle(index)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.free // 1024**2

def _get_gpu_handle(index=0):
    if get_num_gpus() == 0:
        raise RuntimeError("No GPU available")
    pynvml.nvmlInit()
    try:
        return pynvml.nvmlDeviceGetHandleByIndex(index)
    except Exception as e:
        raise RuntimeError(f"Could not access GPU at index {index}: {e}")

def _set_visible_gpus(s):
    """
    Sets the CUDA_VISIBLE_DEVICES environment variable based on the input.
    Accepts a string (e.g. "auto", "none", "0,1") or a list of GPU indices.
    """
    global DISABLE_GPU, REQUIRED_GPU
    if isinstance(s, str):
        if s.lower() == "auto":
            num = get_num_gpus()
            if num == 0:
                s = ""
            else:
                # Choose the GPU with the most free memory
                gpus = sorted(range(num), key=vram_free, reverse=True)
                s = str(gpus[0])
        elif s.lower() == "none":
            s = ""
        else:
            s = s.split(",") if s else []
    if isinstance(s, list):
        s = [int(si) for si in s]
        REQUIRED_GPU = list(range(len(s)))
        s = ','.join(str(si) for si in s)
    if not s:
        DISABLE_GPU = True
    os.environ["CUDA_VISIBLE_DEVICES"] = s

def parse_gpu_args():
    """
    Parses command-line arguments to set GPU configurations.
    Usage examples:
      --gpu auto
      --gpus=0,1
    """
    has_set_gpu = False
    for i, arg in enumerate(sys.argv[1:]):
        lower_arg = arg.lower()
        if lower_arg in ["--gpus", "--gpu"]:
            if i + 2 <= len(sys.argv):
                _set_visible_gpus(sys.argv[i + 2])
            has_set_gpu = True
        elif lower_arg.startswith("--gpus=") or lower_arg.startswith("--gpu="):
            _set_visible_gpus(arg.split("=")[-1])
            has_set_gpu = True
    if not has_set_gpu:
        _set_visible_gpus("auto")

def auto_device():
    """
    Returns a torch.device based on GPU availability.
    """
    import torch
    return torch.device('cuda:0') if (torch.cuda.is_available() and not DISABLE_GPU) else torch.device("cpu")

def use_gpu():
    """
    Returns the list of GPU indices to be used.
    """
    import torch
    if DISABLE_GPU or not torch.cuda.is_available():
        assert REQUIRED_GPU == [], f"GPU required but not available (required GPU: {REQUIRED_GPU})"
        return []
    num_gpus = get_num_gpus()
    if REQUIRED_GPU:
        assert num_gpus - 1 >= max(REQUIRED_GPU), f"More GPU required than available (required GPU: {REQUIRED_GPU}, available GPUs: {list(range(num_gpus))})"
    return REQUIRED_GPU

# --- Cache Directory Setup ---

# Set environment variables for NUMBA and TensorFlow warnings
os.environ["NUMBA_CACHE_DIR"] = "/tmp"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Ensure HOME is set; if not, try to determine it
if not os.environ.get("HOME"):
    path = os.path.dirname(os.path.abspath(__file__))
    if path.startswith("/home/"):
        os.environ["HOME"] = "/".join(path.split("/")[:3])
    else:
        os.environ["HOME"] = os.path.expanduser("~")

def get_cache_dir(name=None):
    """
    Returns the cache directory for a given library name.
    """
    cache_dir = tempfile.gettempdir()
    if os.environ.get("HOME") and os.access(os.environ["HOME"], os.W_OK):
        cache_dir = os.path.join(os.environ["HOME"], ".cache")
    elif __file__.startswith("/home/") and os.access("/".join(__file__.split("/")[:3]), os.W_OK):
        cache_dir = os.path.join("/".join(__file__.split("/")[:3]), ".cache")
    else:
        for folder in ["/usr/share", "/workspace", "/opt"]:
            if os.access(folder, os.W_OK):
                cache_dir = os.path.join(folder, ".cache")
                break
    if name:
        cache_dir = os.path.join(cache_dir, name)
    return cache_dir

# Set cache directories for Hugging Face libraries and datasets
os.environ["HUGGINGFACE_HUB_CACHE"] = get_cache_dir("huggingface/hub")
os.environ["HF_HOME"] = get_cache_dir("huggingface/hub")
os.environ["TRANSFORMERS_CACHE"] = get_cache_dir("huggingface/hub")

import datasets
datasets.config.HF_MODULES_CACHE = get_cache_dir("huggingface/modules")
datasets.config.HF_DATASETS_CACHE = get_cache_dir("huggingface/datasets")
datasets.config.HF_METRICS_CACHE = get_cache_dir("huggingface/metrics")
datasets.config.DOWNLOADED_DATASETS_PATH = get_cache_dir("huggingface/datasets/downloads")

import torch

# --- Main execution ---

if __name__ == "__main__":
    parse_gpu_args()
    gpu_used = use_gpu()
    if gpu_used:
        print("Using GPUs:", gpu_used)
    else:
        print("No GPU will be used; switching to CPU.")
        torch.set_num_threads(multiprocessing.cpu_count())
    
    print("Auto-selected device:", auto_device())
    print("Cache directory:", get_cache_dir())
