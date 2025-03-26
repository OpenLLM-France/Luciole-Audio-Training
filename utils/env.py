import os
import sys
import tempfile
import multiprocessing
import py3nvml.py3nvml as pynvml  # VRAM (GPU) monitoring
import psutil  # RAM monitoring
import datasets
import torch

# Global Variables
DISABLE_GPU = False
REQUIRED_GPU = []
ALL_GPU_INDICES = None

# Environment Configurations
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NUMBA_CACHE_DIR"] = "/tmp"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# GPU Management Functions
def get_num_gpus(ignore_errors=False):
    """Returns the number of available GPUs."""
    global ALL_GPU_INDICES
    if ALL_GPU_INDICES is not None:
        return len(ALL_GPU_INDICES)
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError_DriverNotLoaded:
        if torch.cuda.is_available():
            raise RuntimeError("CUDA available but NVML driver not loaded. Install nvidia-smi.")
        return 0
    except Exception as e:
        if ignore_errors:
            return 0
        raise e
    
    num_gpus = pynvml.nvmlDeviceGetCount()
    ALL_GPU_INDICES = list(range(num_gpus))
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        ALL_GPU_INDICES = [int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    return len(ALL_GPU_INDICES)


def vram_free(index=0):
    """Returns the free GPU memory (in MiB) for the given GPU index."""
    handle = _get_gpu_handle(index)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.free // 1024**2

def has_gpu():
    """
        Returns True if GPU is available
    """
    return get_num_gpus() > 0

def _get_gpu_handle(index=0):
    """Returns the GPU handle for the given index."""
    if not has_gpu():
        raise RuntimeError("No GPU available")
    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetHandleByIndex(index)


def _set_visible_gpus(s):
    """Sets the visible GPUs based on input string or list."""
    global DISABLE_GPU, REQUIRED_GPU
    if isinstance(s, str):
        if s.lower() == "auto":
            gpus = sorted(range(get_num_gpus()), key=vram_free, reverse=True)
            s = str(gpus[0]) if gpus else ""
        elif s.lower() == "none":
            s = ""
        return _set_visible_gpus(s.split(",") if s else [])
    if isinstance(s, list):
        s = [int(si) for si in s]
        REQUIRED_GPU = list(range(len(s)))
        s = ','.join(map(str, s))
    if not s:
        DISABLE_GPU = True
    os.environ["CUDA_VISIBLE_DEVICES"] = s


# Command-line GPU Argument Handling
has_set_gpu = False
for i, arg in enumerate(sys.argv[1:]):
    arg = arg.lower()
    if arg in ["--gpus", "--gpu"]:
        _set_visible_gpus(sys.argv[i+2])
        has_set_gpu = True
    elif arg.startswith("--gpus=") or arg.startswith("--gpu="):
        _set_visible_gpus(arg.split("=")[-1])
        has_set_gpu = True
if not has_set_gpu:
    _set_visible_gpus("auto")


if not os.environ.get("HOME"):
    path = os.path.dirname(os.path.abspath(__file__))
    if path.startswith("/home/"):
        os.environ["HOME"] = "/".join(os.environ["HOME"].split("/")[:3])
        
# Cache Directory Handling
def get_cache_dir(name=None):
    """Returns the cache directory for the given library name."""
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
    return os.path.join(cache_dir, name) if name else cache_dir


# Set Hugging Face Cache Directories
os.environ["HUGGINGFACE_HUB_CACHE"] = get_cache_dir("huggingface/hub")
os.environ["HF_HOME"] = get_cache_dir("huggingface/hub")
os.environ["TRANSFORMERS_CACHE"] = get_cache_dir("huggingface/hub")
datasets.config.HF_MODULES_CACHE = get_cache_dir("huggingface/modules")
datasets.config.HF_DATASETS_CACHE = get_cache_dir("huggingface/datasets")
datasets.config.HF_METRICS_CACHE = get_cache_dir("huggingface/metrics")
datasets.config.DOWNLOADED_DATASETS_PATH = get_cache_dir("huggingface/datasets/downloads")


# Device Handling Functions
def auto_device():
    """Returns the best available device (GPU or CPU)."""
    return torch.device('cuda:0') if (torch.cuda.is_available() and not DISABLE_GPU) else torch.device("cpu")


def use_gpu():
    """Determines whether to use GPU and verifies availability."""
    if DISABLE_GPU or not torch.cuda.is_available():
        assert REQUIRED_GPU == [], "GPU required but not available"
        return []
    num_gpus = get_num_gpus()
    if REQUIRED_GPU:
        assert num_gpus - 1 >= max(REQUIRED_GPU), "More GPUs required than available"
    return REQUIRED_GPU

if not use_gpu():
    torch.set_num_threads(multiprocessing.cpu_count())
