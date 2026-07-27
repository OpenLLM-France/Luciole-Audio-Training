import psutil

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 ** 2  # Memory in MB
