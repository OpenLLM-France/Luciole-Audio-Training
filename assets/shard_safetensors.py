import argparse
import json
import re
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

UNITS = {"KB": 10**3, "MB": 10**6, "GB": 10**9, "KIB": 2**10, "MIB": 2**20, "GIB": 2**30}


def parse_size(s):
    m = re.fullmatch(r"\s*([\d.]+)\s*([A-Za-z]+)\s*", s)
    if not m or m.group(2).upper() not in UNITS:
        raise argparse.ArgumentTypeError(f"Invalid size '{s}', expected e.g. 5GB, 500MB")
    return int(float(m.group(1)) * UNITS[m.group(2).upper()])


def plan_shards(sizes, max_bytes):
    """Greedily group tensor names (in file order) into shards of at most max_bytes.
    A tensor larger than max_bytes gets a shard of its own."""
    shards, current, current_size = [], [], 0
    for name, size in sizes.items():
        if current and current_size + size > max_bytes:
            shards.append(current)
            current, current_size = [], 0
        current.append(name)
        current_size += size
    if current:
        shards.append(current)
    return shards


def main():
    parser = argparse.ArgumentParser(
        description="Shard a large .safetensors file into model-XXXXX-of-XXXXX.safetensors + model.safetensors.index.json (HF format)"
    )
    parser.add_argument("input", type=Path, help="Path to the safetensors file to shard, e.g. .../model.safetensors")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: same directory as the input file)")
    parser.add_argument("--max-shard-size", type=parse_size, default="5GB", help="Max size of each shard, e.g. 5GB, 500MB (default: 5GB)")
    parser.add_argument("--prefix", default="model", help="Shard file name prefix (default: model)")
    parser.add_argument("--remove-original", action="store_true", help="Delete the input file once all shards are written")
    args = parser.parse_args()

    max_bytes = args.max_shard_size if isinstance(args.max_shard_size, int) else parse_size(args.max_shard_size)
    input_path = args.input
    if not input_path.is_file():
        raise FileNotFoundError(f"File not found: {input_path}")
    out_dir = args.output_dir or input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tensors are read lazily, so only one shard is held in memory at a time
    with safe_open(str(input_path), framework="pt") as f:
        metadata = f.metadata() or {}
        sizes = {}
        for name in f.keys():
            sl = f.get_slice(name)
            numel = 1
            for d in sl.get_shape():
                numel *= d
            sizes[name] = numel * _dtype_bytes(sl.get_dtype())

        total_size = sum(sizes.values())
        shards = plan_shards(sizes, max_bytes)
        n = len(shards)
        if n == 1:
            print(f"Total size {total_size / 1e9:.2f} GB fits in one shard of {max_bytes / 1e9:.2f} GB, nothing to do.")
            return

        print(f"Sharding {len(sizes)} tensors ({total_size / 1e9:.2f} GB) into {n} shards")
        weight_map = {}
        for i, names in enumerate(shards, start=1):
            shard_name = f"{args.prefix}-{i:05d}-of-{n:05d}.safetensors"
            tensors = {name: f.get_tensor(name) for name in names}
            save_file(tensors, str(out_dir / shard_name), metadata={"format": "pt", **{k: v for k, v in metadata.items() if k != "format"}})
            weight_map.update({name: shard_name for name in names})
            print(f"  [{i}/{n}] {shard_name}: {len(names)} tensors")
            del tensors

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    index_path = out_dir / f"{args.prefix}.safetensors.index.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    print(f"Wrote {index_path}")

    if args.remove_original:
        input_path.unlink()
        print(f"Removed {input_path}")


def _dtype_bytes(dtype):
    return {
        "F64": 8, "I64": 8, "U64": 8,
        "F32": 4, "I32": 4, "U32": 4,
        "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
        "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1,
    }[dtype]


if __name__ == "__main__":
    main()
