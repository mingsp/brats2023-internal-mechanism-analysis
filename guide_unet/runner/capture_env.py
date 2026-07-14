import argparse
import json
import platform
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path


DEPENDENCIES = [
    "torch",
    "torchvision",
    "numpy",
    "pandas",
    "scipy",
    "SimpleITK",
    "tqdm",
    "matplotlib",
    "seaborn",
    "scikit-image",
    "opencv-python",
    "connected-components-3d",
    "surface-distance",
    "pytest",
]


def get_installed_versions():
    versions = {}
    for package in DEPENDENCIES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def get_torch_runtime_info():
    info = {"available": False}
    try:
        import torch
    except Exception as exc:
        info["error"] = "{}: {}".format(type(exc).__name__, exc)
        return info

    info["available"] = True
    info["version"] = getattr(torch, "__version__", None)
    try:
        info["cuda_is_available"] = bool(torch.cuda.is_available())
    except Exception as exc:
        info["cuda_is_available_error"] = "{}: {}".format(type(exc).__name__, exc)
        return info

    info["cuda_version"] = getattr(torch.version, "cuda", None)
    try:
        info["cudnn_version"] = torch.backends.cudnn.version()
    except Exception as exc:
        info["cudnn_version_error"] = "{}: {}".format(type(exc).__name__, exc)

    if info["cuda_is_available"]:
        try:
            device_count = torch.cuda.device_count()
            info["gpu_count"] = int(device_count)
            info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(device_count)]
        except Exception as exc:
            info["gpu_query_error"] = "{}: {}".format(type(exc).__name__, exc)

    return info


def build_metadata():
    return {
        "captured_at_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "dependencies": get_installed_versions(),
        "torch_runtime": get_torch_runtime_info(),
    }


def main():
    parser = argparse.ArgumentParser(description="Capture runtime and dependency metadata for reproducibility.")
    parser.add_argument("--out_dir", type=str, default="results/metadata", help="Output directory for metadata JSON")
    parser.add_argument(
        "--filename_prefix",
        type=str,
        default="env_capture",
        help="Prefix for metadata filename (timestamp suffix is added automatically)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / "{}_{}.json".format(args.filename_prefix, timestamp)

    metadata_payload = build_metadata()
    out_path.write_text(json.dumps(metadata_payload, indent=2, sort_keys=True), encoding="utf-8")
    print("Wrote environment metadata: {}".format(out_path))


if __name__ == "__main__":
    main()
