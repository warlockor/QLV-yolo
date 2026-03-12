#!/usr/bin/env python3
"""
Export an Ultralytics YOLO .pt model into files for three C++ backends in this repo.

Outputs:
1) <stem>_armnn_int8.tflite       -> Arm NN + OpenCL path
2) <stem>_tflite_cpu_int8.tflite  -> TFLite CPU path
3) <stem>_tflite_gpu_fp16.tflite  -> TFLite GPU delegate path (fallback to fp32 name if needed)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Dict


def _require_ultralytics():
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Failed to import ultralytics. Install with: pip install ultralytics"
        ) from exc
    return YOLO


def _resolve_export_path(export_result: Any) -> Path:
    if isinstance(export_result, (str, Path)):
        return Path(export_result)
    # Some ultralytics versions return dict-like metadata.
    if isinstance(export_result, dict):
        for key in ("file", "path", "model"):
            if key in export_result and export_result[key]:
                return Path(str(export_result[key]))
    raise RuntimeError(
        f"Cannot resolve export path from result type: {type(export_result).__name__}"
    )


def _export_tflite(model: Any, kwargs: Dict[str, Any]) -> Path:
    result = model.export(format="tflite", **kwargs)
    out_path = _resolve_export_path(result)
    if out_path.suffix != ".tflite" or not out_path.exists():
        raise RuntimeError(f"Exported file is invalid: {out_path}")
    return out_path


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Ultralytics YOLO .pt into model files used by this repository."
    )
    parser.add_argument("--pt", required=True, help="Path to Ultralytics .pt model")
    parser.add_argument(
        "--out-dir",
        default="models",
        help="Output directory for converted files (default: models)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Export image size (default: 640)",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Dataset yaml for INT8 calibration, e.g. coco8.yaml (recommended for INT8)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Ultralytics export device, e.g. cpu / 0 (optional)",
    )
    args = parser.parse_args()

    pt_path = Path(args.pt).expanduser().resolve()
    if not pt_path.exists():
        raise FileNotFoundError(f".pt model not found: {pt_path}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pt_path.stem

    YOLO = _require_ultralytics()
    model = YOLO(str(pt_path))

    common_kwargs: Dict[str, Any] = {"imgsz": args.imgsz}
    if args.device:
        common_kwargs["device"] = args.device

    # 1) Export INT8 TFLite once, then duplicate for Arm NN and TFLite CPU.
    int8_kwargs = dict(common_kwargs)
    int8_kwargs["int8"] = True
    if args.data:
        int8_kwargs["data"] = args.data
    else:
        print(
            "[WARN] --data not provided. INT8 calibration quality may be poor.",
            file=sys.stderr,
        )

    print("[INFO] Exporting INT8 TFLite...")
    int8_path = _export_tflite(model, int8_kwargs)

    armnn_int8 = out_dir / f"{stem}_armnn_int8.tflite"
    tflite_cpu_int8 = out_dir / f"{stem}_tflite_cpu_int8.tflite"
    _copy(int8_path, armnn_int8)
    _copy(int8_path, tflite_cpu_int8)

    # 2) Export FP16 (preferred) for TFLite GPU delegate.
    print("[INFO] Exporting FP16 TFLite for GPU delegate...")
    fp16_kwargs = dict(common_kwargs)
    fp16_kwargs["half"] = True

    gpu_out: Path
    try:
        fp16_path = _export_tflite(model, fp16_kwargs)
        gpu_out = out_dir / f"{stem}_tflite_gpu_fp16.tflite"
        _copy(fp16_path, gpu_out)
    except Exception as fp16_exc:
        print(
            "[WARN] FP16 export failed, fallback to FP32 TFLite for GPU delegate.",
            file=sys.stderr,
        )
        print(f"[WARN] Reason: {fp16_exc}", file=sys.stderr)
        fp32_path = _export_tflite(model, common_kwargs)
        gpu_out = out_dir / f"{stem}_tflite_gpu_fp32.tflite"
        _copy(fp32_path, gpu_out)

    print("\n[OK] Conversion finished:")
    print(f"- Arm NN (Scheme 1):      {armnn_int8}")
    print(f"- TFLite CPU (Scheme 2):  {tflite_cpu_int8}")
    print(f"- TFLite GPU (Scheme 3):  {gpu_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
