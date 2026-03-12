#!/usr/bin/env python3
"""
Evaluate exported YOLO models and compare results across three deployment schemes.

Supported model files:
- *_armnn_int8.tflite
- *_tflite_cpu_int8.tflite
- *_tflite_gpu_fp16.tflite or *_tflite_gpu_fp32.tflite
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


def _require_ultralytics():
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Failed to import ultralytics. Install with: pip install ultralytics"
        ) from exc
    return YOLO


def _metric_get(metrics: Any, head: str, key: str) -> Optional[float]:
    try:
        branch = getattr(metrics, head, None)
        if branch is None:
            return None
        value = getattr(branch, key, None)
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _fmt(v: Optional[float], digits: int = 4) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}"


def evaluate_one(
    name: str,
    model_path: Path,
    data_yaml: Path,
    imgsz: int,
    split: str,
    device: str,
    batch: int,
    task: str,
) -> Dict[str, Any]:
    YOLO = _require_ultralytics()

    if not model_path.exists():
        return {"name": name, "model": str(model_path), "ok": False, "error": "file not found"}

    start = time.perf_counter()
    try:
        # Explicitly set task to avoid exported models being auto-detected as "detect".
        model = YOLO(str(model_path), task=task)
        metrics = model.val(
            data=str(data_yaml),
            split=split,
            imgsz=imgsz,
            batch=batch,
            device=device,
            task=task,
            verbose=False,
        )
        elapsed = time.perf_counter() - start

        result = {
            "name": name,
            "model": str(model_path),
            "ok": True,
            "box_mAP50": _metric_get(metrics, "box", "map50"),
            "box_mAP50-95": _metric_get(metrics, "box", "map"),
            "box_precision": _metric_get(metrics, "box", "mp"),
            "box_recall": _metric_get(metrics, "box", "mr"),
            "seg_mAP50": _metric_get(metrics, "seg", "map50"),
            "seg_mAP50-95": _metric_get(metrics, "seg", "map"),
            "seg_precision": _metric_get(metrics, "seg", "mp"),
            "seg_recall": _metric_get(metrics, "seg", "mr"),
            "fps_estimate": None,
            "val_seconds": elapsed,
        }

        if task == "segment" and result["seg_mAP50"] is None:
            result["warning"] = (
                "seg metrics unavailable. Possible causes: model loaded as detect, "
                "dataset yaml/labels are detection-only, or this exported backend/version "
                "does not expose mask predictions for validation."
            )

        # Ultralytics speed values are usually in ms/image.
        try:
            speed = getattr(metrics, "speed", None) or {}
            post_ms = float(speed.get("postprocess", 0.0))
            infer_ms = float(speed.get("inference", 0.0))
            pre_ms = float(speed.get("preprocess", 0.0))
            total_ms = pre_ms + infer_ms + post_ms
            if total_ms > 0.0:
                result["fps_estimate"] = 1000.0 / total_ms
            result["speed_ms"] = {
                "preprocess": pre_ms,
                "inference": infer_ms,
                "postprocess": post_ms,
                "total": total_ms,
            }
        except Exception:
            pass

        return result
    except Exception as exc:
        return {"name": name, "model": str(model_path), "ok": False, "error": str(exc)}


def auto_pick_models(out_dir: Path, stem: str) -> Dict[str, Path]:
    armnn = out_dir / f"{stem}_armnn_int8.tflite"
    cpu = out_dir / f"{stem}_tflite_cpu_int8.tflite"

    gpu_fp16 = out_dir / f"{stem}_tflite_gpu_fp16.tflite"
    gpu_fp32 = out_dir / f"{stem}_tflite_gpu_fp32.tflite"
    gpu = gpu_fp16 if gpu_fp16.exists() else gpu_fp32

    return {"armnn": armnn, "tflite_cpu": cpu, "tflite_gpu": gpu}


def print_table(results: list[Dict[str, Any]]) -> None:
    print("\n=== Exported Model Evaluation Summary ===")
    print(
        f"{'Scheme':<14} {'Status':<8} {'box50':>8} {'box95':>8} {'seg50':>8} {'seg95':>8} {'FPS':>8}  Model"
    )
    print("-" * 110)
    for r in results:
        if not r.get("ok"):
            print(
                f"{r['name']:<14} {'FAIL':<8} {'-':>8} {'-':>10} {'-':>8} {'-':>8} {'-':>8}  {r['model']}"
            )
            print(f"{'':<14} {'':<8} error: {r.get('error', 'unknown')}")
            continue
        print(
            f"{r['name']:<14} {'OK':<8} "
            f"{_fmt(r.get('box_mAP50')):>8} "
            f"{_fmt(r.get('box_mAP50-95')):>8} "
            f"{_fmt(r.get('seg_mAP50')):>8} "
            f"{_fmt(r.get('seg_mAP50-95')):>8} "
            f"{_fmt(r.get('fps_estimate'), 2):>8}  "
            f"{r['model']}"
        )
        if r.get("warning"):
            print(f"{'':<14} {'':<8} warn: {r['warning']}")

    ok_count = sum(1 for r in results if r.get("ok"))
    print(f"\nFinished: {ok_count}/{len(results)} models evaluated successfully.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate three exported models and compare accuracy/speed."
    )
    parser.add_argument("--data", required=True, help="Dataset yaml used for validation")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size")
    parser.add_argument("--split", default="val", help="Dataset split: val/test/train")
    parser.add_argument("--device", default="cpu", help="Ultralytics device, e.g. cpu or 0")
    parser.add_argument("--batch", type=int, default=1, help="Validation batch size")
    parser.add_argument(
        "--task",
        default="segment",
        choices=["segment", "detect"],
        help="Validation task type. Use 'segment' for yolo*-seg models.",
    )

    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory containing exported models; used with --stem for auto-discovery",
    )
    parser.add_argument(
        "--stem",
        default=None,
        help="Model stem for auto-discovery, e.g. yolo26n",
    )

    parser.add_argument("--armnn-model", default=None, help="Explicit path to *_armnn_int8.tflite")
    parser.add_argument("--tflite-cpu-model", default=None, help="Explicit path to *_tflite_cpu_int8.tflite")
    parser.add_argument(
        "--tflite-gpu-model",
        default=None,
        help="Explicit path to *_tflite_gpu_fp16.tflite or *_tflite_gpu_fp32.tflite",
    )
    parser.add_argument("--save-json", default=None, help="Save full evaluation result JSON to this file")
    args = parser.parse_args()

    data_yaml = Path(args.data).expanduser().resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"--data yaml not found: {data_yaml}")

    resolved: Dict[str, Path] = {}
    if args.out_dir and args.stem:
        auto_models = auto_pick_models(Path(args.out_dir).expanduser().resolve(), args.stem)
        resolved.update(auto_models)

    if args.armnn_model:
        resolved["armnn"] = Path(args.armnn_model).expanduser().resolve()
    if args.tflite_cpu_model:
        resolved["tflite_cpu"] = Path(args.tflite_cpu_model).expanduser().resolve()
    if args.tflite_gpu_model:
        resolved["tflite_gpu"] = Path(args.tflite_gpu_model).expanduser().resolve()

    missing = [k for k in ("armnn", "tflite_cpu", "tflite_gpu") if k not in resolved]
    if missing:
        raise ValueError(
            "Missing model paths for: "
            + ", ".join(missing)
            + ". Use --out-dir + --stem, or pass explicit --armnn-model/--tflite-cpu-model/--tflite-gpu-model."
        )

    results = [
        evaluate_one("armnn", resolved["armnn"], data_yaml, args.imgsz, args.split, args.device, args.batch, args.task),
        evaluate_one(
            "tflite_cpu", resolved["tflite_cpu"], data_yaml, args.imgsz, args.split, args.device, args.batch, args.task
        ),
        evaluate_one(
            "tflite_gpu", resolved["tflite_gpu"], data_yaml, args.imgsz, args.split, args.device, args.batch, args.task
        ),
    ]

    print_table(results)

    if args.save_json:
        out = Path(args.save_json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved JSON: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
