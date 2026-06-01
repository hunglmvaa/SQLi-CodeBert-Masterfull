import os
import shutil
from pathlib import Path

import onnx
from onnxconverter_common.float16 import convert_float_to_float16
from transformers import AutoTokenizer, AutoConfig
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTOptimizer, ORTQuantizer
from optimum.onnxruntime.configuration import OptimizationConfig, AutoQuantizationConfig

import config


def _clean_dir(path):
    """Remove only ONNX output folders, then recreate them."""
    p = Path(path)
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_metadata(model_dir, output_dirs):
    """Save tokenizer/config metadata so Optimum/ORT can reload each ONNX directory."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model_config = AutoConfig.from_pretrained(model_dir)
    for d in output_dirs:
        tokenizer.save_pretrained(d)
        model_config.save_pretrained(d)


def _list_onnx_files(path, label):
    p = Path(path)
    files = sorted(p.glob("*.onnx"))
    print(f"[{label}] ONNX files in {p}:")
    for f in files:
        print(f"  - {f.name} ({f.stat().st_size / 1024 / 1024:.2f} MB)")


def main():
    model_dir = Path(config.MODEL_DIR)
    fp32_dir = _clean_dir(config.ONNX_FP32_DIR)
    fp16_dir = _clean_dir(config.ONNX_FP16_DIR)
    int8_dir = _clean_dir(config.ONNX_INT8_DIR)

    print("[0/4] Prepare output directories and metadata")
    print("MODEL_DIR    :", model_dir)
    print("ONNX_FP32_DIR:", fp32_dir)
    print("ONNX_FP16_DIR:", fp16_dir)
    print("ONNX_INT8_DIR:", int8_dir)
    _save_metadata(model_dir, [fp32_dir, fp16_dir, int8_dir])

    print("[1/4] Export PyTorch model to ONNX FP32")
    ort_model = ORTModelForSequenceClassification.from_pretrained(model_dir, export=True)
    ort_model.save_pretrained(fp32_dir)
    raw_fp32 = fp32_dir / "model.onnx"
    if not raw_fp32.exists():
        candidates = sorted(fp32_dir.glob("*.onnx"))
        if not candidates:
            raise FileNotFoundError(f"No exported ONNX model found in {fp32_dir}")
        raw_fp32 = candidates[0]
    print("Raw FP32 ONNX:", raw_fp32.name)

    print("[2/4] Optimize ONNX FP32 with graph optimization level 99/O3")
    optimizer = ORTOptimizer.from_pretrained(ort_model)
    opt_config = OptimizationConfig(optimization_level=99, optimize_for_gpu=False)
    optimizer.optimize(save_dir=fp32_dir, optimization_config=opt_config)

    optimized_fp32 = fp32_dir / "model_optimized.onnx"
    if not optimized_fp32.exists():
        candidates = [f for f in sorted(fp32_dir.glob("*.onnx")) if f.name != raw_fp32.name]
        if candidates:
            optimized_fp32 = candidates[0]
        else:
            print("[warn] Optimized ONNX was not separately created; using raw FP32 for FP16 conversion.")
            optimized_fp32 = raw_fp32
    print("Optimized FP32 ONNX:", optimized_fp32.name)

    print("[3/4] Create ONNX FP16 optimized model")
    fp16_model_path = fp16_dir / "model_optimized.onnx"
    onnx_model = onnx.load(str(optimized_fp32))
    onnx_model_fp16 = convert_float_to_float16(onnx_model, keep_io_types=True)
    onnx.save(onnx_model_fp16, str(fp16_model_path))
    print("FP16 ONNX:", fp16_model_path.name)

    print("[4/4] Create ONNX INT8 dynamic quantized model")
    # Important: fp32_dir intentionally contains multiple ONNX files:
    #   - model.onnx: raw exported model
    #   - model_optimized.onnx: O3 graph-optimized model
    # ORTQuantizer.from_pretrained(directory) fails when there are multiple ONNX files.
    # Therefore we explicitly pass file_name.
    quant_source = raw_fp32.name
    print(f"[quantize] Using source ONNX file: {quant_source}")
    quantizer = ORTQuantizer.from_pretrained(fp32_dir, file_name=quant_source)
    qconfig = AutoQuantizationConfig.avx2(is_static=False)
    quantizer.quantize(save_dir=int8_dir, quantization_config=qconfig)

    _list_onnx_files(fp32_dir, "FP32")
    _list_onnx_files(fp16_dir, "FP16")
    _list_onnx_files(int8_dir, "INT8")
    print("ONNX optimization/quantization completed successfully.")


if __name__ == "__main__":
    main()
