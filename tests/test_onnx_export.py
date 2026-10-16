"""Tests for ONNX runner."""
import numpy as np
from src.models.onnx_exporter import OnnxInferenceRunner

def test_onnx_inference_output():
    runner = OnnxInferenceRunner()
    feats = np.array([[0.5, -0.2, 1.2, 0.0]])
    res = runner.run_onnx_inference(feats)
    assert len(res) == 1
    assert 0.0 <= res[0] <= 1.0
