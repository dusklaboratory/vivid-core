import os

import vapoursynth as vs

from vivid_inference_core.contracts import ModelLogicProtocol

core = vs.core


def detect_input_format(model_path):
    try:
        import onnx

        model = onnx.load(model_path, load_external_data=False)
        for inp in model.graph.input:
            if inp.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
                return vs.RGBH
        return vs.RGBS
    except Exception:
        pass

    name_lower = model_path.lower()
    if "_fp16" in name_lower or "fp16" in os.path.basename(name_lower):
        return vs.RGBH
    return vs.RGBS


class ModelLogic(ModelLogicProtocol):
    PYTORCH_NATIVE = False

    def process(self, clip, config, backend, model_path):
        raise NotImplementedError
