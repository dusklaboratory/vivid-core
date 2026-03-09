import json
import os
import platform
import tempfile

import vapoursynth as vs


class InferenceConfig:
    def __init__(self):
        self.ossystem = platform.system()
        self.tmp_dir = self._get_tmp_dir()
        self.tmp_file = self._resolve_tmp_file()
        self.data = self._load_data()

    def _get_tmp_dir(self):
        if self.ossystem == "Windows":
            return tempfile.gettempdir() + "\\vivid\\"
        return tempfile.gettempdir() + "/vivid/"

    def _resolve_tmp_file(self):
        tmp_file = os.environ.get("tmp")
        if not tmp_file:
            try:
                tmp_file = vs.get_core().get_var("tmp")
            except Exception:
                pass
        if not tmp_file:
            raise ValueError("Input JSON path not provided. Set env var 'tmp' or VS arg tmp=/path.")
        return tmp_file

    def _load_data(self):
        with open(self.tmp_file, encoding="utf-8") as f:
            return json.load(f)

    @property
    def video_path(self):
        return self.data.get("file") or self.data.get("inputVideo")

    @property
    def model_input(self):
        return self.data.get("onnx") or self.data.get("model")

    @property
    def streams(self):
        return int(self.data.get("streams", 1))

    @property
    def fp16(self):
        return self.data.get("fp16", False) or self.data.get("halfPrecision", False)

    @property
    def backend(self):
        backend = self.data.get("backend", None)
        if backend:
            return backend.lower()

        if (
            os.path.exists("/usr/local/lib/vapoursynth/models")
            or os.path.exists("/models")
            or os.path.exists("/workspace/tensorrt")
        ):
            return "trt"

        system = platform.system()
        machine = platform.machine()
        if system == "Darwin" and machine == "arm64":
            return "ort-coreml"
        if system == "Windows":
            return "dml"
        return "ncnn"
