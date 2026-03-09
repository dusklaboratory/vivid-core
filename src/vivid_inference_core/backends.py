import os
import sys

from .contracts import BackendFactoryProtocol

_vsmlrt = None


def _get_vsmlrt():
    global _vsmlrt
    if _vsmlrt is None:
        import vsmlrt as _mod

        _vsmlrt = _mod
    return _vsmlrt


class BackendFactory(BackendFactoryProtocol):
    @staticmethod
    def create_backend(backend_name, config, device_id=None):
        vsmlrt = _get_vsmlrt()
        name = backend_name.lower().replace("_", "-")
        dev_id = (
            device_id if device_id is not None else int(getattr(config, "data", {}).get("deviceId", 0))
        )

        if "ort-coreml" in name or "coreml" in name:
            print("[Backend] Using CoreML backend", file=sys.stderr)
            return vsmlrt.Backend.ORT_COREML(num_streams=config.streams, fp16=config.fp16)
        if "dml" in name:
            print(f"[Backend] Using DirectML backend (device={dev_id})", file=sys.stderr)
            return vsmlrt.Backend.ORT_DML(device_id=dev_id, num_streams=config.streams, fp16=config.fp16)
        if "trt-rtx" in name or "trt_rtx" in name:
            print(f"[Backend] Using TensorRT RTX backend (device={dev_id})", file=sys.stderr)
            return vsmlrt.Backend.TRT_RTX(device_id=dev_id, num_streams=config.streams, fp16=config.fp16)
        if "trt" in name:
            engine_folder = os.environ.get("VIVID_TRT_ENGINE_FOLDER")
            return vsmlrt.Backend.TRT(
                device_id=dev_id,
                num_streams=config.streams,
                fp16=config.fp16,
                engine_folder=engine_folder,
            )
        if "ort-cuda" in name or "ort_cuda" in name:
            return vsmlrt.Backend.ORT_CUDA(device_id=dev_id, num_streams=config.streams, fp16=config.fp16)
        if "ov-gpu" in name or "ov_gpu" in name:
            return vsmlrt.Backend.OV_GPU(device_id=dev_id, num_streams=config.streams, fp16=config.fp16)
        if "ov-cpu" in name or "ov_cpu" in name:
            return vsmlrt.Backend.OV_CPU(num_streams=config.streams, fp16=config.fp16)
        if "migx" in name:
            return vsmlrt.Backend.MIGX(device_id=dev_id, num_streams=config.streams, fp16=config.fp16)
        if "ncnn" in name:
            return vsmlrt.Backend.NCNN_VK(device_id=dev_id, num_streams=config.streams, fp16=config.fp16)
        if "cpu" in name:
            from multiprocessing import cpu_count

            num_streams = min(config.streams, max(1, cpu_count() // 2))
            return vsmlrt.Backend.ORT_CPU(num_streams=num_streams)
        raise ValueError(f"Unknown backend '{name}'")
