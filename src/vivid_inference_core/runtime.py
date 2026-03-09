def run_pipeline(model_logic_class, backend_factory_class, model_type="esrgan", backend_name=None):
    """
    Public API placeholder for host integrations.

    Vivid currently keeps host-specific video IO and bundle/runtime wiring private.
    Downstream hosts should wrap this package and provide their own orchestration.
    """
    raise RuntimeError(
        "run_pipeline host orchestration is not bundled in vivid-inference-core yet. "
        "Use this package for contracts/plugin runtime, and keep host execution in your app layer."
    )
