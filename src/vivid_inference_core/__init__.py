CONTRACT_VERSION = 1

from .contracts import (
    BackendFactoryProtocol,
    EffectLogicProtocol,
    EngineCapabilityContract,
    ModelLogicProtocol,
    PluginManifestContract,
    PluginParameterContract,
    PluginRepoPolicyContract,
    ProgressEventContract,
    RuntimeCallbacksProtocol,
)
from .community_registry import (
    create_backend,
    list_registered_backends,
    list_registered_artifact_resolvers,
    list_registered_model_aliases,
    register_backend,
    register_model_logic,
    register_model_pack,
    resolve_model_artifact,
    resolve_model_logic,
)
from .custom_repo import install_repo_to_sys_path, resolve_repo_root
from .model_extensions import CommunityModelLogicBase, ModelArtifactSpec
from .authoring import (
    ensure_model_repo_on_path,
    get_model_repo_root,
    normalize_backend_name,
    resolve_torch_device,
    resolve_model_repo,
)

# Optional runtime imports:
# Keep authoring utilities importable on machines without VapourSynth.
try:
    from .config import InferenceConfig
    from .plugin_runner import run_plugin
    from .runtime import InferencePipeline, run_pipeline
except ModuleNotFoundError as exc:
    if exc.name != "vapoursynth":
        raise
    InferenceConfig = None
    InferencePipeline = None
    run_pipeline = None
    run_plugin = None

try:
    from .backends import BackendFactory
except ModuleNotFoundError:
    BackendFactory = None

__all__ = [
    "BackendFactoryProtocol",
    "CONTRACT_VERSION",
    "EffectLogicProtocol",
    "EngineCapabilityContract",
    "ModelLogicProtocol",
    "PluginManifestContract",
    "PluginParameterContract",
    "PluginRepoPolicyContract",
    "BackendFactory",
    "create_backend",
    "CommunityModelLogicBase",
    "InferencePipeline",
    "InferenceConfig",
    "install_repo_to_sys_path",
    "list_registered_artifact_resolvers",
    "list_registered_backends",
    "list_registered_model_aliases",
    "ModelArtifactSpec",
    "ensure_model_repo_on_path",
    "get_model_repo_root",
    "normalize_backend_name",
    "ProgressEventContract",
    "register_backend",
    "register_model_logic",
    "register_model_pack",
    "resolve_model_artifact",
    "resolve_model_logic",
    "resolve_repo_root",
    "resolve_torch_device",
    "resolve_model_repo",
    "RuntimeCallbacksProtocol",
    "run_plugin",
    "run_pipeline",
]
