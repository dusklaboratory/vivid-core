# Contributing

## Scope

This repo accepts contributions for:

- model logic integrations,
- backend adapters,
- plugin manifest/runtime contracts,
- compatibility and validation tooling.

This repo does not accept cloud, licensing, auth, billing, or private product policy logic.

## Pull request checklist

- Keep changes modular and focused.
- Add or update tests for any contract changes.
- Avoid breaking public contract names without deprecation.
- Update docs for behavioral changes.
- Prefer the scaffold + helper APIs (`resolve_torch_device`, `resolve_model_repo`) to keep extensions aligned with runtime patterns.

## Model contribution baseline

- Document expected input/output tensor shape.
- Include backend compatibility notes.
- Add deterministic smoke fixture for one frame path.
- Prefer explicit CPU/CUDA/MPS fallback behavior instead of hardcoding CUDA-only paths.
