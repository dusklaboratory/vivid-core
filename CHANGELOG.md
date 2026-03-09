# Changelog

## 0.1.0

- Initial public extraction of inference extension contracts.
- Added plugin manifest schema, contract dataclasses, and plugin runner helpers.
- Added AGPL + commercial exception licensing posture.
- Added `community_registry` API for runtime model/backend extension registration.
- Wired Vivid-side runtime adapter path to consume public core extensions when present.
- Added `catalog/community_catalog.json` for renderer model-picker integration in Vivid.
- Added community model-pack contracts (`CommunityModelLogicBase`, `ModelArtifactSpec`, `register_model_pack`) with artifact resolver support.
