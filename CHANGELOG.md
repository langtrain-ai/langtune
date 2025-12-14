# Changelog

All notable changes to **Langtune** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.2] - 2024-12-14

### Added
- 🎨 Beautiful welcome banner on package import with GPU detection
- 🖥️ New `langtune version` command showing system info table
- 📚 New `langtune info` command with quick start guide
- ✨ Rich console output with colors, progress bars, and spinners
- 📦 PyPI classifiers and keywords for better discoverability
- 🔧 Optional dependencies: `[dev]` for development, `[all]` for full integrations
- 📝 Type hints marker file (py.typed) for PEP 561 compliance
- 🖼️ Logo assets for light/dark mode support in README

### Changed
- Enhanced CLI help with examples and tips
- Improved package description
- Updated project URLs with Changelog link

---

## [0.1.1] - 2024-12-10

### Added
- LoRA fine-tuning for language models
- Configurable transformer backbone
- Training with checkpointing and resume support
- Mixed precision training
- WandB integration for experiment tracking
- CLI commands: `train`, `evaluate`, `generate`, `concept`
- Preset configurations: tiny, small, base, large
- Support for RLHF, DPO, PPO, CoT, and other LLM concepts (stubs)

### Changed
- Initial public release

---

## [0.1.0] - 2024-12-01

### Added
- Initial project structure
- Core modules: models, config, data, trainer, utils
- Basic documentation

---

[0.1.2]: https://github.com/langtrain-ai/langtune/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/langtrain-ai/langtune/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/langtrain-ai/langtune/releases/tag/v0.1.0
