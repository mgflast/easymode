# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.2] - 2025-12-05

### Changed
- Migrated from setup.py to pyproject.toml (PEP 517/621)
- Restructured to src-layout for better packaging practices
- Added modern development tooling (ruff, mypy, pre-commit)
- Added comprehensive CI/CD via GitHub Actions
- Implemented git-based versioning via hatch-vcs

### Infrastructure
- Python 3.10+ required (dropped 3.7-3.9 support)
- Added pytest-based testing infrastructure
- Added pre-commit hooks for code quality
- Added multi-platform CI (Ubuntu, macOS, Windows)
- All tests run on Python 3.10-3.13

### Maintained
- All existing functionality preserved
- GPL v3 license maintained
- All dependencies kept as-is
- Package imports unchanged (`import easymode.X`)
