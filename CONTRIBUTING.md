# Contributing to easymode

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/mgflast/easymode.git
cd easymode
```

2. Create a virtual environment and install development dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

3. Install pre-commit hooks:
```bash
pre-commit install
```

## Running Tests

Run all tests:
```bash
pytest
```

With coverage:
```bash
pytest --cov --cov-report=html
```

View coverage report:
```bash
open htmlcov/index.html  # On macOS
xdg-open htmlcov/index.html  # On Linux
```

## Code Quality

This project uses:
- **ruff** for linting and formatting
- **mypy** for type checking (gradual adoption)
- **pre-commit** for automated checks

Before committing, run:
```bash
pre-commit run --all-files
```

Or rely on the pre-commit hooks to run automatically on `git commit`.

## Code Style

- Line length: 88 characters (Black/Ruff default)
- Docstring convention: NumPy style
- Import sorting: isort via ruff

## Making Changes

1. Create a new branch for your feature or bugfix:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and commit them:
```bash
git add .
git commit -m "Description of changes"
```

3. Push to your fork and create a pull request:
```bash
git push origin feature/your-feature-name
```

## Making a Release

Releases are managed via git tags and automated through GitHub Actions:

1. Update CHANGELOG.md with the new version
2. Commit changes:
```bash
git add CHANGELOG.md
git commit -m "Prepare release vX.Y.Z"
```

3. Create and push a tag:
```bash
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

4. GitHub Actions will automatically build and publish to PyPI

## Project Structure

```
easymode/
├── src/easymode/           # Source code (src-layout)
│   ├── main.py            # CLI entry point
│   ├── core/              # Core utilities
│   ├── segmentation/      # 3D UNet segmentation
│   ├── ddw/               # DeepDeWedge denoising
│   └── n2n/               # Noise2Noise denoising
├── tests/                 # Test suite
├── .github/workflows/     # CI/CD configuration
└── pyproject.toml         # Project configuration

```

## Questions or Issues?

- Open an issue on GitHub
- Check existing issues and pull requests first
- Provide minimal reproducible examples for bugs
