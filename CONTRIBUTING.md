# Contributing to Langtune

Thank you for your interest in contributing to Langtune! This guide will help you get started.

## 🚀 Quick Start

1. **Fork the repository**
2. **Clone your fork:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/langtune.git
   cd langtune
   ```
3. **Install in development mode:**
   ```bash
   pip install -e ".[dev]"
   ```

## 📋 Development Setup

### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- Git

### Install Dependencies
```bash
pip install torch numpy tqdm pyyaml scipy wandb rich pytest black flake8 mypy
```

## 🔧 Making Changes

### Code Style
We use [Black](https://github.com/psf/black) for code formatting:
```bash
black src/
```

### Linting
```bash
flake8 src/
mypy src/
```

### Running Tests
```bash
pytest tests/ -v
```

## 📝 Pull Request Process

1. **Create a branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and commit:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

3. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Open a Pull Request** on GitHub

### Commit Message Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

| Type | Description |
|------|-------------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation |
| `style:` | Formatting |
| `refactor:` | Code refactoring |
| `test:` | Adding tests |
| `chore:` | Maintenance |

## 🧪 Testing Guidelines

- Write tests for new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage

### Test Structure
```
tests/
├── test_models.py      # Model tests
├── test_trainer.py     # Trainer tests
├── test_data.py        # Data loading tests
└── conftest.py         # Pytest fixtures
```

## 📖 Documentation

- Update docstrings for new functions
- Update README if adding features
- Add examples for new functionality

## 🤝 Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow

## 📧 Getting Help

- Open an issue for bugs or feature requests
- Join discussions in GitHub Issues
- Email: contact@langtrain.ai

---

**Thank you for contributing! 🎉**
