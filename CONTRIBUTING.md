# Contributing to nirapi

Thank you for your interest in contributing to nirapi! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

By participating in this project, you are expected to uphold our code of conduct. Please be respectful and professional in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/nirapi.git
   cd nirapi
   ```
3. **Create a new branch** for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Installation

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install in development mode**:
   ```bash
   pip install -e .
   ```

4. **Install development dependencies**:
   ```bash
   pip install pytest pytest-cov black flake8 mypy
   ```

## Contributing Guidelines

### Types of Contributions

We welcome the following types of contributions:

- **Bug fixes**: Fix existing issues or improve error handling
- **New features**: Add new functionality for NIR spectroscopy analysis
- **Documentation improvements**: Enhance docstrings, README, or examples
- **Performance optimizations**: Improve code efficiency
- **Test coverage**: Add or improve unit tests

### Before You Start

1. **Check existing issues** to see if your idea is already being discussed
2. **Open an issue** for new features or major changes to discuss the approach
3. **Keep changes focused** - submit separate PRs for different features/fixes

## Code Style

### Python Code Style

- **Follow PEP 8** style guidelines
- **Use Black** for code formatting:
  ```bash
  black nirapi tests
  ```
- **Use meaningful variable names** and add comments where necessary
- **Keep functions focused** and reasonably sized

### Docstring Style

Use Google-style docstrings:

```python
def function_name(param1: type, param2: type) -> return_type:
    """
    Brief description of the function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When invalid input is provided
    """
```

### Import Organization

- **Standard library imports** first
- **Third-party imports** second
- **Local imports** last
- **Use absolute imports** when possible

## Testing

### Running Tests

```bash
# Run all tests
python -m unittest discover tests

# Run specific test file
python -m unittest tests.test_preprocessing

# Run with coverage
python -m pytest --cov=nirapi tests/
```

### Writing Tests

- **Write tests for new functionality**
- **Include edge cases and error conditions**
- **Use descriptive test names**
- **Follow the existing test structure**

Example test structure:
```python
class TestNewFeature(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.test_data = create_test_data()
    
    def test_basic_functionality(self):
        """Test basic functionality works as expected."""
        result = new_feature(self.test_data)
        self.assertIsNotNone(result)
    
    def test_input_validation(self):
        """Test input validation raises appropriate errors."""
        with self.assertRaises(ValueError):
            new_feature(invalid_input)
```

## Documentation

### API Documentation

- **Add docstrings** to all public functions and classes
- **Include examples** in docstrings when helpful
- **Document parameters, return values, and exceptions**

### README Updates

- **Update README.md** if your changes affect usage
- **Add examples** for new features
- **Keep installation instructions current**

## Submitting Changes

### Pull Request Process

1. **Update documentation** if needed
2. **Add tests** for new functionality
3. **Ensure all tests pass**:
   ```bash
   python -m unittest discover tests
   ```
4. **Run code formatting**:
   ```bash
   black nirapi tests
   ```
5. **Create a pull request** with a clear description

### Pull Request Guidelines

- **Use a descriptive title** that explains the change
- **Reference related issues** using "Fixes #123" or "Addresses #123"
- **Provide context** about why the change is needed
- **Describe the solution** and any alternative approaches considered
- **List any breaking changes** if applicable

### Pull Request Template

```markdown
## Description
Brief description of the changes made.

## Related Issues
Fixes #123

## Changes Made
- Added new feature X
- Fixed bug in function Y
- Updated documentation

## Testing
- [ ] Added tests for new functionality
- [ ] All existing tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

- **Python version** and operating system
- **nirapi version** being used
- **Minimal reproducible example**
- **Expected vs actual behavior**
- **Error messages** and stack traces

### Feature Requests

When requesting features, please include:

- **Use case description** and motivation
- **Proposed solution** or approach
- **Alternative solutions** considered
- **Additional context** that might be helpful

### Issue Templates

Use the following format for bug reports:

```markdown
## Bug Description
Clear description of the bug.

## Steps to Reproduce
1. Step one
2. Step two
3. Step three

## Expected Behavior
What should happen.

## Actual Behavior
What actually happened.

## Environment
- OS: [e.g., Windows 10, macOS 11, Ubuntu 20.04]
- Python version: [e.g., 3.9.7]
- nirapi version: [e.g., 1.0.0]

## Additional Context
Any other relevant information.
```

## Development Tips

### Setting Up IDE

For VS Code users, consider these extensions:
- Python
- Black Formatter
- Pylint
- Python Docstring Generator

### Common Commands

```bash
# Format code
black nirapi tests

# Run linting
flake8 nirapi

# Type checking
mypy nirapi

# Run tests with coverage
python -m pytest --cov=nirapi tests/

# Build documentation
python -m sphinx docs/ docs/_build/
```

## Questions?

If you have questions about contributing, please:

1. **Check existing issues** for similar questions
2. **Open a new issue** with the "question" label
3. **Reach out** via the project's communication channels

Thank you for contributing to nirapi! 