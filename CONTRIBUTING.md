# Contributing to LBRX-MLX

Thank you for considering contributing to LBRX-MLX! This document outlines the process for contributing to the project and the standards we expect from contributors.

## Development Environment

### Prerequisites

- **Rust** (latest stable version, 1.77.2+)
- **Apple Silicon Mac** (M1/M2/M3 series)
- **macOS** (Sonoma 14.5+ or newer)
- **Xcode Command Line Tools** (for Metal API access)
- **Git LFS** (for model weights management)

### Setting Up

1. Clone the repository:
```bash
git clone https://github.com/LibraxisAI/lbrx-mlx.git
cd lbrx-mlx
```

2. Install dependencies:
```bash
cargo install cargo-criterion # For benchmarking
cargo install cargo-flamegraph # For profiling
```

3. Build the project:
```bash
cargo build --release
```

4. Run tests:
```bash
cargo test
```

## Architecture Guidelines

### Core Design Principles

1. **Performance First**: Every implementation decision must prioritize performance
2. **Memory Efficiency**: Minimize allocations, use zero-copy where possible
3. **Thread Safety**: All components must be safe for multi-threaded use
4. **API Consistency**: Maintain consistent API patterns across modules
5. **Error Handling**: Use explicit error types with rich context

### Coding Standards

- **Code Style**: Follow Rust's official style guidelines
- **Documentation**: All public APIs must have rustdoc comments with examples
- **Testing**: Maintain 90%+ code coverage with unit and integration tests
- **Benchmarking**: All performance-critical paths must have benchmarks
- **Error Messages**: Error messages should be clear and actionable

### Performance Requirements

- Core tensor operations must be within 80% of equivalent C/Metal implementations
- Memory overhead should be <10% compared to raw data size
- All format conversions must have streaming support for models larger than RAM
- Full test suite must run in under 5 minutes

## Development Workflow

### Branching Strategy

- `main`: Stable releases only
- `dev`: Primary development branch
- Feature branches: `feature/name-of-feature`
- Bug fixes: `fix/brief-description`
- Performance improvements: `perf/component-name`

### Pull Request Process

1. Create a branch from `dev` with the appropriate prefix
2. Implement your changes with tests and documentation
3. Ensure all tests and benchmarks pass
4. Submit a PR against the `dev` branch
5. Address review comments
6. Maintain the PR until it's merged

### Commit Guidelines

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` for new features
- `fix:` for bug fixes
- `perf:` for performance improvements
- `refactor:` for code changes that neither add features nor fix bugs
- `docs:` for documentation changes
- `test:` for adding or improving tests
- `chore:` for maintenance tasks

Example:
```
feat(tensor): implement zero-copy view with Metal buffer sharing
```

## Testing Guidelines

### Unit Tests

- Every module should have a `tests` submodule
- Test both success and failure paths
- Use parameterized tests for complex logic
- Mock external dependencies

### Integration Tests

- Test end-to-end workflows
- Verify format compatibility with other tools
- Test with real model weights (small test models)
- Measure and assert performance characteristics

### Benchmarking

- Use criterion.rs for reliable benchmarks
- Benchmark against baseline implementations
- Test with various sizes (small, medium, large tensors)
- Include memory allocation metrics

## Documentation

### API Documentation

- Every public function, struct, and trait must be documented
- Include examples in documentation
- Explain parameter requirements and return values
- Document performance characteristics and complexity

### Architecture Documentation

- Maintain up-to-date architecture diagrams
- Document design decisions in ADRs (Architecture Decision Records)
- Keep README and other guides in sync with implementation

## License and Copyright

All contributions will be licensed under the MIT License.

By contributing to this project, you agree to license your contributions under the same license.

## Code of Conduct

Please follow our [Code of Conduct](CODE_OF_CONDUCT.md) when participating in this project.

## Questions?

If you have questions about contributing, please open a discussion in the GitHub repository.

---

Thank you for helping build LBRX-MLX into the ultimate model manipulation toolkit for Apple Silicon!