# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-08-07

### Added

- Production packaging: `Dockerfile`, `docker-compose.yml`, `.dockerignore`
- GitHub Actions: lint, pytest, evaluation, benchmark, quality gates, Docker build
- Makefile targets: `setup`, `test`, `lint`, `evaluate`, `benchmark`, `demo`, `docker`, `ci`
- One-command interview demo: `scripts/demo.py` (offline by default)
- CI quality-gate validator: `scripts/validate_quality_gates.py`
- Architecture Mermaid diagrams: `docs/architecture/diagrams.md`
- Interview pack: questions + 90-minute walkthrough
- Design trade-offs: `docs/design/tradeoffs.md`
- `ROADMAP.md`, `CHANGELOG.md`, `LICENSE`, `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`
- `sample_outputs/` committed demo artifacts
- Fully rewritten README for public GitHub release

### Changed

- Package version bumped to **1.0.0**
- Architecture frozen for the v1.0 release (no MCP / Memory / Multi-Agent)

### Notes

- Sprints 1–12 delivered the platform capabilities; Sprint 13 is release engineering only.

## [0.1.0] — Pre-release

Internal sprint deliveries (foundation through continuous AI evaluation / quality gates).
