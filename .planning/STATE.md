# Project State: TuroArnis

## Status Overview

| Phase | Name | Status | Plans | Progress |
|-------|------|--------|-------|----------|
| 1 | Core CV Pipeline | ✓ Complete | 3/3 | 100% |
| 2 | GCN Model Integration | ✓ Complete | 4/4 | 100% |
| 3 | Feedback System | ✓ Complete | 3/3 | 100% |
| 4 | GUI and User Interface | ✓ Complete | 5/5 | 100% |
| 5 | Database and Persistence | ✓ Complete | 2/2 | 100% |
| 6 | Packaging and Deployment | ○ Planned | 0/TBD | 0% |
| 7 | Distribution and Updates | ○ Future | - | - |
| 8 | Performance Optimization | ○ Future | - | - |

## Milestone Progress

**Milestone 1: MVP (Phases 1-5)** — ✓ COMPLETE (2026-04-13)
- Real-time pose detection
- GCN-based technique classification
- Feedback generation
- Multi-mode GUI
- Database persistence

**Milestone 2: Deployment Ready (Phase 6)** — ○ IN PROGRESS
- Standalone executable
- Asset bundling
- Installer creation

## Current Decisions

| Decision | Value | Made | Phase |
|----------|-------|------|-------|
| Build Tool | PyInstaller 6.12.0 | 2026-04-13 | 6 |
| Package Mode | One-directory (_internal) | 2026-04-13 | 6 |
| Console | Keep enabled (debugging) | 2026-04-13 | 6 |
| UPX | Enabled for compression | 2026-04-13 | 6 |

## Blockers

None currently.

## Technical Debt

1. **torch_geometric.distributed exclusion** — Required to prevent RPC errors during build
2. **Manual hiddenimports** — Many ML libraries need explicit import declarations
3. **Model file sizes** — GCN .pth files are large (~50MB each)

## Last Activity

- 2026-04-13: Planning phase initialized for Packaging (Phase 6)

---

*State file: .planning/STATE.md*
