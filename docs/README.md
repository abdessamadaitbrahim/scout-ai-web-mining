# Scout-AI — Documentation

This directory contains supplementary documentation for the Scout-AI project.

## Structure

```
docs/
├── README.md               ← You are here
├── architecture/           ← System architecture diagrams and descriptions
├── reports/                ← Analysis reports and results
├── presentations/          ← Presentation files (PDF, PPTX) — LOCAL ONLY, not in Git
└── setup/                  ← Step-by-step setup guides for databases and APIs
```

## Presentation Files

The original project presentations are stored **locally only** in `docs/presentations/`.
They are excluded from Git via `.gitignore` because they exceed GitHub size limits.

- `Presentation pdf.pdf`  — Full project report (~69 MB)
- `Presentation ppt.pptx` — Slide deck (~12 MB)

To share them, use Google Drive, OneDrive, or a similar service.

## Key Figures

Machine learning output figures are in `figures/` at the project root:
- `ia_model_comparison.png`    — R² comparison of Random Forest, SVR, Gradient Boosting
- `ia_transfer_ranking.png`    — Transfer potential ranking of 10 target players
- `ia_feature_importance.png`  — Feature importance for the best ML model

## Setup Documentation

See `docs/setup/` for:
- `api_setup.md`    — How to obtain NewsAPI and Football-Data API keys
- `docker_setup.md` — How to configure and start NoSQL databases via Docker
