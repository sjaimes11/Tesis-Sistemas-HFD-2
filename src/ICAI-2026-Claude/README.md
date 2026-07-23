# ICAI 2026 — Anonymous Submission

This folder contains the first draft of the paper for the **9th International
Conference on Applied Informatics (ICAI 2026)**, University of Central
Florida, October 13–17, 2026.

## Contents

```
ICAI-2026/
├── README.md                    ← this file
├── paper.pdf                    ← compiled PDF (13 pages)
├── Submission_anonymous.zip     ← bundle ready to upload to CMT
└── source/
    ├── paper.tex                ← LaTeX source (Springer LNCS template)
    ├── paper.bib                ← bibliography (21 entries)
    └── figs/                    ← all figures (7 PNGs)
```

## Status

- [x] **Anonymized** (no author names, no institution, no acknowledgements,
      no GitHub links).
- [x] **Springer LNCS** class (`\documentclass[runningheads]{llncs}`).
- [x] **English** academic prose, not a literal translation of the thesis.
- [x] **13 pages** (within the 12–16 page limit required by ICAI 2026).
- [x] **9-section structure**: Intro, Related Work, Architecture, Federated
      Pipeline, ASCON Layer, Setup, Results, Threats to Validity,
      Conclusions.
- [x] **Threats to Validity** section explicit and honest (closed 3-class
      set, heuristic labelling, limited testbed scale, simulated traffic,
      no side-channel coverage, statistical significance).

## How to recompile

### Option A — local (MiKTeX / TeX Live)
```powershell
cd source
pdflatex paper.tex
bibtex   paper
pdflatex paper.tex
pdflatex paper.tex
```

### Option B — Overleaf
1. Create a new project → "Upload Project" → select
   `Submission_anonymous.zip`.
2. Set the main file to `paper.tex`.
3. Click **Recompile**. Overleaf has `llncs.cls` and `splncs04.bst`
   pre-installed.

## What needs to happen before camera-ready

If the paper is accepted, the camera-ready version must:

1. **Add authors and affiliations**, including ORCID:
   ```latex
   \author{Santiago A. Jaimes Puerto\inst{1}\orcidID{0000-0000-...} \and
           Nicolás Casas Ibarra\inst{1}\orcidID{0000-0000-...}}
   \institute{Universidad de los Andes, Bogotá, Colombia\\
              \email{first.last@uniandes.edu.co}}
   ```
2. **Add acknowledgements** (the advisor: *Prof. Carlos Andrés Lozano
   Garzón, PhD*, the COMIT/SISTIC group, funding sources if any).
3. **Disclose the public repository** of the implementation.
4. **Run Turnitin** before submission; ICAI rejects on >20 % total
   similarity or >10 % from a single source.

## Pending technical points to verify with real CSVs

The current draft uses values consolidated from the SRE analysis layer of
the thesis. Before submission, double-check that the following numbers
match what `analysis_outputs/complete_hfl_analysis/global_round_sli_summary.csv`
actually reports:

- Accuracy / loss of all 8 variants in **Table 4** (`tab:variants_results`).
- Per-channel p95 latency for ASCON in **Figure 5(a)**.
- ASCON envelope overhead (bytes, %) in **Table 5** (`tab:overhead_size`).
- Per-class recall / F1 for the best variant (currently estimated as
  "$\approx$0.972 macro-F1, recall above 0.96 in all three classes" — if you
  want exact numbers, run a `sklearn.metrics.classification_report` on the
  final test set of the best run).

If any number drifts more than 0.5 percentage points from the SRE CSV,
update the LaTeX and recompile.

## Submission checklist (ICAI 2026)

- [x] PDF in Springer LNCS format
- [x] 12–16 pages
- [x] English
- [x] Fully anonymized
- [x] Bibliography with DOI/year metadata
- [ ] Turnitin check (run before upload)
- [ ] Upload through Microsoft CMT (open after April 2026)
- [ ] Track key dates:
  - Submission deadline: **June 7, 2026**
  - Notification: **July 5, 2026**
  - Camera ready + registration: **August 9, 2026**
