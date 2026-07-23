# ICAI 2026 paper draft

This folder contains an anonymous first draft for ICAI 2026 using the Springer LNCS/CCIS LaTeX template.

Main files:

- `main.tex`: paper source.
- `references.bib`: bibliography copied from the thesis project.
- `llncs.cls` and `splncs04.bst`: Springer proceedings template files.
- `figures/`: figures used by the draft.
- `output/main.pdf`: compiled PDF.

Important checks before submission:

- Validate the local buffer size inconsistency. The analyzed results log reports 30 samples per local update, while one inspected gateway script sets 40.
- Confirm whether all figures are anonymous and do not contain institutional or author metadata.
- Remove or anonymize any repository URL if added later.
- Re-run all result tables from a single frozen dataset/results folder before camera-ready.
- Keep the first submission anonymous: no author names, affiliation, acknowledgments, ORCID, GitHub profile, or local filesystem paths.

ICAI 2026 constraints verified on the official site:

- Full paper, 12 to 16 pages.
- English.
- Springer LNCS/CCIS template.
- First submission as anonymous PDF.
- Microsoft CMT submission.
