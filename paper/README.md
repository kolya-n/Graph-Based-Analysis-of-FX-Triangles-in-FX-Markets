# Thesis paper (LaTeX)

Bachelor diploma thesis. Structure follows `ai-memory/THESIS_PLAN.md`: six prose-based
body sections (Introduction; Literature Review & Theoretical Background; Model &
Methodology; Experimental Design; Results & Discussion; Conclusion, Limitations & Future
Work) plus Abstract, Russian Аннотация, and Appendices A/B/C.

## Files
- `main.tex` — the thesis (`article` class, sections; `natbib` bibliography).
- `references.bib` — citations (web-verified; do not add unverified sources).

## Compile
Easiest: drop the `paper/` folder into **Overleaf** (builds as-is).

Locally (MiKTeX/TeX Live) — run `build.ps1` / `build.bat`, or:
```
pdflatex main
bibtex main
pdflatex main
pdflatex main
```
The Cyrillic Аннотация needs the `T2A` encoding (MiKTeX/Overleaf auto-install it).

## Notes
- Edit the five `\Inst*` macros at the top of `main.tex` (university, faculty, degree,
  supervisor, city) before printing — the title page reads from them.
- Result numbers in Table 1 are current single-seed values (model
  `td3_fx_graph_60k_v4`); to be updated with multi-seed mean ± std.
- Figures are wired from `../results/plots/eval_v4_{indist,regime}_comparison.png` via
  `\graphicspath`; on Overleaf, upload those PNGs (e.g. into a `figures/` folder).
