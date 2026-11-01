# The Deterministic Butterfly Cone of a Structural Glass

This project measures how fast a structural glass forgets its own microstate. Deterministic NVE twins branched by one localized kick give a noise-free difference field that grows as a butterfly cone.

## Headline results

- The difference between the twins grows exponentially behind an advancing front and then stops at a ceiling with no free parameter, `D_sat/N = c* u_DW` with `c* = 4/sqrt(3 pi) = 1.3029`.
- Tested particle by particle at fixed cage size, the identity holds to within `2` percent: the slope is `1.2806 +/- 0.0006` across four deep states, or `0.983 c*`. Aggregate ratios at ten states of six glass formers lie between `1.20` and `1.27`, below `c*`, as a spread of cage sizes requires.
- Twins never kicked saturate at the same amplitude, independently equilibrated configurations separate more than three times further, and a harmonic replay transmits the kick without amplifying it.
- Run at a common `T = 0.075`, parents annealed at `T_f = 0.150` to `0.075` give growth rates of `0.897` to `1.003`, against `0.897` to `1.469` at their own temperatures, so the rate mainly follows temperature. The ceiling follows the cage: `D_sat/N` rises from `0.147` to `0.220 sigma` as the cage amplitude rises from `0.119` to `0.223 sigma`.
- Confinement to a `4.01 sigma` film lowers `D_sat/N` by `17.0` percent and the cage amplitude by `15.8` percent.

## Install and reproduce

Install CPython 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
bash reproduce.sh
```

## Outputs

The replay recomputes `83` of the `101` frozen values, re-reads `18`, runs the tests and a cpu self-test, and audits every byte.

## Repository map

- `src/butterfly_cone/`: engine, branching, kicks, entropy, measurement
- `scripts/`: campaign drivers, re-analyses, verifier, and figures
- `configs/` and `data/`: claim surface and external-asset boundaries
- `tests/`: automated checks
- `results/`: retained tables, figures, and checksums
- `results/revision/`: summary of the annealing and diameter-scaled kick runs added in the revision, made by `scripts/revision_*.py` and `scripts/figures/fig_revision.py`

## License

Released under the [MIT License](LICENSE); external assets remain subject to their original terms.
