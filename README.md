# The Deterministic Butterfly Cone of a Structural Glass

This project measures how fast a structural glass forgets its own microstate. Deterministic NVE twins branched by one localized kick give a noise-free difference field that grows as a butterfly cone.

## Headline results

- The cone ceiling equals the Debye-Waller cage amplitude at `6` of `6` glass formers, `c` in `1.2347` to `1.2695` against `1.3029`.
- Over `12` states the rate `lambda` tracks basin entropy at Pearson `r = +0.917` and stiffness at only `r = -0.469`.
- Confinement holds the ceiling to `3.54` percent across `4` films, but anisotropy leaves bulk by `5.42` standard errors at `h = 4.01 sigma`.

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

## License

Released under the [MIT License](LICENSE); external assets remain subject to their original terms.
