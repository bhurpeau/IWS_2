# IWS Paper II — first iteration

This repository contains a first simulation prototype for **Paper II** of the Inner World System (IWS) program.

It extends the Paper I dynamics with **structural plasticity**:

- division with **trace inheritance**,
- apoptosis by **crisis**,
- apoptosis by **senescence** through a residual division capacity `kappa`.

## Files

- `model.py` — core hybrid simulation with variable population size
- `experiments.py` — four comparative regimes
- `plotting.py` — figure generation
- `output/` — generated figures

## Comparative regimes

- **E1**: no structural plasticity
- **E2**: division only
- **E3**: division + crisis apoptosis
- **E4**: full plasticity (division + crisis apoptosis + senescence)

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python experiments.py --steps 3000 --nodes 10 --seeds 7 11 19 23 31
```

This generates:

- `output/figure_population.png`
- `output/figure_lineages.png`

## Notes

This is a **first iteration** meant to quickly explore the qualitative regimes of Paper II.
The code favors readability and direct modifiability over optimization.
