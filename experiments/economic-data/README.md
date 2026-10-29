# Economic data (v1)

Question this experiment answers:

> Since 2000, how do GDP growth, inflation, and unemployment move together — globally, and in a few large economies?

The cache is **all countries** plus the World Bank **World** (`WLD`) aggregate. Charts and the regression still zoom in on six large economies, because 200 overlapping lines are not a better global view than the World series.

## Setup

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python experiments/economic-data/fetch_world_bank.py
python experiments/economic-data/render_figures.py
jupyter notebook experiments/economic-data/analyze.ipynb
```

`fetch_world_bank.py` writes `data/raw/world_bank_indicators.parquet` (gitignored). `render_figures.py` writes PNGs to `figures/`. The notebook loads the cache (and fetches if it is missing or still the old six-country file).

## Scope

| Indicator | World Bank code |
| --- | --- |
| GDP growth (% annual) | `NY.GDP.MKTP.KD.ZG` |
| Inflation, CPI (% annual) | `FP.CPI.TOTL.ZG` |
| Unemployment (% of labor force) | `SL.UEM.TOTL.ZS` |

- **Global series:** World Bank World aggregate (`WLD`)
- **Close-up:** United States, China, Germany, India, Brazil, Japan
- **Years:** 2000 through the latest World Bank year

All-country rows are in the parquet for snapshots (for example the latest-year inflation histogram). They are not all plotted as time series, and they are not all thrown into one unweighted regression.

## Later

A simple forecast, interest rates, and an LLM write-up of the findings.
