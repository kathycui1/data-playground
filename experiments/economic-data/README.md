# Economic data

World Bank annual indicators: GDP growth, CPI inflation, and unemployment from 2000 onward.

The cache includes all available countries plus the World (`WLD`) aggregate. Analysis highlights the World series, a six-country comparison set, and a pooled regression on that set.

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
python experiments/economic-data/render_report.py
jupyter notebook experiments/economic-data/analyze.ipynb
```

`fetch_world_bank.py` writes `data/raw/world_bank_indicators.parquet` (gitignored).  
`render_report.py` writes `report.html` — open it in a browser for charts and tables.

## Indicators

| Indicator | World Bank code |
| --- | --- |
| GDP growth (% annual) | `NY.GDP.MKTP.KD.ZG` |
| Inflation, CPI (% annual) | `FP.CPI.TOTL.ZG` |
| Unemployment (% of labor force) | `SL.UEM.TOTL.ZS` |

**World aggregate:** `WLD`  
**Comparison set:** United States, China, Germany, India, Brazil, Japan
