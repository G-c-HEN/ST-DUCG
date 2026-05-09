# ST-DUCG Flood Risk Inference Model

A simplified Python implementation of a **DUCG-based flood risk inference model**.
The model infers flood risk levels from multiple environmental and socioeconomic indicators through a three-layer structure:

* **Risk layer**: `B1` (high risk), `B2` (medium risk), `B3` (low risk)

* **Intermediate layer**: `X4` (exposure), `X5` (vulnerability), `X6` (hazard)

* **Indicator layer**: elevation, slope, rainfall, land use, GDP, etc.

The script also prints detailed intermediate reasoning results, including:

* classified indicator states,

* inferred `X4`, `X5`, `X6` states,

* connected risk nodes,

* expanded evidence terms `Pr{E}`,

* expanded joint terms `Pr{H,E}`,

* scores and normalized posterior probabilities.

***

## File

```text
ST-DUCG.py
```

