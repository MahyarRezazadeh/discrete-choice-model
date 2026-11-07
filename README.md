# Random-coefficient demand estimation with PyBLP

Author: Mahyar Rezazadeh

Written for ECON 913, Empirical Industrial Organization, Simon Fraser University (problem set 2). The script is my problem-set code, reorganized into functions with command-line paths. The estimating equations, instruments, starting values and optimizer settings are unchanged.

## What it does

The script estimates demand for breakfast cereals in the framework of Berry, Levinsohn and Pakes, with market and product fixed effects, and uses each model to simulate one price cut.

1. Allows taste for one product attribute (`mushy`) to vary with log income.
2. Halves the price of product F1B04 in market C01Q1 and compares substitution with a plain logit.
3. Adds a random coefficient on price and a price by log income interaction, with four instruments.
4. Repeats the price cut under this richer model.
5. Adds a micro moment for the mean log income of buyers, which brings income variation back into a model with market fixed effects.

## Files

| File | Purpose |
|---|---|
| `demand_estimation.py` | Full estimation and counterfactual |
| `requirements.txt` | Package versions used for the run below |
| `results/estimates_summary.csv` | Main coefficients and price-cut results |
| `results/model_output.txt` | Full PyBLP output for the three estimated models |

## Data

The two input files were provided by the course instructor, so they are not included here. For the same reason, `results/counterfactual_results.txt`, which the script writes and which lists prices and shares from those files, is not included.

| File | Columns used |
|---|---|
| `cereals_PS2.csv` | `market_ids`, `product_ids`, `mushy`, `servings_sold`, `city_population`, `prices`, `shares`, `demand_instruments0`, `IV2`, `IV3` |
| `demographics_PS2.csv` | `market_ids`, `log_income` (1,000 sampled consumers per market) |

## How to run

```bash
pip install -r requirements.txt
python demand_estimation.py --data-dir PATH_TO_CSVS --output-dir results
```

A full run took about three minutes on a laptop. Random draws use a fixed seed (0).

## Results from the run on November 6, 2026

The run used Python 3.13 and the package versions in `requirements.txt`. All three nonlinear models converged, and the numbers match my original problem-set write-up.

| Quantity | Estimate |
|---|---|
| Price coefficient, observed heterogeneity | -30.60 (SE 0.97) |
| Price coefficient, random coefficients | 11.31 (SE 13.34) |
| Price by log income, random coefficients | -5.68 |
| Implied price coefficient at mean log income (8.087) | about -34.6 |
| Own share change from the price cut: logit | +194.0% |
| Own share change: observed heterogeneity | +193.8% |
| Own share change: random coefficients | +246.8% |
| Micro moment: target and model value | 10.00 and 8.38 |

The price coefficient turns positive in the third model because price sensitivity is now carried by the income interaction, so the coefficient at the average consumer is the relevant number. The target value of 10 for the micro moment is the value used in the exercise.
