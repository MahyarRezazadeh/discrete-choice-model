"""Random-coefficient (BLP) demand estimation with PyBLP.

Author: Mahyar Rezazadeh
Course: ECON 913, Empirical Industrial Organization, Simon Fraser University
Source: restructured from my problem-set code. The estimating equations,
instruments, starting values and optimizer settings are unchanged. See README.md.

Steps
  1. Price coefficient with observed heterogeneity (mushy x log income)
  2. Counterfactual: halve the price of one product, compare substitution
  3. Random coefficient on price plus price x log income
  4. The same counterfactual under the richer model
  5. Add a micro moment on the income of buyers

Usage
  python demand_estimation.py --data-dir PATH_TO_CSVS --output-dir results
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyblp
import statsmodels.formula.api as smf

COUNTERFACTUAL_MARKET = "C01Q1"
PRICE_CUT_PRODUCT = "F1B04"
SEED = 0
FIXED_EFFECTS = "C(market_ids) + C(product_ids)"
OPTIMIZATION = pyblp.Optimization("trust-constr", {"gtol": 1e-8, "xtol": 1e-8})


def load_data(data_dir):
    """Read product and consumer data and build the income-based instrument."""
    products = pd.read_csv(data_dir / "cereals_PS2.csv")
    products["market_size"] = products["city_population"] * 90
    products["market_share"] = products["servings_sold"] / products["market_size"]

    demographics = pd.read_csv(data_dir / "demographics_PS2.csv")
    demographics = demographics.rename(columns={"market": "market_ids"})
    income = demographics.groupby("market_ids", as_index=False).agg(
        log_income_mean=("log_income", "mean"), log_income_std=("log_income", "std")
    )

    products = products.merge(income[["market_ids", "log_income_mean"]], on="market_ids")
    products["demand_instruments1"] = products["log_income_mean"] * products["mushy"]

    agents = demographics[["market_ids", "log_income"]].groupby(
        "market_ids", as_index=False
    ).sample(n=1000, replace=True, random_state=SEED)
    agents["nodes0"] = 1
    agents["weights"] = 1 / agents.groupby("market_ids")["market_ids"].transform("size")
    return products, agents, demographics


def estimate_observed_heterogeneity(products, agents):
    """Question 1: mushy x log income, no unobserved heterogeneity."""
    formulations = (
        pyblp.Formulation("0 + prices", absorb=FIXED_EFFECTS),
        pyblp.Formulation("0 + mushy"),
    )
    problem = pyblp.Problem(formulations, products, pyblp.Formulation("0 + log_income"), agents)
    return problem.solve(sigma=0, pi=1, method="1s", optimization=OPTIMIZATION)


def estimate_plain_logit(products):
    """Benchmark logit with fixed effects, used for the counterfactual comparison."""
    problem = pyblp.Problem(pyblp.Formulation("0 + prices", absorb=FIXED_EFFECTS), products)
    return problem.solve(method="1s")


def first_stage(products):
    """First-stage regression of price on the excluded instrument and fixed effects."""
    model = smf.ols(f"prices ~ 0 + demand_instruments0 + {FIXED_EFFECTS}", products)
    return model.fit(cov_type="HC0")


def add_share_change(table, results, column):
    """Predict shares at the new prices and store the percent change from observed."""
    shares = results.compute_shares(market_id=COUNTERFACTUAL_MARKET, prices=table["new_prices"])
    table["new_shares"] = shares
    table[column] = 100 * (table["new_shares"] - table["shares"]) / table["shares"]
    return table


def counterfactual_table(products):
    """Set up the price cut for one product in one market."""
    columns = ["product_ids", "mushy", "prices", "shares"]
    table = products.loc[products["market_ids"] == COUNTERFACTUAL_MARKET, columns].copy()
    table["new_prices"] = table["prices"]
    table.loc[table["product_ids"] == PRICE_CUT_PRODUCT, "new_prices"] /= 2
    return table


def estimate_random_coefficients(products, agents):
    """Question 3: random coefficient on price and price x log income."""
    products = products.copy()
    products["demand_instruments2"] = products["IV2"]
    products["demand_instruments3"] = products["IV3"]
    formulations = (
        pyblp.Formulation("0 + prices", absorb=FIXED_EFFECTS),
        pyblp.Formulation("0 + mushy + prices"),
    )
    problem = pyblp.Problem(formulations, products, pyblp.Formulation("0 + log_income"), agents)
    results = problem.solve(
        sigma=[[0, 0], [0, 1]], pi=[[0.2], [1]], method="1s", optimization=OPTIMIZATION
    )
    return results, products


def estimate_with_micro_moment(products, agents):
    """Question 5: add a micro moment for mean log income of buyers."""
    formulations = (
        pyblp.Formulation("0 + prices", absorb=FIXED_EFFECTS),
        pyblp.Formulation("1 + mushy + prices"),
    )
    problem = pyblp.Problem(formulations, products, pyblp.Formulation("0 + log_income"), agents)

    weights = lambda t, p, a: np.einsum("i,j", np.ones(a.size), np.ones(p.size))
    values = lambda t, p, a: np.einsum("i,j", a.demographics[:, 0], np.ones(p.size))
    dataset = pyblp.MicroDataset("Income Survey", 1, weights, market_ids=["C01Q1", "C01Q2"])
    part = pyblp.MicroPart("E[log income i | j > 0]", dataset, values)
    moment = pyblp.MicroMoment("E[log income i | j > 0]", 10, part)

    return problem.solve(
        sigma=[[0, 0, 0], [0, 0, 0], [0, 0, 1]],
        pi=[[1], [0.1], [-6]],
        method="1s",
        optimization=OPTIMIZATION,
        micro_moments=[moment],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pyblp.options.digits = 3
    pyblp.options.verbose = False
    pd.options.display.precision = 3
    pd.options.display.max_columns = 50

    products, agents, demographics = load_data(args.data_dir)

    observed = estimate_observed_heterogeneity(products, agents)
    logit = estimate_plain_logit(products)
    stage1 = first_stage(products)

    table = counterfactual_table(products)
    table = add_share_change(table, logit, "iv_change")
    table = add_share_change(table, observed, "mushy_change")

    agents[["nodes0", "nodes1", "nodes2"]] = np.random.default_rng(seed=SEED).normal(
        size=(len(agents), 3)
    )
    random_coefficients, products = estimate_random_coefficients(products, agents)
    table = add_share_change(table, random_coefficients, "rc_change")

    micro = estimate_with_micro_moment(products, agents)

    (args.output_dir / "counterfactual_results.txt").write_text(
        table.to_string(index=False), encoding="utf-8"
    )
    summary = {
        "mean_log_income": demographics["log_income"].mean(),
        "first_stage_coefficient": stage1.params["demand_instruments0"],
        "q1_beta_price": observed.beta[0, 0],
        "q1_beta_price_se": observed.beta_se[0, 0],
        "q1_pi_mushy_income": observed.pi[0, 0],
        "q3_beta_price": random_coefficients.beta[0, 0],
        "q3_beta_price_se": random_coefficients.beta_se[0, 0],
        "q3_pi_mushy_income": random_coefficients.pi[0, 0],
        "q3_pi_price_income": random_coefficients.pi[1, 0],
        "q3_sigma_price": random_coefficients.sigma[1, 1],
        "q5_beta_price": micro.beta[0, 0],
        "q5_pi_price_income": micro.pi[2, 0],
        "q5_sigma_price": micro.sigma[2, 2],
        "price_cut_own_change_logit": table["iv_change"].iloc[0],
        "price_cut_own_change_observed": table["mushy_change"].iloc[0],
        "price_cut_own_change_random_coefficients": table["rc_change"].iloc[0],
    }
    frame = pd.Series(summary, name="value").rename_axis("quantity").reset_index()
    frame.to_csv(args.output_dir / "estimates_summary.csv", index=False)

    with open(args.output_dir / "model_output.txt", "w", encoding="utf-8") as handle:
        for title, results in [
            ("Question 1: observed heterogeneity", observed),
            ("Question 3: random coefficients", random_coefficients),
            ("Question 5: micro moment", micro),
        ]:
            handle.write(f"{title}\n{results}\n\n")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
