# wharton-monte-carlo-allocator

A Monte Carlo simulator that tests six-asset portfolio allocations under yearly withdrawals, made for the Wharton Investment Competition.

**For reviewers:** Python · NumPy · pandas · solo project

## Problem

The competition asked for an allocation across Emerging Markets, Japan, International Developed, Bonds, Gold, and US Equities. A single expected-return number doesn't show how often a portfolio ends up below where it started once withdrawals and drawdowns are included, so I wanted to simulate many possible paths and compare allocations on those outcomes.

## Approach

- **Returns:** monthly returns are drawn from a multivariate normal distribution, using means and covariances estimated over rolling windows (24 months by default) of a monthly returns series. That series is either synthetic, generated from assumed means, volatilities, and a correlation matrix, or loaded from a CSV of historical returns.
- **Portfolio rules:** a $12,000 withdrawal is taken at each year end starting in year 3, along with a simple dividend-tax drag (2% assumed yield). Weights are checked every 3 months and rebalanced only if one has drifted more than 5% from its target.
- **Search:** every allocation on a 5% grid that meets configurable caps and floors is simulated across a process pool. Each run records mean, median, and 5th-percentile ending value, the chance of ending above the starting value, the chance of reaching a 3x target, Sharpe and Sortino ratios, and max drawdown.
- **Tradeoff:** to keep the grid search fast, `run_monte_carlo` averages the rolling-window statistics into a single mean and covariance and simulates all paths in one batch. A slower per-path version that samples a different window each month (`simulate_portfolio`) is also included.

## Results

These are taken from the result CSVs committed to the repo, which came from runs on synthetic data. The current defaults are a $350,000 start, a $12,000 yearly withdrawal, a 10-year horizon, and a $1,350,000 "3x" target. The CSVs don't record the exact settings each run used.

| File | Allocation | Ends above start | Reaches 3x target | Mean ending value |
|------|------------|------------------|-------------------|-------------------|
| `top_50_strategies.csv` (Emerging Markets ≥ 40%) | 40% EM / 5% Bonds / 55% US | 75.0% | 3.2% (best in file) | ~$729k |
| `top_50_strategies.csv` | 40% EM / 25% Bonds / 35% US | 83.6% (best in file) | 2.6% | ~$780k |
| `top_50_LOOSEconstraints.csv` (no EM floor) | 100% US | 90.4% | 6.6% (best in file) | ~$896k |
| `top_50_LOOSEconstraints.csv` | 40% Bonds / 5% Gold / 55% US | 99.65% (best in file) | 4.55% | ~$1.01M |
| `specificSIM.csv` | ~54% EM / ~46% US | 55.4% | 0.44% | ~$424k |

Across these runs, reaching 3x in 10 years is unlikely under these assumptions. Adding bonds mostly improves the chance of ending above the starting value and reduces drawdowns. `portfolio_optimization_results.csv` comes from an earlier, low-iteration run and is only useful for exploration.

## How to run

**Prerequisites:** Python 3 with pip (tested on Python 3.12).

```bash
git clone https://github.com/ZishaanM/wharton-monte-carlo-allocator.git
cd wharton-monte-carlo-allocator
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

**1. Simulate a specific allocation (default, takes a few seconds):**

```bash
python portfolio_optimizer.py
```

With the defaults (`RUN_SPECIFIC_ALLOCATIONS = True`, `USE_REAL_DATA = False`), this simulates the allocations in `SPECIFIC_ALLOCATIONS` and writes `specificSIM.csv`. The console prints the configuration and a short summary for each allocation.

**2. Run the full grid search:** in `portfolio_optimizer.py`, set `RUN_SPECIFIC_ALLOCATIONS = False`, then run it again. The results go to `top_50_LOOSEconstraints.csv`, sorted by `Success_3x_Pct`. That filename is fixed in the code, so rename it if you want to keep separate runs.

All settings are constants at the top of `portfolio_optimizer.py`:

- `INITIAL_VALUE`, `ANNUAL_WITHDRAWAL`, `YEARS`, and `TARGET_FINAL_VALUE`
- `ITERATIONS` (paths per allocation) and `WINDOW_SIZE` (rolling window in months)
- `REBALANCE_ENABLED`, `REBALANCE_CHECK_FREQUENCY`, and `REBALANCE_THRESHOLD`
- `MIN_EMERGING_MARKETS`, `MAX_JAPAN`, `MAX_INTL_DEVELOPED`, `MAX_BONDS`, and `MAX_GOLD` (grid constraints)
- `EXPECTED_RETURNS`, `VOLATILITIES`, and `CORRELATION_MATRIX` (monthly assumptions for synthetic data)

**3. Optional helper scripts:**

```bash
python validate_correlation_matrix.py   # checks the correlation matrix; saves correlation_matrix_input.png and correlation_matrix_comparison.png
python analyze_covariances.py           # prints rolling-window covariance stats for synthetic data
python analyze_results.py               # ranks portfolio_optimization_results.csv; writes analysis_top_50.csv and strategy_summary.csv
python analyze_drawdowns.py             # drawdown summary of portfolio_optimization_results.csv
python custom_allocation_runner.py      # success rate at a $1.5M target for a hand-picked list; writes custom_allocation_success.csv
```

`analyze_results.py` and `analyze_drawdowns.py` read the older `portfolio_optimization_results.csv` format by default. `analyze_drawdowns.py` needs its `Median_Max_Drawdown` column, which newer outputs don't include.

**Using historical data:** set `USE_REAL_DATA = True` and put a `historical_asset_returns.csv` file in the repo root. It needs a `Date` column followed by `Emerging_Markets, Japan, Intl_Developed, Bonds, Gold, US_Equities`, with monthly returns as decimals and no blank cells. That file isn't included. The `Monte Carlo Results - 6-asset model*.csv` files contain partial historical series: Bonds, Gold, and US Equities from 1929, with much shorter histories for the other three assets. They use different column names, so they can't be loaded as-is. `create_historical_csv_RUNME.py` writes an empty template to fill in.

**Gotchas:**

- Running the optimizer overwrites the committed `specificSIM.csv` and `top_50_LOOSEconstraints.csv`, so copy them first if you want to compare.
- The default grid constraints produce about 40,000 allocations. At `ITERATIONS = 10000`, that takes hours on a laptop. Tighten the constraints or lower `ITERATIONS` for a quick test.
- The simulation isn't seeded (only the synthetic history is), so reruns vary slightly. For example, `specificSIM.csv` reran at about $420k mean ending value against the committed $424k.
- `monte_carlo_portfolio.ipynb` is an unfinished stub. Use the `.py` scripts instead.

## What's interesting in the code

- **`monte_carlo_portfolio.py`, `PortfolioSimulator.run_monte_carlo`:** simulates all paths at once as NumPy arrays (iterations × months × assets). Look at how year-end withdrawals, depleted portfolios, and threshold rebalancing are handled inside one monthly loop, and how max drawdown is computed from a running maximum.
- **`portfolio_optimizer.py`, `main` and `_worker_simulate`:** builds the constrained 5% weight grid with nested loops in integer steps. It sends the returns data to each worker process once, through a pool initializer, instead of with every task.

## Stack

Python · NumPy · pandas · Matplotlib · tqdm · multiprocessing · seaborn (optional)
