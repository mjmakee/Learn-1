# Audited deposit attribution v2

`attribution_v2.py` supports both reporting modes:

1. **Scenario pair:** CCAR Base vs Business Outlook (or any two scenarios with the same projection dates).
2. **Cycle over cycle:** `[Stress − Base] current cycle − [Stress − Base] previous cycle`, with separate NII and margin bridges.

## Input columns

`month,scenario,ftp_dollars,interest_expense_dollars,total_avg_balance,dda_avg_balance,ib_avg_balance,fed_funds_rate` for single-cycle data. Add `cycle` for multi-cycle data. Column names can be mapped in `COLUMN_MAP` at the top of the script. One row per month, scenario and cycle; monthly dates are first of month.

## Commands

```
pip install numpy pandas matplotlib
python attribution_v2.py --input sample_deposits.csv --mode pair --a Baseline --b Severe --out pair_output
python attribution_v2.py --input sample_deposits_cycles.csv --mode cycle --prior-cycle 2025 --current-cycle 2026 --base Baseline --stress Severe --out cycle_output
```

By default, `--day-count fixed` uses 1/12 per month: for 27 months the full-horizon margin denominator is `2.25 × mean monthly average deposit balance`. Switch to `--day-count actual` for monthly ACT/365 accrued exposures if that is your internal convention. Make sure the input FTP and expense dollar projections are consistent with their stated accrual period.

## Mathematical specification

Let `B = total average deposit balance`, `w = DDA/B`, `F = FTP_dollars/(B*tau)`, `r = interest_expense_dollars/(IB*tau)`. For any month:

`NII = tau * B * [F − (1 − w)*r]`.

Horizon margin: `M = sum(NII)/sum(tau*B)`.

The Shapley decomposition switches the full 27-month vectors of `B`, `w`, `F`, and `r` between two scenarios across all 24 orders, averages their contributions, and reconciles exactly to the differences in NII dollars and full-horizon margin bps. Balance contribution to margin arises only through changing weights across months, not uniform level scaling. DDA has no expense and all interest expense is attributed to IB. FTP yield is aggregate blended, so the mix/FTP economic split is conditional; it is **not** an identified FTP product mix effect.

For cycle over cycle, the code computes two independent, exact `Stress − Base` bridges and subtracts corresponding effects. These differences also reconcile exactly to the change in stress impact. Periods are aligned within each cycle; cross-cycle contributions are compared by month index (month 1 of the horizon, etc.). This is a reporting attribution, not causal estimation.

Normalized cycle stress impact (bps annualized) is `10,000 * (NII_stress − NII_base) / baseline deposit dollar-years`. This differs from the stress minus baseline margin gap if scenario balances differ.

## Output

The pair mode produces 2 waterfalls (`pair_nii.png`, `pair_margin.png`), CSV factor contributions, and monthly factor contributions. Cycle mode produces 2 direct cycle-change waterfalls (`cycle_change_nii.png`, `cycle_change_margin.png`), the two component stress-vs-base analyses, and `cycle_summary.csv` with dollar and normalized bps impacts.

## Cautions

- Actual deposit balance equals DDA plus IB for this version.
- A distinct causal balance/mix/FTP decomposition is not identified with only aggregate FTP. The model's allocation is a transparent, internally exact counterfactual convention.
- For cycle-year comparisons with differing calendar dates, month indices are treated as the same forecast horizon positions. Do not interpret this as controlling for changes in macroeconomic environment or starting portfolio unless explicitly adjusted.
