# Factory Decision Lab — Optimizer Edition

**New, separate-repository build. The original SF-VV1 is untouched.**

See [NEW_REPO_SETUP.md](NEW_REPO_SETUP.md) for installation, interface, validation, limitations and repository setup.


A Streamlit discrete-event simulation for a smart-factory teaching/research demonstration. **The simulator uses a real published Google Sheet, product-specific rolling calibration and Monte Carlo replications.** Individual machine reliability assumptions can be changed without modifying recorded processing data.

## Quick start

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Click **Run simulation**. The app fetches the latest Google Sheets CSV on every click, recalibrates 5 observations per product (20 rows total with complete data), and runs N Monte Carlo replications. It **does not silently substitute a cached dataset if the fetch fails**.

The source URL is in `config.py`. A 36-order reference snapshot is included at `data/sample_observations.csv` **solely for offline engineering tests**; the website does not load that snapshot automatically.

## UI

1. **Simulation results:** configurable PR1–PR4 quantities, global WIP, machine buffers, Monte Carlo run count, individual Weibull failure + repair parameters, and aggregate KPIs and charts.
2. **Processing-time parameters:** exact product × machine Normal mean/SD fitted for the executed run; latest five selected records per product; reliability parameter table; changes since previous run.
3. **Live data & validation:** source and version metadata, product-specific five-observation rolling out-of-sample distribution checks, 90% interval coverage, deviation from fitted mean, and raw observations.

All processing times, MTTF, eta, makespan and downtime have explicit units. Stacked state shares show **mean percentages alongside mean absolute minutes per run**; no deceptive universal second axis is drawn for runs with different makespans.

## Factory and assumptions

- Routing: `WS1 → WS2(MC1/MC3) → WS3(MC2/MC4) → WS4 → WS5(MC5) → WS6(MC6/MC7) → WS7(MC8/MC9) → WS8`.
- Product release is cyclic PR1 → PR2 → PR3 → PR4, skipping any exhausted types.
- Each paired workstation alternates successive assignments starting from its lower-numbered machine. **Strict alternation:** if a selected machine is unavailable or has no queue room, the job waits upstream instead of being reassigned. Each machine processes at most one product, FIFO buffers are separate, and WIP is a global admitted-but-unfinished limit.
- Processing: product-specific **positive-truncated Normal**. Arithmetic sample mean/SD estimated from the last 5 valid chronological observations per product. These are Normal-family inputs with rejection of nonpositive durations.
- The live sheet measures WS2/WS3 **workstation durations, not separate machine durations**. Both machines in a pair receive the respective product-specific station distribution. This is an assumption, not independent proof for each physical machine.
- WS5–WS7 is an **aggregate** source measurement. The site uses fixed, indicative allocation shares: `WS5=52/84`, `WS6=27/84`, `WS7=5/84`, based on earlier approximate times. It samples the total once per simulated product and splits it; durations are therefore correlated and sum to their aggregate sample. Separate per-machine measurements are **not claimed**. The shares are not experimentally identified and machine-specific bottleneck conclusions depend on them.
- The final modeled station is `WS8`, directly matched to the observed `WS8` column in the Google Sheet.
- Individual MC1–MC9 Weibull failure clocks advance **only while processing**, not while idle. A failure raises an instantaneous `B7 Error/Alarm` event and enters `B5 Under maintenance`; repair times are lognormal; repair is perfect and interrupted processing resumes at the unprocessed remainder. Manual workstations do not fail.
- Source machine states from `States_filled (2) (3).xlsx`: B1 Idle, B2 Setup, B3 Processing, B4 Off, B5 Under maintenance, B6 Waiting for machine, B7 Error/Alarm, B8 Not applicable. The simulator only assigns duration to B1, B3, B5, B6. B7 is an event (zero duration), and B2/B4/B8 are not invented. In the simulation, B6 means a resource holds a finished job pending transfer downstream; this is an *operational interpretation* of the supplied label.
- Weibull options are only `MTTF + β → η` and `η + β → MTTF` (unique valid solutions). The UI does not offer ambiguous `MTTF + η → β`.
- Monte Carlo summaries aggregate across runs. The diagnostic event log is deliberately **only the first run** and is labelled as such. There is no falsely averaged event timeline.
- WIP-versus-time averages across all runs and treats completed replications as zero WIP thereafter.

## Rolling validation

For each product and each of the six *observed station columns*, chronology is determined by the date embedded in order ID `ORddMMyyyyHHmmss`. A Normal distribution is fitted using observations 1–5 and then checked against the held-out observation 6; the window shifts to 2–6 and is checked against observation 7, etc. A 9-observation product yields four held-out checks per duration column. The plot compares held-out observations with fitted means and central 90% model intervals. No future data enters the calibration window. **This is a processing-time distribution check, NOT a separate forecasting model and NOT validation of machine reliability or whole-factory makespan.**

## Testing

```bash
python -m pip install pytest
python -m pytest -q
```

Tests cover reading the 36-row snapshot, 48 resource/product input pairs, 96 rolling validation windows, cyclic order release, Weibull parameter equivalence, WIP/queue conservation across configurations, frequent failures and repairs, seeded reproducibility, and state duration accounting.

## Publishing and limitations

Deploy to [Streamlit Community Cloud](https://share.streamlit.io/) with main file `app.py`. Google Sheets' published CSV feed may **lag updates**. The app records the returned row count, last chronological order ID and data hash; it does not falsely say a new row was used unless it appears in the fetched CSV. Only publish **non-sensitive demonstration data**, as the CSV is public.

This is a scenario simulator with documented simplifying assumptions—not a production-validated digital twin, physical-machine failure predictor, or guarantee of client delivery times. Product-mix reliability and workstation allocation are scenario-dependent; only genuine measured quantities should be presented as empirical.


## Optimizer V2.1 changes (Nov 2026)

- Uses a bounded adaptive WIP search. Reports theoretical critical WIP separately from the simulated recommendation, actual mean WIP, and mean per-run peak WIP.
- Uses common-seed Monte Carlo screening and equal-count refinement among shortlisted interventions. The paired confidence intervals estimate sampling precision and **do not** establish physical reliability or external validity.
- Intervention analyses are independent what-if scenarios: MTTF +25%/+50%, repair mean -25%/-50%, or buffer +1/+2. There is **no** preventive maintenance scheduling, cost optimization, or combined intervention search.
- An online workload heuristic reduces the candidate catalogue for huge orders. The output discloses how many of the 54 candidate interventions were evaluated. Search limits are **not global guarantees**.
- The original three tabs remain; the separate optimization tab adds the actual-WIP chart and baseline bottleneck diagnostics.
- Large-scale optimization on shared Streamlit infrastructure is not benchmarked or guaranteed to complete. Offline Python may be required for ~2,000-hour production studies.
- Open `NEW_REPO_SETUP.md` for repository deployment. This package is intended for the **new** `SF-VV2-Optimizer` only. Never overwrite `SF-VV1`.
