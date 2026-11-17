# Factory Decision Lab — Optimizer Edition

**Separate repository version.** The original `Khushi-314/SF-VV1` repository is not modified.

## Install and launch

```bash
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pytest -q
streamlit run app.py
```

To deploy on Streamlit Community Cloud, create **a new GitHub repository** (for example `SF-VV2-Optimizer`), upload this folder's contents to its root, then point a **new** Streamlit app at `app.py`. Do not change the existing Streamlit deployment or its GitHub repository.

## One-click workflow

1. Enter PR1–PR4 quantities (up to 200,000 per product), machine reliability, repair distributions, machine buffers, and desired Monte Carlo replications.
2. Optionally enter **up to three distinct WIP limits** for additional comparison. Blank/zero fields mean no extra comparison.
3. Click **Simulate & Recommend**. The app downloads a single live-sheet snapshot and fits the existing product-specific last-five Normal distributions.
4. It estimates a theoretical idealized critical WIP, evaluates a WIP–throughput and WIP–cycle-time curve, and selects a provisional operating WIP using a 95%-of-best-throughput rule.
5. It screens reliability (+25%, +50% MTTF), corrective repair (-25%, -50% mean duration), and per-machine buffer (+1, +2 capacity) changes. It refines ten candidates while ensuring representation of all three intervention categories.
6. It runs the user's requested full Monte Carlo count for the recommended WIP and any extra WIP values, caching all outputs in the Streamlit session.
7. Choose which WIP to inspect. **All original simulation charts and numerical outputs** switch together, while the Normal parameter and rolling-validation tabs continue to show the *same* underlying frozen sheet calibration, as they should.

The original tabs remain: **Simulation results**, **Processing-time parameters**, and **Live data & validation**. A fourth **Optimization & recommendations** tab contains WIP curves, idealized benchmark, recommendations, uncertainty, and Little's Law accounting.

## Limits and honest interpretation

- This is **finite-order**, not calendar-horizon, optimization. It is valid for small and large requested batches subject to runtime/event safeguards. A ~2,000-hour example is made by choosing appropriate quantities; 2,000 hours is not hardcoded.
- **WIP search is heuristic**, not a mathematical proof of global optimality. It starts from an idealized Factory Physics estimate; bounds search to 96 WIP slots by default, or fewer than total products; refined Monte Carlo candidates may still be noisy. If the throughput curve does not plateau, the website warns the user.
- Current intervention search uses **4 screening replications** per candidate and **12 refinement replications** for selected candidates by default; this is intentionally a provisional screening tool, not proof of superiority for small effects. Results include approximate 95% Monte Carlo intervals. Use larger study-specific replication budgets before making high-stakes decisions.
- "Buffer +1" means adding one waiting place to a machine. It doesn't imply this is physically or financially feasible.
- MTTF increases represent **hypothetical reliability engineering outcomes**, not a service schedule. No periodic preventive maintenance is simulated or recommended. Costs are not included.
- Existing source data and stage allocation assumptions remain unchanged. Machine specific prioritization is conditional on those assumptions.
- Little's Law is used only as an **internal conservation check**, not external factory validation.
- Monte Carlo mean-utilization charts display confidence-interval error bars and preserve mean processing percentages and absolute durations.
- Diagnostic event log is capped at 5,000 rows. Per-product raw lead-time arrays are bounded by a 3,000-value uniform reservoir in detailed runs; **exact lead-time sums/counts and means** remain available for all runs. P95 product lead time is an approximation when sampling occurs. Non-detailed runs omit individual lead samples. WIP event traces are bounded and may be decimated for large batches, but exact integrated mean WIP is unchanged. The per-run WIP curves reflect all display runs, using bounded traces. All these changes protect memory.
- Source input has a built-in simulation event cap of 4 million events per replication and a large elapsed-time guard. A capacity-limited run may still be computationally expensive; the program reports errors instead of fabricating results.
- The UI was syntax-checked locally; backend tests passed, but the **Streamlit browser interface was not exercised locally**, because Streamlit is not installed in the available testing environment. Verify on a new Streamlit deployment before relying on it.

## Key code files

- `app.py` — preserved tabs plus new fourth tab and one-button workflow
- `optimizer.py` — adaptive WIP exploration, WIP data, single-intervention ranking
- `simulation.py` — original event logic with bounded detail storage, per-run lead sums and Monte Carlo utilization confidence intervals
- `calibration.py`, `reliability.py`, `config.py` — copied unchanged from original
- `tests/test_factory.py`, `tests/test_optimizer.py` — original and new regression/large-order tests

## Important statistical nuance

Using the same random seed across scenarios is helpful but does not guarantee perfect common-random-number matching, because changing a machine's reliability or buffer can change the order and number of random draws. The reported CI on per-run matched-index differences should be treated as provisional until a disciplined common-random-stream or independent-replication comparison protocol is implemented.
