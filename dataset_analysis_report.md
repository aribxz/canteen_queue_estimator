# Smart Canteen Queue Estimator — Dataset Analysis Report

**Dataset file analysed:** `queue.csv` in this folder (copy of
`queue_waiting_time_large_dataset.csv` from
[Queue-Waiting-Time-Prediction-System](https://github.com/ghatolenidhi/Queue-Waiting-Time-Prediction-System))

**Code:** `analyze_queue_dataset.py` (reproducible; does not modify the CSV).
Run it with `python analyze_queue_dataset.py`. It regenerates all plots in `plots/`.

**Repository documentation inspected:** the repo README (Project Overview, Problem Statement,
Dataset Description, Random Forest + Streamlit approach). The README defines the columns but
gives no data-collection methodology, no time period, no location, no sensor description,
and no discussion of real vs. simulated data.

---

## Executive summary

**Should you use this dataset? Yes — as a prototype only. No — not as a production canteen model.**

| Question | Answer |
|---|---|
| What is it? | 5,000 rows × 6 columns of queue records. Target is `Waiting_Time` (minutes). Features are queue length, open counters, average service time, hour of day, and queue type (Bank / Hospital / Office). |
| Is it clean? | Very clean: 0 missing values, 0 exact duplicate rows, no negatives, sensible ranges. |
| Is it real? | **Almost certainly synthetic/simulated, not measured in a real canteen.** Evidence: inputs are near-uniformly distributed; `Waiting_Time` correlates **r = 0.9999** with the textbook formula `Number_of_People × Avg_Service_Time / Counters_Open` plus a per-type offset (~+10 Bank / ~+15 Office / ~+20 Hospital) plus small noise (SD ≈ 7 min). The repo gives no collection protocol. Treat this as a finding with strong evidence, not a proven fact. |
| Can it predict canteen waits directly? | No. There is no canteen data, no meal/food information, and the `Bank/Hospital/Office` categories do not include a canteen. Service dynamics (batch cooking, meal-specific prep, lunch rushes, payment queues) are absent. |
| How is it still useful? | It is an excellent **scaffold**: same input-output shape your app needs (queue size + counters + service time + time → wait minutes). You can build the full pipeline (cleaning → features → baseline → evaluation → Streamlit app) now, then swap in real canteen data later. |
| Biggest modelling trap? | A model will just re-learn the hidden formula and look implausibly good (R² ≈ 0.999). That score will **not** transfer to a real canteen. Also, `Avg_Service_Time` is a leakage risk if defined as "what we observed while these people were served". |
| What to do next? | 1) Keep this data for prototyping. 2) Engineer the interaction feature `N×S/C`. 3) Start with a mean/median + linear baseline. 4) Collect real canteen data (queue size, counters, per-meal service/prep time, timestamp, actual wait). 5) Re-train and validate on time-ordered canteen data. |

All numbers below were computed from the actual CSV. Assumptions are labelled **[Assumption]** and
uncertainties **[Uncertain]**.

---

## 1. Dataset overview

### 1.1 Size and shape

- **Rows:** 5,000
- **Columns:** 6
- Verified with `pd.read_csv(...).shape`. The original dataset is not modified by any script here.

### 1.2 Columns, types, meanings

| # | Column | dtype (pandas) | Meaning (README + inspection) | Units / values |
|---|---|---|---|---|
| 1 | `Number_of_People` | int64 | Total people in queue ahead (queue length). Your app's main manual input. | count, 5–299, integer |
| 2 | `Counters_Open` | int64 | Number of service counters currently open. | count, 1–14, integer |
| 3 | `Avg_Service_Time` | int64 | Average service time per person. **[Assumption]** README says "minutes"; per-person service time in minutes is the only unit consistent with the target scale (see §5.4). | **[Likely]** minutes/person, 2–19 integer |
| 4 | `Time_of_Day` | int64 | Hour of day. README says 0–23. Data contains every integer 0–23. Should be treated as hour, not a continuous quantity. | hour-of-day 0–23 |
| 5 | `Queue_Type` | object/string | Type of service location. Only 3 values in data: `Bank`, `Hospital`, `Office`. README also mentions "Ticket Counter" but it does **not** appear in the CSV. No `Canteen` value exists. | categorical, 3 levels |
| 6 | `Waiting_Time` | float64 | **Target variable**: waiting time to be predicted. README says minutes. Scale is consistent with minutes (median ~204 min is unrealistically long for a canteen — see limitations — but consistent with the synthetic formula). | **[Likely]** minutes, 2.98–5,460.82, 2 decimals |

Descriptive statistics (actual output of `df.describe(include="all")`):

| stat | Number_of_People | Counters_Open | Avg_Service_Time | Time_of_Day | Waiting_Time |
|---|---|---|---|---|---|
| count | 5,000 | 5,000 | 5,000 | 5,000 | 5,000 |
| mean | 152.86 | 7.49 | 10.69 | 11.60 | 403.38 |
| std | 85.05 | 4.03 | 5.21 | 6.93 | 611.51 |
| min | 5 | 1 | 2 | 0 | 2.98 |
| 25% | 80 | 4 | 6 | 6 | 95.26 |
| 50% (median) | 153 | 7 | 11 | 12 | 203.55 |
| 75% | 226 | 11 | 15 | 18 | 416.32 |
| max | 299 | 14 | 19 | 23 | 5,460.82 |

Category balance: `Hospital` 1,705 / `Bank` 1,657 / `Office` 1,638 — essentially balanced.

### 1.3 Sample records (first 5 rows, verbatim)

```
Number_of_People  Counters_Open  Avg_Service_Time  Time_of_Day Queue_Type  Waiting_Time
             107              8                10           12   Hospital        161.96
             275              8                11            3   Hospital        398.24
             111              7                 5           17   Hospital        104.86
              76             10                18           23     Office        156.83
             193              4                15           22   Hospital        747.52
```

### 1.4 Target variable

`Waiting_Time` is unambiguously the target: the README labels it "Target variable – waiting time (minutes)",
it is the only float column with 2-decimal precision, and the repo's `app.py` predicts it from the other five inputs.

---

## 2. Data quality assessment

| Check | Method | Result |
|---|---|---|
| Missing / null | `df.isnull().sum()` | **0 missing in every column.** |
| Exact duplicate rows | `df.duplicated().sum()` | **0.** |
| Near-duplicate feature rows (same 5 inputs, different target) | `df.drop(columns=["Waiting_Time"]).duplicated().sum()` | **3 pairs** (see below). No exact feature-target duplicates. |
| Wrong dtypes | inspection | All numeric columns are integers except target (float) — plausible. `Time_of_Day` is stored as int but is semantically an hour/cyclical category, not a true numeric quantity. `Queue_Type` is string — correct. |
| Inconsistent categorical values | `value_counts()` + case/whitespace check | Only `Bank`, `Hospital`, `Office`. No misspellings, no extra spaces, no hidden case variants. Note: README mentions "Ticket Counter" — absent from data. |
| Invalid values (negatives, zeros) | range check | **None negative.** `Counters_Open` min = 1 (zero would mean division by zero — correctly absent). `Number_of_People` min = 5 (no empty-queue rows). `Avg_Service_Time` min = 2 (no zero-service rows). `Waiting_Time` min = 2.98 (no negative waits). |
| Outliers | IQR rule on target | 522 rows (10.4%) above upper bound 897.91. **Interpretation:** these are not errors — they are the structural tail produced when `Counters_Open = 1` (see §3). Skew = 3.82, kurtosis = 18.42 (heavy right tail). |
| Real vs. synthetic | distributions + formula test + docs | **Strong evidence of synthetic generation** (see §2.1). No collection methodology in repo. |

### 2.1 Why this looks synthetic (evidence, not assumption)

1. **Inputs are near-uniform** — the fingerprint of `random.randint` generation, not human queues:
   - `Time_of_Day`: every hour 0–23 appears 183–234 times (expected 208.3 under uniform). Real queues spike at lunch/opening hours.
   - `Counters_Open`: every value 1–14 appears 330–398 times. Real counters follow staffing shifts.
   - `Number_of_People` spans 5–299 with 295 distinct integers and flat frequencies (~26–29 per value at the top). Real arrival counts are bursty/Poisson-like.
   - `Avg_Service_Time` covers every integer 2–19 with 286–323 rows each.
2. **Target is a near-deterministic function of inputs.** Defining `naive = Number_of_People × Avg_Service_Time / Counters_Open` (textbook queue approximation, computed by us for diagnosis — not a dataset column):
   - Pearson `corr(naive, Waiting_Time) = 0.9999`.
   - `Waiting_Time − naive` has mean 15.19, SD 8.11, and splits cleanly by type: Bank +10.18 ± 7.00, Office +15.18 ± 7.07, Hospital +20.05 ± 7.02. That is: `target ≈ naive + type_offset + small noise`.
   - Plot `plots/13_waiting_vs_naive_formula.png` is essentially a straight line; `plots/14_residual_by_queuetype.png` shows three offset humps.
   - Extreme "ratio" rows (e.g. 5 people, 14 counters, waiting 34.85 min, ratio 32.5) are not anomalies — with tiny `naive` (≈1.07), the +20 Hospital offset dominates. This additive-offset behaviour is exactly what a generator script would produce.
3. **No provenance.** The README describes the Streamlit app and Random Forest but never says who measured what, where, when, or with which instrument. A real 5,000-row multi-site queue study would document this.

**Conclusion:** treat the CSV as simulated practice data. This does not make it useless (see §4), but any accuracy measured on it is an upper bound that will fall on real canteen data.

### 2.2 Near-duplicate pairs (train/test contamination risk)

Three input combinations occur twice with different targets:

| inputs (People, Counters, Svc, Hour, Type) | Waiting_Time (row A) | Waiting_Time (row B) |
|---|---|---|
| 86, 4, 10, 13, Hospital | 235.10 | 237.94 |
| 121, 13, 2, 18, Office | 46.36 | 32.75 |
| 196, 5, 3, 23, Hospital | 125.72 | 150.24 |

With random numeric ranges, exact input collisions are rare — these 3 pairs are consistent with independent uniform sampling plus noisy targets. Risk is small (6 rows / 5,000) but real: a random split can place one twin in train and one in test. Deduplication or grouping is cheap insurance (see §6).

---

## 3. Exploratory data analysis (EDA)

All charts are in `plots/` and reproducible via `analyze_queue_dataset.py`. Key numbers are real computed values.

### 3.1 Distribution of waiting times — `plots/01–03`

- Highly right-skewed: mean (403.4) ≈ 2× median (203.6). 25th pct 95.3, 75th pct 416.3, max 5,460.8.
- **Plain English:** most waits are 100–400 minutes in this fictional world, but a long tail stretches past 1,000 whenever only 1 counter is open. For a canteen (where 5–20 min is normal) these absolute numbers are unrealistic — use the *patterns*, not the magnitudes.
- The zoomed histogram (`02`) shows the bulk clearly; the boxplot (`03`) is a single line with a forest of outlier dots — the classic picture of a `1/Counters` tail.

### 3.2 Distribution of queue lengths — `plots/04`

- Roughly flat from 5 to 299 (mean 152.9, median 153). No lunch peak, no closing dip.
- **Plain English:** every queue size is equally likely in this dataset. Your canteen will not look like this — expect spikes at 12:30–14:00. Do not mistake this flatness for reality.

### 3.3 Waiting time vs. queue length — `plots/05`

- Positive but diffuse cloud (Pearson r = +0.37). The spread fans out as queues grow.
- **Plain English:** longer queues usually wait longer, but queue length alone is a weak predictor because counters and service speed matter just as much. A 250-person queue with 12 fast counters can beat a 60-person queue with 1 slow counter.

### 3.4 Waiting time vs. active counters — `plots/06–07`

- Strong inverse relationship (r = −0.52, the strongest single linear signal). Median wait collapses as counters rise; `Counters_Open = 1` produces all the 2,000+ minute monsters.
- **Plain English:** opening another counter is the fastest way to cut waits — the effect is hyperbolic (going 1→2 helps far more than 10→11). Your app should surface this to canteen staff, not just students.

### 3.5 Waiting time vs. average service time — `plots/08`

- Positive (r = +0.31). Slower per-person service → longer waits, roughly proportionally.
- **Plain English:** shaving seconds off each transaction (faster payment, pre-plated meals) multiplies across the whole queue. In a canteen this is where meal choice matters — but this dataset has no meal concept (see §4).

### 3.6 Waiting time across time-of-day — `plots/09–10`

- No pattern. Hourly means wander 342–499 min with no lunch/dinner peaks; boxplots per hour look identical; r(Time, Wait) = +0.01.
- **Plain English:** the "hour" column is decorative in this dataset — it carries no signal. In a real canteen, hour will be one of your most predictive features. Do not conclude "time doesn't matter" from this data.

### 3.7 Correlations between numerical features — `plots/11`

| pair | r | reading |
|---|---|---|
| Counters ↔ Wait | −0.52 | strong inverse, real |
| People ↔ Wait | +0.37 | moderate positive |
| Service ↔ Wait | +0.31 | moderate positive |
| Time ↔ Wait | +0.01 | none |
| People ↔ Counters / Service ↔ others | ≈ 0 | inputs are independent by construction (again, synthetic fingerprint) |

**Important nuance for beginners:** pairwise correlations understate the true structure because the real rule is multiplicative (`People × Service / Counters`). A plain linear model on raw columns only reaches R² ≈ 0.50 (MAE ≈ 266 min in-sample) — not because the data is noisy, but because a straight line cannot bend into a `1/x` hyperbola. Once you add the interaction feature, R² jumps to ≈0.999. Lesson: engineer the physics before blaming the algorithm.

### 3.8 Differences across categorical features — `plots/12`

- Raw means: Bank 389.8, Hospital 402.9, Office 417.6 — small gaps relative to SD ≈ 600.
- **Plain English:** queue *place* barely shifts the average wait in absolute terms. Its real role is the hidden +10/+15/+20 minute offset inside the generator (visible only after subtracting `naive`). For your canteen, `Queue_Type` as coded here is unusable — there is no "Canteen" level, and a hospital offset must not be applied to dosa lines.

---

## 4. Suitability for the Smart Canteen project

Your app plan: student picks a meal + enters current queue size; app auto-fills time and (if available) active counters.

### 4.1 Which dataset features are useful for predicting waiting time?

| Feature | Predictive in this data? | Canteen analogue |
|---|---|---|
| `Number_of_People` | Yes (core of formula) | Direct: student-entered queue size. Keep. |
| `Counters_Open` | Yes (strongest linear signal) | Direct: staff-entered or sensor count of serving windows. Keep. |
| `Avg_Service_Time` | Yes (core of formula) | **Problematic** (see 4.3): needs per-meal prep + payment timing you don't have yet. |
| `Time_of_Day` (hour) | No signal here, but *will* matter in reality | Auto from device clock. Keep for real data; expect it to become predictive. |
| `Queue_Type` | Only as generator offset | **Not transferable.** No canteen level. Drop or replace with your own site/meal-station identifier. |

### 4.2 What can the app realistically collect?

- **Queue size:** manual student input (photo/count button; accept ±2–3 error and design for it).
- **Time:** automatic (`datetime.now()` — free, reliable, no user effort).
- **Counters open:** semi-automatic — a staff toggle in the app, or a default per shift with manual override. Do not assume students know this.
- **Actual waiting time (label):** needs a "I got my food" button or staff confirmation to log ground truth for future training. Without this you can never leave the synthetic world.

### 4.3 What needs manual input or new sensors?

- **`Avg_Service_Time`:** the hardest field. In this dataset it is handed to you per row. In a canteen it depends on meal (maggi vs. thali), batch readiness, payment mode, and staff speed. Options: (a) start with a per-meal default table timed by stopwatch study; (b) estimate it as total-service-time ÷ people-served over the last 15 min; (c) add a low-cost clicker/IR beam at the counter. **Do not ask students to guess "average service time"** — they can't.
- **Meal identity:** entirely absent here; requires a menu dropdown + per-meal prep-time table you build yourself.
- **Queue dynamics:** arrival rate, balking (students leaving), batch completions — none in this data; needs observation shifts.

### 4.4 Is there meal-specific preparation information? Can we do meal-specific predictions?

**No.** The CSV has no dish, no ingredients, no prep time, no veg/non-veg flag, no station ID. `Avg_Service_Time` is a single blended number. Therefore:

- With this data alone: **general queue waiting-time predictions only.**
- Meal-specific predictions require you to collect: meal ID, station, batch size, time-to-ready, and per-meal service durations — then join them to queue snapshots.

### 4.5 Assumptions and limits blocking direct transfer

1. Magnitudes are off by ~10–20× (median 204 min vs. real canteen 5–15 min).
2. No canteen category, no students, no menu, no rush-hour structure.
3. Synthetic noise (SD ≈ 7) is tidier than real chaos (rushes, stock-outs, UPI outages).
4. `Time_of_Day` as bare hour ignores day-of-week, timetable gaps, and holidays.
5. One row = one independent snapshot; real queues are time series (a 13:00 row predicts 13:05).

### 4.6 What is actually in the data vs. what you must collect

- **In the data:** generic queue physics (N, C, S → wait) + a demo Streamlit pattern to copy.
- **You must collect:** canteen queue sizes, counter states, timestamps, meal IDs, per-meal timings, and measured waits — even 300–500 real rows will teach you more than these 5,000 synthetic ones.

---

## 5. Data leakage and modelling risks

### 5.1 Features that reveal the target

- `Avg_Service_Time` is the prime suspect. If it means *"the average we measured while serving exactly these people"*, it is only known **after** the wait is over — classic leakage. If it means *"historical average for this counter/hour"*, it is legitimate but must be computed strictly from past data, never including the current queue. **Rule for your app:** any service-time feature must be timestamped *before* the prediction moment; document its definition.
- The engineered `naive = N×S/C` is not a dataset column — we built it for diagnosis. Do not add it as a raw input and then claim a fair evaluation; it *is* the generator. It is fine as an engineered baseline feature, but report it transparently.

### 5.2 Features available only after waiting/serving

- None of `Number_of_People`, `Counters_Open`, `Time_of_Day`, `Queue_Type` is inherently post-hoc — all can be known at queue-entry. Only `Avg_Service_Time` needs the timestamp discipline above.

### 5.3 Duplicate / near-duplicate contamination

- 0 exact duplicates (good). 3 near-duplicate input pairs with targets differing by 3–25 min (listed in §2.2). A random split may strand twins across train/test, slightly inflating scores. Fix: drop or group these pairs (`groupby` inputs) before splitting. Impact here is negligible but the habit matters for real data (same lunch rush logged twice).

### 5.4 Was the target computed by formula? (evidence)

Yes, with high confidence but stated as inference: `corr(naive, target) = 0.9999`, per-type additive offsets (+10/+15/+20 ± 7), uniform inputs, and absent documentation jointly imply `target = N×S/C + offset[type] + noise`. This is **not** an independently measured wait. Consequence: any flexible regressor will memorise the formula and report near-perfect errors that evaporate on real data.

### 5.5 Random vs. time-based split

- **For this dataset:** rows carry no date/sequence ID, only hour-of-day which is signal-free and uniform. There is no temporal order to respect, so a **random split is appropriate** (stratify by `Queue_Type`/`Counters_Open` if you like).
- **For your canteen data:** use a **time-based split** (train on weeks 1–3, validate week 4, test week 5+) once timestamps exist, because queues are autocorrelated and menus rotate. Random shuffling of time series leaks the future.

---

## 6. Recommended machine-learning approach

> Guidance, not a committed algorithm choice. Per the brief, establish a baseline first.

### 6.1 Problem formulation

- **Supervised regression:** predict continuous `Waiting_Time` (minutes) from `[Number_of_People, Counters_Open, Avg_Service_Time, Time_of_Day, Queue_Type]` (+ engineered features below).
- Keep the raw target in minutes for interpretability. Optionally compare a `log1p(Waiting_Time)` variant during experiments since the tail is extreme — but always report errors back in minutes.
- This is **not** classification ("long/short") and not forecasting yet (no sequence modelling until you log timestamps).

### 6.2 Sensible baselines (run these before anything fancy)

1. **Mean / median predictor** (median ≈ 203.55 min): quantifies "do nothing" error.
2. **Naive physics formula** `N×S/C` (+ optional per-type constant): the domain baseline; any ML model must beat it to justify itself.
3. **Plain LinearRegression on raw features:** we measured R² ≈ 0.50, MAE ≈ 266 min in-sample — deliberately weak, demonstrating why interaction features are needed.

### 6.3 Candidate algorithms to try (2–3)

1. **Random Forest Regressor** (repo's choice) — handles the `1/Counters` non-linearity and categorical type with no scaling; robust baseline for tabular data.
2. **Gradient Boosting (HistGradientBoosting / XGBoost / LightGBM)** — usually beats RF on tabular regression once tuned; handles skewed targets well.
3. **Regularised linear model (Ridge) on engineered features** (`N×S/C`, `N/C`, `S/C`, `log` variants + one-hot type + cyclical hour) — interpretable, fast, and a sanity check that the physics, not the ensemble, does the work.

Do not start with deep learning: 5 features / 5,000 rows does not warrant it.

### 6.4 Preprocessing strategy

- **Numerical** (`Number_of_People`, `Counters_Open`, `Avg_Service_Time`): median imputation (no missing now, but keeps pipeline robust); scaling (`StandardScaler`) for linear models, unnecessary for trees. Add engineered `N×S/C` and optionally `N/C`.
- **Time** (`Time_of_Day`): do **not** feed 0–23 as a plain number (23:00 is adjacent to 00:00). Use cyclical encoding `sin(2πh/24), cos(2πh/24)`; keep raw hour for tree models if desired.
- **Categorical** (`Queue_Type`): one-hot encoding (3 levels, no ordinality). For your canteen version, replace with `station_id` / `meal_id` one-hots or target encoding once cardinality grows.
- **Target:** consider `log1p` training for the tail; always inverse-transform before scoring in minutes.
- Wrap everything in a single `sklearn` `Pipeline`/`ColumnTransformer` so preprocessing is fitted on train folds only.

### 6.5 Evaluation metrics (include MAE and RMSE)

- **MAE** (minutes): primary, student-facing ("on average we're off by X minutes"). Robust to the tail.
- **RMSE** (minutes): penalises huge misses (the `Counters_Open=1` disasters) — report alongside MAE; the RMSE−MAE gap diagnoses tail risk.
- **MedianAE** and **R²**: MedianAE for the typical experience; R² for variance explained (expect ≈0.999 here — flag it as synthetic, not success).
- Avoid MAPE as a headline (tiny waits blow it up). Slice errors by `Counters_Open`, `Queue_Type`, and wait decile — a model that is fine on average but +40 min wrong at lunch is a failed canteen model.
- Qualitative check: plot predicted vs. actual and residuals vs. `naive`; residuals should show no structure.

### 6.6 Validation strategy

- **Now (synthetic):** 70/15/15 train/validation/test random split (stratify by `Queue_Type`), or 5-fold CV; keep test locked until final reporting; fix `random_state` for reproducibility.
- **Later (real canteen):** time-based split + grouped CV by day; re-validate after each menu/shift change. Never tune hyperparameters on the test set.
- Report baseline → candidate deltas with confidence intervals (CV std), not single lucky seeds.

---

## 7. Recommended next steps (roadmap)

### 1. Data cleaning (30 min, on this data)
- [x] Verified: no nulls, no negatives, no type fixes needed.
- [ ] Treat `Time_of_Day` as hour/cyclical, `Queue_Type` as categorical (not strings compared alphabetically).
- [ ] Flag the 522 IQR "outliers" as structural tail — do **not** delete them; optionally add a `log` target experiment.
- [ ] Group/dedupe the 3 near-duplicate input pairs before splitting.

### 2. Feature selection & engineering
- [ ] Keep all 5 raw inputs for the prototype.
- [ ] Add `N×S/C` (+ `N/C`, `S/C`) and cyclical hour features; ablate to show their value.
- [ ] Drop/replace `Queue_Type` when moving to canteen data (new: `meal_id`, `station_id`, `day_of_week`, `is_peak`).

### 3. Baseline model development
- [ ] Implement median, naive-formula, and linear baselines inside a `Pipeline`; record MAE/RMSE/R² on a held-out test set.

### 4. Model comparison & evaluation
- [ ] Compare Random Forest vs. Gradient Boosting vs. Ridge-on-engineered-features with 5-fold CV.
- [ ] Error slices (by counters, type, wait decile) + predicted-vs-actual and residual plots.
- [ ] Write up why scores ≈0.999 here will not transfer — set expectations before demo day.

### 5. Additional canteen data collection (the real project)
- [ ] Log per serving event: timestamp, queue size at entry, counters open, meal ordered, service start/end, measured wait. Target ≥300–500 real rows across peak/off-peak days.
- [ ] Build a per-meal service-time table via stopwatch sampling (don't burden students).
- [ ] Re-run §3 EDA on real data; expect hour and meal to become top signals.

### 6. Web-app integration
- [ ] Reuse the repo's Streamlit pattern: inputs (meal dropdown, queue size, auto time, staff counter toggle) → `pipeline.predict` → "≈ X min" + confidence band (±MAE).
- [ ] Persist the fitted `Pipeline` (preprocessing + model) with `pickle`/`joblib`, version it, and log predictions vs. realised waits for monitoring.
- [ ] Add guardrails: clamp outputs ≥0, warn when inputs are out-of-training-range (e.g. queue > 300), and show "prototype model — real canteen calibration pending" until §5 data arrives.

---

## Reproducibility

- Original CSV untouched. Analysis script: `analyze_queue_dataset.py`; plots: `plots/01–14_*.png`.
- Key library versions at analysis time: Python 3.12, pandas 3.0.5, scikit-learn 1.9.0, matplotlib 3.11.2, seaborn (latest pip).
- Randomness: EDA uses no random sampling; ML guidance prescribes fixed `random_state` values when you build models.

## Limitations & uncertainties

- Units of `Avg_Service_Time`/`Waiting_Time` as minutes follow the README; no stronger proof exists in the data **[Uncertain]**.
- Synthetic-data conclusion is inference from strong statistical evidence + missing provenance, not access to the generator script **[Assumption, high confidence]**.
- Canteen transfer advice assumes a counter-style serving line with per-meal prep variation — validate against your actual canteen layout.

*Prepared for a first-time end-to-end ML team: build the pipeline on this data now, earn your evaluation habits here, then replace the data with real canteen observations before any production claim.*
