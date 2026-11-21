# Foundation models under drift, v2: experiment plan

Committed before any v2 model fit that produces a result, the pilot included. The only fits made before this commit are the default LightGBM runs of the population rules in section 5 (ceiling and floor), and the loader, gate and analysis code with its tests. v2 builds on [plan.md](plan.md) (v1), which stays unchanged together with every v1 file under `results/`. SDB is [SmallDataBenchmarks](https://github.com/firefly1248/SmallDataBenchmarks) at the commit pinned in `sdb.py`; its datasets, loader and model wrappers are read in place.

## 1. Question and purpose

v1 rejected none of its five tests, and its intervals do not exclude a change of about half a typical lead. v2 turns that into claims that can be checked, and adds the drift case v1 left out:

1. **Equivalence.** Does the TFM lead change under covariate drift by less than a stated margin, on a fresh and larger set of datasets? (family E)
2. **Extrapolation.** What happens to the TFM lead when test rows leave the range of the training window along one direction of the data, where trees predict a constant and TFM behavior is unknown? (family X)
3. **Fairness and cost.** Re-tuned boosters in the sliding window, and the measured cost of a refresh for each model. (secondary, and Part 0)
4. **Window size.** Do the lead and its response to drift change when the training window grows from 1,000 to 20,000 rows? (Study C, family C)
5. **Fresh real data.** Study A2 is described in section 5.4 and is dropped unless a stated dry run names qualifying datasets.

Every part stays in the plan. A budget reduction (section 10) only narrows a part.

## 2. Ground rules

- Outputs go to `results/v2/` (tables) and `results/v2/figures/`; checkpoints go to `results/runs_v2/` (git-ignored) and are released later, packed like v1.
- v1 code (`drift_windows.py`, `drift_run.py`, `drift_analysis.py`) is not changed. v2 adds modules (`drift_windows_v2.py`, `drift_data_v2.py`, `drift_population.py`, `drift_analysis_v2.py`) and extends the runner by new options only; a regression test pins v1's scenario list and the sampled windows of every v1 scenario but the tilt, and compares v1's gate for one dataset with the committed gate.
- Same models, versions, wrappers, tuning spaces, metrics, clipping and seed rules as v1 (plan.md 4.4 to 4.6), except where this plan says otherwise (the seed rule of section 6.5, the tree limit of Study C in section 8).
- The v1 datasets (the ten Study B datasets, ringnorm, skin-segmentation, twonorm, the seven public Study A datasets), versions of them and copies of them are not in any v2 confirmatory test. They may appear in sensitivity tables, labeled as such.
- Nothing derived from data whose owner's terms bar it is published (section 5.3).
- Text of the plan, README and docstrings: plain English, US spelling, no em-dash.

## 3. Order of work

1. Part 0 (section 4): the cost of a refresh from v1's checkpoints. Done, see `results/v2/refresh_cost.csv`.
2. The loader, the gate code and its tests, the power simulation.
3. The population (section 5): candidates, rules, ceiling and floor, gate, licenses.
4. This plan, committed with the code, the tables and the margins.
5. The pilot (section 10): timings only, no scores.
6. The budget and the reductions it needs, committed as an amendment before the queue starts.
7. The queue; then the analysis of section 7, after the checks of section 9 pass.

## 4. Part 0: the cost of a refresh

From the `times` of v1's checkpoints (Study B and the public Study A datasets, control scenario, T2 to T4): the median wall time of a refresh, reported by model and device (TabFM ran on MPS, the rest on CPU, so costs compare within a device only), in two parts that are not summed across model types: the fit (for a booster, plus the re-tuning under S2) per 1,000 window rows, and the prediction per 1,000 block rows. A TFM fits lazily, so its cost is in the prediction. Boosters ran on 3 threads and tuned on 4 folds in parallel; TFMs used their own threading. Exploratory.

Medians, `results/v2/refresh_cost.csv` (216 refreshes per model and strategy):

| model | device | strategy | fit s / 1,000 window rows | predict s / 1,000 block rows | median re-tune s |
|---|---|---|---|---|---|
| catboost | cpu | S1 | 0.25 | 0.003 | 0.0 |
| catboost | cpu | S2 | 14.463 | 0.003 | 26.362 |
| lgbm | cpu | S1 | 0.18 | 0.006 | 0.0 |
| lgbm | cpu | S2 | 16.781 | 0.008 | 24.959 |
| logreg | cpu | S1 | 0.01 | 0.003 | 0.0 |
| logreg | cpu | S2 | 0.633 | 0.003 | 1.775 |
| tabfm | mps | S1 | 0.07 | 15.649 | 0.0 |
| tabfm | mps | S2 | 0.023 | 26.293 | 0.0 |
| tabicl | cpu | S1 | 0.155 | 2.897 | 0.0 |
| tabicl | cpu | S2 | 0.069 | 6.823 | 0.0 |
| tabpfn35 | cpu | S1 | 0.558 | 5.395 | 0.0 |
| tabpfn35 | cpu | S2 | 0.279 | 10.262 | 0.0 |
| xgboost | cpu | S1 | 0.155 | 0.004 | 0.0 |
| xgboost | cpu | S2 | 9.885 | 0.005 | 14.217 |


## 5. Populations

### 5.1 Loader

`drift_data_v2.py` gives OpenML frames SDB's conventions: a column that parses as a number more than 90 % of the time is numeric, the rest are strings, label 1 is the minority class, all-missing columns are dropped. Rows that repeat features and label are dropped first, in every study. A file above the cap is cut to it by a uniform seeded draw (seed 0; Study C draws the cap class by class). SDB datasets keep SDB's own 10,000-row cap, applied before deduplication. `openml` 0.15.1 is pinned in `pyproject.toml`; dataset ids and versions are in `results/v2/population.csv`.

### 5.2 Rules, applied in this order

1. The candidate is binary (SDB files are checked by loading).
2. Not a v1 dataset, a version of one, or a copy of an earlier candidate (same name, or the same rows, columns and class counts after deduplication). When a dataset is in several sources the order of precedence is TabArena, OpenML-CC18, SDB, then the reserve suites.
3. At least 1,000 rows after deduplication and the cap, at most 500 features, minority share at least 0.05.
4. Not at the ceiling or the floor: the 5-fold stratified (seed 0) ROC AUC of default LightGBM 4.6.0 on the file the study runs (the capped, deduplicated file) is at least 0.60 and below 0.99. A dataset below 0.65 is marked low-signal and left out of the sensitivity table named in section 7.1.
5. The gate of section 6.4: the control and tilt gate (the E population) and the extrapolation gate (the X population) are separate, and a dataset enters each family on its own gate.
6. The owner's terms allow publishing derived numbers (section 5.3).

Sources: SDB's binary datasets; the OpenML-CC18 suite (id 99); the TabArena-v0.1 suite (id 457). Reserve sources, admitted only to bring the E population to 30, in the order suite 271 (AutoML benchmark), suite 337, suite 334 (Grinsztajn), and within a suite by dataset name: used only because the first three sources leave fewer than the 25 datasets the E family needs. Candidates that pass all rules are the population; there is no sampling.

Study B2 candidates through the rules, first three sources (SDB, OpenML-CC18, TabArena):

| rule | rejected | left |
|---|---|---|
| Candidates |  | 268 |
| Binary | 133 | 135 |
| Not a v1 dataset or a version of one | 28 | 107 |
| Not a copy of an earlier candidate | 14 | 93 |
| Rows, features, minority share | 54 | 39 |
| LightGBM ROC AUC in [0.60, 0.99) | 7 | 32 |
| File could be read | 0 | 32 |
| Gate E (control and tilt) | 6 | 26 |
| Owner's terms not restricted, E | 3 | 23 |
| Gate X (extrapolation arms), among the E-gate datasets | 6 | 17 |

Reserve suites (271, 337, 334):

| rule | rejected | left |
|---|---|---|
| Candidates |  | 94 |
| Binary | 30 | 64 |
| Not a v1 dataset or a version of one | 9 | 55 |
| Not a copy of an earlier candidate | 25 | 30 |
| Rows, features, minority share | 7 | 23 |
| LightGBM ROC AUC in [0.60, 0.99) | 3 | 20 |
| File could be read | 1 | 19 |
| Gate E (control and tilt) | 0 | 19 |
| Owner's terms not restricted, E | 3 | 16 |
| Gate X (extrapolation arms), among the E-gate datasets | 1 | 15 |

The E population is the 23 datasets of the first three sources that pass the E gate and are not restricted, plus the first 7 reserve datasets in the order of section 5.2 (Higgs, MiniBooNE, ada, airlines, albert, jasmine, kick), 30 in all. The X population is the 23 of them that also pass the X gate. `results/v2/population.csv` lists every candidate with the rule that rejected it; `results/v2/population_b2_final.csv` the admitted 30:

| name | source | rows | seeds | terms | population |
|---|---|---|---|---|---|
| Bank_Customer_Churn | TabArena-v0.1 | 10000.0 | 5.0 | unresolved | E+X |
| E-CommereShippingData | TabArena-v0.1 | 10000.0 | 5.0 | unresolved | E+X |
| Fitness_Club | TabArena-v0.1 | 1499.0 | 10.0 | unresolved | E+X |
| HR_Analytics_Job_Change_of_Data_Scientists | TabArena-v0.1 | 10000.0 | 5.0 | open | E+X |
| Marketing_Campaign | TabArena-v0.1 | 2058.0 | 5.0 | unresolved | E |
| NATICUSdroid | TabArena-v0.1 | 7491.0 | 5.0 | open | E+X |
| churn | TabArena-v0.1 | 5000.0 | 5.0 | unresolved | E+X |
| coil2000_insurance_policies | TabArena-v0.1 | 8380.0 | 5.0 | open | E+X |
| credit_card_clients_default | TabArena-v0.1 | 10000.0 | 5.0 | open | E+X |
| customer_satisfaction_in_airline | TabArena-v0.1 | 10000.0 | 5.0 | open | E+X |
| hazelnut-spread-contaminant-detection | TabArena-v0.1 | 2400.0 | 5.0 | open | E+X |
| in_vehicle_coupon_recommendation | TabArena-v0.1 | 10000.0 | 5.0 | open | E+X |
| jm1 | TabArena-v0.1 | 8912.0 | 5.0 | unresolved | E+X |
| online_shoppers_intention | TabArena-v0.1 | 10000.0 | 5.0 | open | E+X |
| polish_companies_bankruptcy | TabArena-v0.1 | 5850.0 | 5.0 | open | E |
| adult | OpenML-CC18 | 10000.0 | 5.0 | open | E+X |
| kc1 | OpenML-CC18 | 1212.0 | 10.0 | unresolved | E |
| madelon | OpenML-CC18 | 2600.0 | 5.0 | open | E+X |
| wilt | OpenML-CC18 | 4819.0 | 5.0 | open | E |
| child-obesity-beijing | sdb | 6486.0 | 5.0 | open | E+X |
| cholera-tz-result | sdb | 2927.0 | 5.0 | open | E |
| drrisk-retinopathy | sdb | 10000.0 | 5.0 | open | E+X |
| equine-sinonasal-primary | sdb | 1021.0 | 10.0 | open | E |
| Higgs | AutoML benchmark | 10000.0 | 5.0 | open | E+X |
| MiniBooNE | AutoML benchmark | 10000.0 | 5.0 | open | E+X |
| ada | AutoML benchmark | 4146.0 | 5.0 | unresolved | E |
| airlines | AutoML benchmark | 10000.0 | 5.0 | open | E+X |
| albert | AutoML benchmark | 10000.0 | 5.0 | unresolved | E+X |
| jasmine | AutoML benchmark | 2984.0 | 5.0 | unresolved | E+X |
| kick | AutoML benchmark | 10000.0 | 5.0 | unresolved | E+X |


### 5.3 Licenses

OpenML's license field is not the owner's statement ("Public" is its default). For every admitted dataset the owner's terms were read and are recorded in `results/v2/licenses.csv` with the page read. A dataset whose terms limit use to a competition, to non-commercial use or to a named purpose, require consent, or bar redistribution of derived work, leaves the confirmatory populations, and so does one whose terms could not be found. Competition data under competition rules and data under written-consent terms are examples (v1 met the second kind in himalaya and f1-dnf). The v2 checkpoint release holds per-block row indices and labels, so it follows the same rule.

Of the 47 datasets that passed the gate in Study B2 or Study C, 24 have open terms, 6 are restricted and 17 are unresolved (no owner's terms found). Restricted: Amazon_employee_access, GiveMeSomeCredit, heloc, Click_prediction_small, credit, house_16H. The restricted datasets leave every population. Unresolved datasets stay (a missing license is not a bar, and dropping them would leave the E family with 19 datasets, fewer than the 25 it needs), are marked in the tables, are reported with and without in the sensitivity of section 7.1, and have their row indices and labels left out of the checkpoint release. 11 of the 30 E-population datasets are unresolved.

### 5.4 Study A2

The four SDB external datasets named as candidates before the build do not qualify. `malpdf-fs43-binary`: `creationDate` takes the values 0 and 1, it is a flag and not a time. `dengue-mexico-deaths`: 577 deaths in 91,318 rows, so a 2,000-row block holds 5 to 18 of them, under the floor of 30. `cholera-tz-result`: two of five time-ordered blocks of 590 rows hold fewer than 30 minority rows. `biotech-y36`: the label is a 36-month outcome, so the share of window labels not yet known when a block starts is 1. SDB's `early_stopping_time_plan.md` excludes the same three for the same reasons. BeyondArena's temporal tasks are older public data that TFM authors report on, so they are not fresh. Study A2 therefore has no datasets and is dropped. It returns only through an amendment, before any v2 run, that names at least three datasets passing SDB's ordering, tie, leak and label-horizon rules, a block size of floor(n / 5) (at most 2,000, at least 300 rows, at least 30 of the smaller class per block) and v1's fold minima. With eight or more datasets a sign test of the TFM lead at T4 under S1 would be added in that amendment; with fewer it is descriptive.

## 6. Study B2

### 6.1 Geometry

As v1 (plan.md 4.2): five class-stratified pools P1 to P5 of one seed, B1 = P1, block T_h drawn from P_(h+1), dose fraction h/4. The control and the tilt arm are v1's control and covariate arm, unchanged (tilt weight exp(d_h z^2), target mean z^2 of 1 + 0.5 h/4), with z the first-principal-component score of the whole file, as in v1.

### 6.2 Extrapolation arms

z for these arms is v1's score with ties in PC1 broken by a fixed jitter (a file with few distinct values, such as one with mostly categorical columns, would otherwise have ties at the quantile) and with the sign set so that z is higher where label 1 is more frequent. This sign convention fixes what "up" means on every dataset.

- **xup**: B1x holds the rows of P1 with z at or below the 0.6 quantile of z over the whole file. T_h holds m_x rows: a share h/4 (rounded) drawn uniformly from the rows of P_(h+1) with z above the 0.6 quantile, the rest uniformly from its rows at or below it. T4 lies entirely above the quantile.
- **xdown**: the mirror image, B1x with z at or above the 0.4 quantile and T_h drawn beyond it from below.
- m_x is 0.8 of the fewest rows beyond a quantile in any of P2 to P5, in either direction, so it is set per dataset and seed from the realized counts and is the same in both arms.
- **U**, the reference: a seeded uniform subsample of P1 as large as the smaller B1x, fitted once per seed and used for both directions. U trains on the full range of z at the size of B1x.
- Windows: S1 before T_h is the whole of T_(h-1) plus a seeded random part of T_(h-2) (for h = 2, of T0, a seeded uniform half of B1x), the part sized so that the window has |B1x| rows when T_(h-2) has enough. S2 is B1x and T1 to T_(h-1). Boosters are tuned on B1x, on U, and for S2 on each S2 window.
- Recorded per block: the share of rows beyond B1x along z, and the share with at least one numeric feature outside B1x's per-feature range (for a categorical column, an unseen level counts as outside; a file without numeric columns records none).

There are no truncated control arms. The X contrast holds the test rows fixed instead: both fits score the same xup (or xdown) rows, so the region, the class balance and the tilt of those rows cancel, and what differs is that one fit has never seen the region.

### 6.3 Strategies and models

S0, S1 and S2 as in v1, on the arms control, tilt, xup and xdown. S1r, for the boosters only: a full 50-trial re-tune (stratified 4-fold, as v1) on every S1 window of the four arms. The oracle and the concept arms are not run in v2. Logistic regression is a baseline outside the families. TFM family means are over TabFM, TabPFN-3.5 and TabICLv2; booster family means over CatBoost, LightGBM and XGBoost; per-dataset values are seed means.

### 6.4 Gate

Run before this commit, with no model. A dataset enters the E population if v1's gate holds for the control and the tilt on its seeds (at least 15 rows of the smaller class in every block and window, 30 in B1, the tilt's T4 mean z^2 within 0.1 of 1.5); the X population if, on every seed, m_x is at least 1, every block holds at least 15 rows of its smaller class, every S1 and S2 window at least 15, and B1x and U at least 30. A dataset failing one gate may still enter the other family.

### 6.5 Seeds

10 seeds for a dataset of fewer than 2,000 rows (v1's rule), 5 for every other (v1 used 3: v1's seed-to-seed standard error of a dataset's H1 value is as large as the margin, and about half of the variance the equivalence test faces).

## 7. Hypotheses

Notation as in v1: lead = TFM family minus booster family; T4 is the full-dose block; the unit of analysis is the dataset.

### 7.1 Equivalence family E (Study B2, tilt arm)

| test | per-dataset quantity | margin |
|---|---|---|
| E1 | lead in ROC AUC at T4 under S0, tilt minus control | 0.005 |
| E2 | lead in the refit gain (S1 - S0) in ROC AUC at T4, tilt minus control | 0.005 |
| E3 | lead in log loss at T4 under S0, tilt minus control | 0.009 |

The margins were set from v1's control at T4 under S0 on the nine Study B datasets without hill-valley-with-noise (`results/drift_leads.csv`): median lead 0.0142 in ROC AUC and 0.0274 in log loss, each margin about a third of it. They are absolute, they do not change after this commit, and E2's margin is borrowed from E1: it is larger than the median refit gain of either family in v1's covariate arm (0.0004 for TFMs, 0.0035 for boosters), so E2 can hold even if a refit helps one family only, and it is worded as weak evidence.

Test: two one-sided exact Wilcoxon signed-rank tests of the per-dataset values shifted by +margin and by -margin; the TOST p-value is the larger. Holm over E1 to E3 on the TOST p-values; the verdict is the Holm-adjusted p below 0.05. The reported interval is the Hodges-Lehmann interval at the level the adjusted test uses (1 - 2 alpha divided by the test's step in the Holm order). A test needs at least 25 datasets; below that it is reported as underpowered, with its interval, and claims nothing. Zeros are dropped and tied absolute differences stop the analysis (checked on the shifted values).

Power, by simulation (`results/v2/power.csv`, normal per-dataset values, exact TOST, 4,000 replicates per cell; standard deviations from v1 over all ten datasets: 0.0072, 0.0068, 0.0089 for E1 to E3; the nine datasets without hill-valley give 0.0041, 0.0041, 0.0066): at 30 datasets and a true change of 0 the power of E1 is 0.95 unadjusted and 0.87 at the first Holm step; at a true change of 0.002 it is 0.70 and 0.50; at 0.003, 0.43 and 0.24. v1's own estimate for H1 was -0.0016 (median) to -0.003 (mean), so E1 and E2 are likely inconclusive if the true change is that size, at 30 or 40 datasets alike.

Wording of a result: "the typical (Hodges-Lehmann) change of the lead across datasets is within 0.005 in ROC AUC" for an equivalence, "no equivalence shown at this margin" otherwise. The signed-rank test is about the pseudo-median under symmetry, not about the mean or any single dataset (in v1, 3 of 10 datasets lie outside the margin on H1). Reported next to every test, no verdict: the number of datasets whose own change lies outside the margin; the mean change with a 95 % bootstrap interval (10,000 resamples); each family's own change in the metric from control to tilt with its Hodges-Lehmann interval (the strength of the drift: an equivalence is read as robustness to covariate drift only if at least one family moves by more than the margin, otherwise as "the drift was too weak to test equivalence"); the median control lead of the v2 population and each margin as a share of it; v1's companion, the log error ratio, with the same TOST at a margin set from the v2 control; E1 without the TabArena datasets, without the reserve datasets and without the low-signal datasets.

### 7.2 Extrapolation family X (Study B2, xup and xdown)

Per dataset, direction d and seed: the lead at T4 under S0 of the fit on B1x_d, minus the lead at T4 under S0 of the fit on U, both scored on the same T4 rows of xd. X1 is in ROC AUC, X2 in log loss. The per-dataset value is the mean of the two directions. Two-sided exact Wilcoxon, Hodges-Lehmann estimate and 95 % interval, Holm over X1 and X2; a test needs at least 20 datasets (the X population is smaller than E's, section 5.2), otherwise it is underpowered. No expected sign. Reported, no verdict: each direction separately; X at T1 to T3 (dose); X2 next to its intercept-recalibrated version (v1's S0*, intercept fitted on the S1 window before T4); X split at the median over datasets of the per-feature extrapolation share (the arm's share minus the control's share at T4); the S1 and S1r quantities: lead in (S1 - S0) at T4 on the xup and xdown blocks, with S1r in place of S1 (a descriptive extrapolation refit contrast, not a companion of X1).

### 7.3 Window-size family C (Study C, section 8)

The TFM family in family C is TabPFN-3.5 and TabICLv2 (the models that see the whole window at every size, section 8); the booster family is as before. All quantities are at T4 under S0.

| test | per-dataset quantity |
|---|---|
| C0 | the control lead in ROC AUC at window 20,000 (one-sided: is the lead above zero?) |
| C3 | the control lead in ROC AUC, window 20,000 minus window 1,000 (two-sided: did the lead change with size?) |

Exact Wilcoxon signed-rank tests, Holm over C0 and C3; a test needs at least 8 datasets (as v1), otherwise Study C is descriptive (per-dataset values and the interval of the Hodges-Lehmann estimate, no verdict). Estimates with 95 % intervals and no verdict: C1, the tilt effect on the lead (tilt minus control) at 20,000 minus at 1,000; C2, X1 at 20,000 minus at 1,000; each of the two quantities at 5,000; log loss versions; the three-model TFM family; per-model curves of the lead against window size.

### 7.4 Error control

Error is controlled within a family only: Holm inside E, inside X and inside C. Across the three families, the chance of at least one confirmatory rejection under a global null is about 14 % (1 - 0.95^3). Everything reported with no verdict carries estimates and intervals and no p-value.

### 7.5 Secondary, no verdict

S1r in every arm (section 6.3); refresh cost measured in the v2 runs as in Part 0, per window size in Study C; per-model results; dose slopes over T1 to T4; S2; skill, ECE, Brier; leave-one-out lines; the pooled v1 + v2 sample.

## 8. Study C: window size

Question: whether the lead and its response to drift change when the training window grows. v1 and Study B2 windows are at most about 2,600 rows, the regime where TFMs are strongest.

**Population.** Binary datasets from the sources of section 5.2 including the reserve suites, with at least 36,000 rows after deduplication, at most 500 features, minority share at least 0.05, not a v1 dataset or a version or copy of one, LightGBM ROC AUC in [0.60, 0.99) on the 36,000-row file the study runs, the gate below, and open terms. A dataset may be in both Study B2 (capped at 10,000 rows) and Study C (36,000 rows): the families are tested separately and the geometries differ. Result of the build (`results/v2/population_c.csv`): of the 21 large binary candidates read, 9 pass every rule and have terms that are not restricted (Diabetes130US (open), adult (open), Higgs (open), MiniBooNE (open), airlines (open), albert (unresolved), kick (unresolved), sf-police-incidents (open), road-safety (open)); 9 of them also pass the extrapolation gate. That is below the 10 to 20 datasets first hoped for, so family C is tested with the 9 and its power is low; at least 8 datasets are needed to run a test (the minimum of v1's tests).

**Geometry per dataset and seed.** The file is cut to exactly 36,000 rows (class-stratified, seeded); z (with the one-hot width capped, levels beyond the 20 most frequent in a categorical column pooled) is fitted on those rows. P1 holds 20,000 rows and P2 to P5 4,000 rows each, class-stratified and disjoint. Test blocks T1 to T4 are 2,000 rows drawn from P2 to P5 as in v1; the extrapolation blocks use the m_x of section 6.2. The window of size w (1,000, 5,000, 20,000) is a class-stratified subset of P1, nested (1,000 inside 5,000 inside P1), so sizes differ only in rows added. All windows of a dataset and seed face the same blocks. For the extrapolation arms B1x(w) is the window's rows on the allowed side of the quantile (about 0.6 w rows) and U(w) a seeded uniform subset of the window of the same size, nested across w. Arms: control, tilt, xup, xdown. Strategy: S0 only. S1 is not run in Study C (at w = 20,000 the window is P1, nothing is unused, and at every size the staleness of a sliding window would change with w); S2 is not run.

Gate per dataset, 3 seeds: the control and tilt gate of section 6.4 on the pools of this geometry, and for the extrapolation arms m_x at least 1, blocks with at least 15 of the smaller class, windows, B1x(w) and U(w) with at least 30, at every w.

**Models.** The same seven. Context limits at the pinned versions, from the code: TabPFN-3.5 (tabpfn 9.0.0, checkpoint `tabpfn-v3.5-20260909.safetensors`) 1,000,000 rows (SDB's wrapper passes `ignore_pretraining_limits=True`, which lifts the 5,000-row CPU guard); TabICLv2 (tabicl 2.1.1) no row limit in the classifier; TabFM (tabfm 1.0.1) none in the library, and SDB's wrapper sets a context of 5,000 rows, a memory cap for this 24 GB machine (a 10,000-row context took 13.55 GB), drawn per ensemble member by the library, not class-stratified. TabFM therefore sees at most 5,000 rows at w = 20,000, so the primary C quantities use the two TFMs that see the whole window; the three-model family is secondary. Whether TabPFN-3.5 and TabICLv2 complete a 20,000-row context on CPU in memory and time is learned in the pilot; if one does not, the amendment of section 3 step 6 states the largest context it is given before any result is read.

**Boosters** are tuned on each window size with v1's stratified 4-fold scheme and 50 trials, with the upper bound of `n_estimators` raised from 500 to 2,000 (an exception to v1's spaces, so that a capped booster does not understate what size gives it). Logistic regression is a baseline.

## 9. Checks

v1's checks 2 to 7 on v2 outputs, plus:

- the realized share of block rows beyond the quantile at T4 of xup (below it for xdown) is 1, and the shares at T1 to T3 equal round(m_x h / 4) / m_x;
- leakage: no row is in two blocks of a scenario, no S1 or S2 window holds a row of the block it precedes, B1x and U are never tested, in Study C a window never holds a test-block row;
- v1's regression test (section 2);
- control diagnostic, as v1's check 6: under each control the 99 % Hodges-Lehmann interval over datasets of the lead's change from T1 to T4 under S0, per outcome, in Study B2 and per window size in Study C. A chance trigger sends the code to review; it does not stop the study.

The analysis stops after the checks unless all pass, with `--checks-reviewed` as in v1.

## 10. Compute

v1 took 21.2 h for 588 runs in 84 dataset-seed pairs, about 15 minutes per pair, but that average includes Study A and the concept-only datasets; v1's Study B pairs averaged 18.4 minutes and the 10,000-row datasets about 39. v2 Study B2 runs 4 scenarios (control, tilt, xup, xdown) with 25 tunings per booster and pair (v1: 4): B1 (1), B1x up and down (2), U (1), the S2 windows before T2 to T4 for the control, xup and xdown (9), and S1r on the three S1 windows of each of four arms (12). Taking each pair at about twice v1's minutes for a file of similar size (v1: 5 to 10 minutes for files under 2,000 rows, 15 for banana, 33 to 45 for the 10,000-row files), a pair costs about 15, 25, 44 and 70 minutes for files of under 2,000, 2,000 to 4,000, 4,000 to 9,000 and at least 9,000 rows. 14 of the 30 admitted datasets are at the 10,000-row cap. The seed rule of section 6.5 gives 165 dataset-seed pairs, so Study B2 is on the order of 130 machine hours. Study C has three window sizes, boosters tuned on up to 20,000 rows with up to 2,000 trees and TFM contexts of up to 20,000 rows on CPU; it is expected to be the larger part and its cost is not estimated here. All of this is arithmetic, not measurement.

Pilot: three Study B2 datasets (chosen to include one at the 10,000-row cap) and two Study C datasets, one seed, all models and arms, timings only, no scores kept. The budget is set after it. If it is short, reduce in this order, narrowing a part and never removing one: (1) S1r (kept on control, tilt and xup); (2) the third and later seeds of datasets below 2,000 rows; (3) S2 tuning (B1's parameters reused); (4) Study C seeds, from 3 to 2; (5) Study C's xup and xdown arms together (C2 becomes descriptive); (6) the E population's reserve datasets. Study C's 1,000, 5,000 and 20,000 windows and its control and tilt arms are not cut.

## 11. Deliverables

- This plan, the code with tests (new scenarios and gate in `drift_windows_v2.py`, the loader, the population and power scripts, the v2 analysis with the TOST, the X and C families and Part 0, and later the runner options for S1r, the extrapolation arms and Study C, and a v2 queue).
- `results/v2/*.csv`, `results/v2/figures/*.png`, a README section "v2" with the findings per outcome, nulls and failed equivalences included.
- A v2 checkpoint release, packed like v1 (no restricted data), and its manifest.

## 12. Limits

- Equivalence holds only at the stated margins, for the tilt arm's strength of drift (a reweighting inside B1's support that at most doubles the density of any region), and for the typical dataset; it is worded and tested as the pseudo-median, and a pass supports robustness only where a family's own change exceeds the margin.
- E2's margin exceeds the whole refit gain of either family in v1, so E2 is weak evidence.
- The extrapolation arm leaves the range along one component (a shift of about 1.6 standard deviations along z), not along every feature. The per-feature share outside B1x's range runs from about 0 on mostly categorical files to about 1 on files with many numeric features, so on some datasets no tree sees a feature outside its training range; X is reported against that share, not controlled for it. Its X2 includes a prior shift in training (B1x's class mix differs from U's); X2 is reported next to the recalibrated version.
- Study C reaches 20,000-row windows; larger tables and other model limits are not covered. TabFM is capped at 5,000 rows by the memory of this machine, which is a statement about this setup and not about the model.
- OpenML, TabArena and SDB datasets are public, so pretraining exposure cannot be excluded, and TabArena is also where TFM authors report results and choose checkpoints and defaults (selection exposure). The E1 sensitivity without the TabArena datasets addresses that without removing it.
- The datasets of the NASA defect family (jm1, kc1, pc3, pc4 where they appear) and of a shared origin are not independent designs.
- Boosters are tuned on log loss and scored by ROC AUC in E1, E2 and X1; TFMs are not tuned.
- TabFM runs on MPS and the rest on CPU, so refresh costs compare within a device only, and TabFM has no same-device comparator.
- The signed-rank tests assume symmetric per-dataset values around the center they test.

## 13. Decisions

Fixed by the project owner on 21 Nov 2026, before this plan: equivalence margins of 0.005 in ROC AUC and 0.009 in log loss; Study C window sizes of 1,000, 5,000 and 20,000 rows; OpenML-CC18 and TabArena as the sources; all strengthening points in the plan, with budget cuts only narrowing a part.

Fixed in this plan by default and to be confirmed or changed before the pilot starts:

1. **Licenses.** Restricted datasets leave every population; unresolved ones stay, marked, with their labels out of the release, and are reported with and without. Leaving unresolved datasets out instead would give the E family 22 datasets (underpowered) and Study C 7.
2. **Sources.** The reserve suites 271, 337 and 334 widen the owner's choice of OpenML-CC18 and TabArena; without them the E family has 23 datasets after the licenses, under its minimum of 25.
3. **Study A2** is dropped (section 5.4).
4. **E2** keeps the margin of 0.005 as weak evidence (section 7.1); the alternative is to report E2 with no verdict.
5. **Seeds**: 5 for datasets of 2,000 rows or more (v1: 3); this is two-thirds more runs of each large dataset than 3 seeds, see section 10.
6. **Study C**: family C needs 8 datasets, its TFM family is TabPFN-3.5 and TabICLv2, S1 is not run, `n_estimators` up to 2,000.
7. **Extrapolation reference**: the fit on U replaces truncated control arms (section 6.2), which also removes two of six arms and their cost.
8. **Floor**: ROC AUC of at least 0.60 for LightGBM on a dataset, with a sensitivity table without the low-signal ones.
9. **Machine**: another project's job occupies the machine for several hours; the pilot starts after it.
