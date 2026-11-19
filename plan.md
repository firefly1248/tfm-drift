# Foundation models under drift: experiment plan

Committed before any model run, the pilot included. The benchmark is [SmallDataBenchmarks](https://github.com/firefly1248/SmallDataBenchmarks) at commit d8c2e7b (SDB below); this project reads its loader, datasets, checkpoints and prepared files in place through `sdb.py`, and every path marked SDB is a path there. The injection gate (check 1) fits no model and ran first: `scripts/drift_windows.py`, committed with this plan, writes `results/drift_gate.csv`, and its result is in 4.2. Anything this plan does not specify is exploratory.

## 1. Question

When the test period drifts away from the training period, do tabular foundation models (TFMs) keep their lead over tuned gradient boosters, and does a fresher context help a TFM as much as a refit helps a booster?

The benchmark scores random splits, which hide drift whatever the data. Its headline, TFMs ahead of boosters, may not carry over to data that drifts.

## 2. Prior work

- [TabReD](https://arxiv.org/abs/2406.19380): time-based splits change the ranking of methods against random splits; MLP-like models and GBDT do best. It predates every TFM used here.
- [TableShift](https://arxiv.org/abs/2312.07577): 15 binary tasks with domain shifts; in-distribution and out-of-distribution accuracy follow a linear trend, and the shift gap is tied to label shift.
- [WhyShift](https://arxiv.org/abs/2307.05284): Y|X shifts dominate real tabular shifts and are region-specific, which supports a regional concept arm.
- [Beyond IID / BeyondArena](https://arxiv.org/abs/2606.30410) (NeurIPS 2026 Datasets and Benchmarks, per its arXiv comment): 142 datasets (103 IID, 21 temporal, 18 grouped), 11 models, three of them TFMs. TFMs lead on tiny and small IID data and are competitive up to 100k rows; tuned RealMLP leads the temporal, grouped, medium and large subsets. Once size is controlled, the temporal penalty is not significant and the grouped one is. Its temporal tasks train on all earlier data at each split point, with no sliding window and no contrast of a static model with a refit. It is the closest prior work to Study A and shares hotel bookings, bank marketing, german credit and qsar-biodegradation with this plan.
- [TabPFN-3.5 technical report](https://arxiv.org/abs/2609.17895): first on BeyondArena overall; tuned MLPs keep the lead on grouped, temporal and large data, but on tables up to 100k rows TabPFN-3.5 matches the best baselines on temporal data. No drift or refit experiments.
- [Out-of-distribution performance of TFMs](https://arxiv.org/abs/2607.26000): nine TFMs on three TableShift datasets (TabFM and LimiX skip the largest); they degrade on average, with ROC AUC gaps of 0.003 to 0.060, and the in-distribution/out-of-distribution relation of traditional models carries over, read from scatter plots. No boosters, no temporal shift.
- [TabPFN v2 in open environments](https://arxiv.org/abs/2505.16226): limitations in open environments, tree models the better general choice, but TabPFN v2 is adequate on small, covariate-shifted, class-balanced tasks.
- [Drift-Resilient TabPFN](https://arxiv.org/abs/2411.10634): a prior with temporal drift, 18 synthetic and real datasets, gains over TabPFN, XGBoost and CatBoost; built on the 2024 TabPFN line.
- [In-context learning of evolving data streams](https://arxiv.org/abs/2502.16840): the first TabPFN with a context of 600 to 1,000 rows (recent rows plus a class-balanced long-term buffer) against grid-tuned online Hoeffding-tree ensembles on seven real streams, by prequential accuracy; the fresher-context idea of H2, H3 and H5, against online trees rather than refit gradient boosters. Its Electricity stream is Study A's elec, where the paper reads a high no-change accuracy as strong dependence between neighbouring rows, so a fresher context there may gain from autocorrelation rather than from drift.
- [Cai and Ye](https://arxiv.org/abs/2502.20260): under temporal shift a random validation split selects models better than a temporal one.

What is new here: injected drift with a dose-response by drift type, and a fresher context against a refit at matched window sizes, for current TFMs against tuned boosters. The closest evidence on real temporal data already finds that the TFM gap there is a size effect and is absent below 100k rows, so the controlled drift and the refit contrast are the contribution, not another real-data ranking.

## 3. Hypotheses

Every primary test is an exact two-sided Wilcoxon signed-rank test on per-dataset values. "TFM" and "booster" are family means: the mean of TabFM, TabPFN-3.5 and TabICLv2, and the mean of CatBoost, LightGBM and XGBoost. The lead is the TFM mean minus the booster mean. Blocks T1 to T4, the strategies S0 and S1, the oracle and the region arm are defined in 4.2 and 4.3. Covariate and prior effects are read against the no-drift control of the same dataset and seed (difference in differences); the concept effect is read as a recovered share (section 5). No hypothesis has an expected sign.

- H1. Under covariate drift the TFM lead in ROC AUC at T4, static strategy, differs from the lead under the control. The null is an unchanged lead.
- H2. Under covariate drift a refit (S1 minus S0 at T4, in ROC AUC) helps TFMs by a different amount than boosters. The interval is reported and no non-inferiority claim is made.
- H3. Under prior drift the change in log loss from a refit, measured against the static model with its intercept re-estimated on the refit's window, differs between TFMs and boosters.
- H4. Under covariate drift the TFM lead in log loss at T4, static strategy, differs from the lead under the control.
- H5. Under concept drift (the region arm) the share of its own oracle gain that a sliding refit recovers at T4, in ROC AUC, differs between TFMs and boosters.

What is not tested, and why.

- The static model under prior or concept drift. Its score there is a fixed function of control quantities that do not involve x: its class-conditional losses under prior drift, its predictions and the relabeling rule under concept drift. It shows how sharp the model is, not how it responds to drift. H3 therefore sets the refit against a static model with a re-estimated intercept. Under covariate drift the static score also follows from control predictions, but through the model's skill in the tails of the data, which is the question of H1 and H4.
- The plain distance between a refit and the oracle under concept drift. It is zero for a model at chance, grows with what the model had learned, and in log loss follows the model's confidence on the rows whose labels changed after the window.
- The noise and swap arms. They leave the relabeled region without information at T4, so a stale ranking of its rows costs almost nothing in ROC AUC and the oracle gain vanishes. They are run and reported as exploratory.

What was looked at before the lock: the injection gate; cheap stand-in learners (scikit-learn models, trees against smooth learners) on the plan's windows, which set the population and the reading rule of H5; and the benchmark's stored predictions of the six models, used to find which quantities follow from control predictions (the static scores above, the log-loss distance to the oracle, and the share an ideal learner of the S1 window would reach). Those predictions were not reweighted along z, which would preview H1 and H4.

## 4. Design

### 4.1 Study A, real temporal data (descriptive)

Population: the nine datasets admitted by SDB's `early_stopping_time_plan.md` (hotels, himalaya, wildlife, f1-dnf, nba, elec, nebraska-weather, nycflights, bank-full), read from SDB's `results/es_time/prepared/*_thin.csv` with the sha256 manifest asserted. Their ordering, tie shuffle, boundary snapping, leak drops, absolute-time-column rule and label-horizon check (checks 1 to 5 of that plan, not the checks of section 6 here) apply as written there; check 2 and check 3 are recomputed on the windows below, and a dataset that fails is dropped and listed.

All nine are public data older than every TFM release, so Study A cannot separate pretraining exposure from robustness to drift; each dataset's first-public date is reported beside its curves.

Windows: the base window B1 is the first 2,000 rows, then four test blocks T1 to T4 of about 2,000 rows each, every cut moved to the next timestamp boundary. The step from B1 to T1 includes the loss of interleaved neighbours, so drift is read from T1 to T4.

Outputs: per dataset, model and strategy, the metric curves over T1 to T4 with percentile intervals from 1,000 bootstrap resamples of each block's rows, and sign counts. No p-values.

### 4.2 Study B, injected drift

n is the benchmark loader's output (the 10,000-row cap, seed 0). Population, by rules applied in this order (counts from SDB's `results/datasets.csv` and checkpoints):

| rule | left |
|---|---|
| binary, at least 1,000 rows, minority share at least 0.05, not in SDB's `DUPLICATE_DATASETS` | 23 |
| one file per source, the larger kept, on a tie the first by name (drops bank-marketing, mushroom, hill-valley-without-noise) | 20 |
| scored by all six TFMs and boosters in the benchmark (drops madelon) | 19 |
| not at the ceiling: the weakest of the six benchmark PR AUCs below 0.99 (drops banknote-authentication, mushroom-expanded, ringnorm, skin-segmentation, twonorm) | 14 |
| an expected minority count of at least 15 in the T4 block of the prior arm with r = 0.5, and 30 in B1 (drops autoUniv-au1-1000 at 14.9, ozone-eighthr, seismic-bumps, wilt) | 10 |

Study B: banana, bank-marketing-full, eeg-eye-state, hill-valley-with-noise, magic, phoneme, qsar-biodegradation, spambase, statlog-german-credit, thyroid-sick-euthyroid. These are public benchmark data that TFMs may have seen in pretraining. The benchmark's PR AUC scores label 1, which is the minority class in all ten.

The injection gate (check 1) passes every dataset in the control, covariate and prior arms. thyroid-sick-euthyroid has no admissible top feature (each of its numeric columns with at least 20 distinct values has one value above 5 % of the rows), so it leaves the three concept arms, which keep nine datasets.

The concept arm has a population of its own for H5, since the minority rule of the prior arm and the ceiling rule serve H1 to H4, and a flipped region is hard for a stale model on any dataset. Rules: the first three rows of the table (19 datasets); a minority share of at least 0.10, since on a rarer class the flip turns almost every T4 label into one side of the threshold (drops ozone-eighthr, seismic-bumps and wilt, where a step at the B1 median of the top feature alone scores ROC AUC 0.94 to 0.96 on the T4 labels); an admissible top feature (drops thyroid-sick-euthyroid, autoUniv-au1-1000, mushroom-expanded); and the region gate (drops banknote-authentication: 14 rows of the smaller class at T4 of seed 9). That leaves 12: the nine above and ringnorm, skin-segmentation and twonorm. On those three only the region arm runs, with S0, S1 and the oracle. The gate records the step's ROC AUC for every concept dataset; the highest left is 0.90 (bank-marketing-full).

Positive class: the minority class of the whole dataset (on a tie, label 1), set by `minority_is_one` in `scripts/drift_windows.py`. Prevalence and the PR AUC floor always refer to it.

Geometry, per dataset and seed: rows are split into five disjoint pools P1 to P5, stratified by class, sizes differing by at most one row. B1 is all of P1 and has no drift in any scenario, so B1, its tuning and every static model are shared by all scenarios of the dataset and seed. Test block T_h is a draw without replacement of m = floor(|P_(h+1)| / 2) rows from pool P_(h+1), h = 1..4, the same m in every scenario; each scenario draws from its own random stream. The dose fraction at T_h is h/4, so T4 carries the full dose.

Notation for the table: R is a set of rows of the block, pos_R and neg_R its counts of label 1 and label 0 before relabeling, and round is to the nearest integer, halves to even.

Scenarios, seven per dataset (the prior arm has two levels and counts twice; the three concept arms relabel the rows of a uniform draw):

| arm | construction of T_h | levels |
|---|---|---|
| none (control) | uniform draw | - |
| covariate | weighted draw over all rows of the pool, ignoring the label, with weight exp(d_h * z^2), which favours rows in either rank tail of the first principal component; P(y given x) is unchanged | - |
| prior | stratified draw to minority odds odds0 * r^(h/4), uniform within class; odds0 from the pool | r 0.5 and 2 |
| concept, noise | uniform draw; R = rows above the B1 median of the top feature; round(0.5 * h/4 * |R|) rows of R, chosen at random, get flipped labels, so R carries no information at T4 and the prevalence moves | - |
| concept, swap | uniform draw; R as above; k_h = round(h/4 * pos_R * neg_R / (pos_R + neg_R)) positives and as many negatives of R, chosen at random, swap labels, so R carries no information at T4 and the prevalence is unchanged | - |
| concept, region | uniform draw; R_h = rows above the B1 quantile 1 - 0.5 * h/4 of the top feature; every label in R_h flipped; the boundary moves every block, so a refit can learn the new rule only partly | - |

Covariate details: z is the normal score of the rank of the first principal component of the median-imputed, standardized numeric and one-hot columns, fitted on all rows without labels. The weighted draw uses Efraimidis-Spirakis keys. d_h is set per block by root finding so that the expected mean of z^2 in T_h is 1 + 0.5 * h/4, that is 1.5 at T4 (a half-pool draw can reach about 1.86 in a large pool); the inclusion probabilities of the draw come from Rosén's approximation. The tilt has no direction on purpose: a shift of the mean along z has an effect whose sign depends on the direction, and over two opposite directions the linear part of it cancels. In the gate the realized mean of z^2 at T4 is 1.48 to 1.54 against 0.97 to 1.03 under the control, and the seed mean of z stays within 0.04 of zero. The seed-mean prevalence at T4 differs from the control's by at most 0.05 (by up to 0.11 in one seed of a 100-row block). A half-pool draw at most doubles the density of any region, so no test row lies outside the range B1 covers.

Top feature: the numeric column, among those with at least 20 distinct values and no value holding more than 5 % of the rows, with the highest mutual information with the label (`mutual_info_classif`, `random_state` 0). It is chosen once per dataset on all rows, since it defines the drift and is not a model input: with its mutual information, V2 (banana, 0.11), V12 (bank-marketing-full, 0.07), V7 (eeg-eye-state, 0.05), V31 (hill-valley-with-noise, 0.03), V9 (magic, 0.12), V4 (phoneme, 0.18), V36 (qsar-biodegradation, 0.17), V57 (spambase, 0.15), V5 (statlog-german-credit, 0.01), V18 (ringnorm, 0.11), V3 (skin-segmentation, 0.25), V4 (twonorm, 0.11). A dataset without such a column is left out of the concept arms. Realized prevalence is reported for every block; skill and the PR AUC floor use it.

Oracle window, concept arms only: the rows of P4 and P5 left out of T3 and T4, a uniform draw whose size is within two rows of |B1|, relabeled by the arm's rule at the full dose (noise: half of R flipped; swap: k at h = 4 computed on the window's own R; region: every label above the B1 median flipped). It shares no row with any test block or fitting window of its scenario.

Every fitting window carries the labels of its scenario: in the concept arms S1, S2 and the oracle are fitted on relabeled rows. B1 is never relabeled. The gate and the runner take every window from one function, `windows` in `scripts/drift_windows.py`, which fixes the random stream of each scenario.

Study B has no time: its axis is dose, not horizon.

### 4.3 Strategies (both studies)

- S0 static: one fit on B1, scored on T1 to T4.
- S1 sliding: before T_h, h >= 2, a refit on the most recent |B1| rows. In Study B these are T_(h-1) and T_(h-2), with T0 a uniform draw of floor(|B1| / 2) rows of B1, drawn once per dataset and seed. In Study A the window is extended to the timestamp boundary.
- S2 expanding: before T_h, h >= 2, a refit on B1 and T1 to T_(h-1). In Study A the S2 windows at T3 and T4 exceed TabFM's 5,000-row context cap, so the runner passes TabFM the most recent 5,000 rows and the Study A S2 means are shown with and without TabFM. In Study B the largest S2 window is n/2, within the cap.
- Oracle, concept arms only: one fit on the oracle window of 4.2, scored on T4. S1's window at T4 (T3 and T2) has the same size and carries the doses 3/4 and 2/4.

At T1, S1 and S2 are S0. For TFMs a refit only changes the context.

### 4.4 Models and tuning

- TFMs: SDB's wrappers (`benchmark/models/wrappers.py`) at their constructor defaults and no grid, unlike the benchmark, which grid-searched TabPFN-3.5 and TabICLv2. All three get `random_state` equal to the seed, through the wrappers' keyword arguments. Categoricals reach them as in the benchmark. The TabPFN and TabFM weights carry non-commercial licences (see the README); the results are a research benchmark.
  - TabFM: `n_estimators` 4 (a cost choice; the library default is 32), `max_num_rows` 5,000, on MPS.
  - TabPFN-3.5: `n_estimators` 4 on every dataset (half its default of 8; an explicit count is never auto-scaled), `balance_probabilities` off, on CPU.
  - TabICLv2: `n_estimators` 8, `softmax_temperature` 0.9, on CPU.
- Versions, pinned in this project's `pyproject.toml` with the runner and asserted by it: tabpfn 9.0.0 with `model_version` v3.5, tabicl 2.1.1 with checkpoint `tabicl-classifier-v2-20260212.ckpt`, tabfm 1.0.1, catboost 1.2.10, lightgbm 4.6.0, xgboost 3.2.0, optuna 4.7.0, scikit-learn 1.7.2, numpy 2.4.6, scipy 1.16.3, torch 2.10.0. Drift-Resilient TabPFN is an optional arm outside the family means, in its own environment, kept only if it runs; the pilot decides.
- Boosters: CatBoost with its starting value at the class prior, LightGBM, XGBoost, the benchmark's search spaces without early stopping, LightGBM with `class_weight` fixed to None, since TFMs run unbalanced. On banana, the only dataset with fewer than three columns, the column-sampling parameters (`colsample_bytree` of XGBoost and LightGBM, `colsample_bylevel` of CatBoost) are fixed at 1: at two columns XGBoost gives a tree one feature at any smaller value, which cost it 0.19 PR AUC on banana in the benchmark.
- Categoricals: CatBoost native; LightGBM and XGBoost (`enable_categorical`) get pandas `category` with the level set of the whole file, a departure from the benchmark's ordinal encoder, which turns categoricals into a time index on ordered rows.
- Logistic regression is a baseline outside the family means: the benchmark's pipeline, its grid of C searched by log loss on the folds and windows the boosters are tuned on, `class_weight` None.
- Booster tuning: Optuna TPE seeded by the seed, 50 trials, on log loss (the benchmark's log-loss re-tune kept CatBoost's PR AUC and cost LightGBM 0.003). Study B: stratified 4-fold inside the window being tuned (B1, or a control S2 window), split with the seed; its rows are exchangeable. Study A: three forward folds, fitting on the first 55, 70 and 85 % of the window and validating on the next 15 %, following SDB's `early_stopping_time_plan.md`; the random-split alternative of Cai and Ye is not tested here.
- S0, S1 and the oracle use the hyperparameters tuned on B1 (their windows have B1's size). S2 windows grow, so S2 is re-tuned per window size: in Study B once on the control windows of the same dataset and seed, reused across scenarios (those windows share no row with the block they precede); in Study A on each S2 window with the forward folds above.
- Every model runs in its own process (LightGBM and torch cannot share one on macOS).

### 4.5 Metrics

Per model, strategy and block: ROC AUC, average precision of label 1, its floor (the block's realized share of label 1), skill = (AP - floor) / (1 - floor), log loss, Brier, ECE (the benchmark's function, 15 bins). Log loss is computed on float64 probabilities clipped to [1e-6, 1 - 1e-6] for every model, and S0* (section 5) is built from clipped probabilities and clipped again; the benchmark's clip depends on the output dtype, which differs by model. Every test block in Study B has m rows, so control and drift are compared at equal size. ROC AUC is the primary ranking metric: it does not move with the block prevalence alone, which the prior and concept arms change. Skill is reported next to it.

### 4.6 Seeds

Three seeds, 0 to 2; datasets under 2,000 rows (qsar-biodegradation, hill-valley-with-noise, statlog-german-credit) get ten, since their blocks of 100 to 121 rows are the noisiest. In Study B the seed sets the pools, the draws and every model's and sampler's seed; in Study A only the model and sampler seeds. Per-dataset values are averaged over seeds before any test.

## 5. Analysis

Unit: the dataset, n = 10 for H1 to H4. H5 has at most 12 and is expected to run on 11: the boosters are at chance on hill-valley-with-noise in the benchmark, so its guard below should fail there, while every other dataset cleared it for both stand-in families with a gain of at least 0.13. Holm over the five primary tests, alpha 0.05; each p is reported with a Hodges-Lehmann estimate and its 95 % interval from the Walsh averages and the exact signed-rank quantile; a zero difference is dropped. A test needs at least 8 datasets (two-sided p_min 0.0078 at n = 8); with fewer it is reported as untestable and leaves the Holm family, which then has four tests. A dataset with a missing score in any cell of a test leaves that test. Every test is also reported with each dataset left out in turn, as the Hodges-Lehmann estimate and the unadjusted exact p, with no verdict.

| test | quantity per dataset | outcome |
|---|---|---|
| H1 | lead at T4 under S0, covariate minus control | ROC AUC |
| H2 | lead in (S1 - S0) at T4, covariate minus control | ROC AUC |
| H3 | lead in (S1 - S0*) at T4, prior minus control | log loss |
| H4 | lead at T4 under S0, covariate minus control | log loss |
| H5 | TFM share minus booster share at T4, region | ROC AUC |

For H1 to H4 the value is the mean over the seeds (and, for the prior arm, its two levels), minus the same quantity under the control; each prior level is also reported alone.

H1 and H4 are read with two companions, since the tilted block is easier than the control's for most learners and an easier block shrinks the lead of whichever family is ahead: the change from control to covariate of each family's error (1 - ROC AUC) and log loss, and the log error ratio log((1 - AUC of TFMs) / (1 - AUC of boosters)), covariate minus control, with errors floored at 0.001. A rejection of H1 is read as a different response to drift only if the log error ratio moves in the same direction; otherwise it is reported as a change of scale.

S0* in H3 is the static model with its intercept re-estimated on the S1 window before T4: the one constant c that maximizes the likelihood of the window's labels given logit(S0) + c, with S0's probabilities clipped as in 4.5. S0 already predicts those rows, so no model is fitted. It is computed per model, the same way under the control, and makes H3 zero for any model that only updates its intercept, whatever its calibration.

The share in H5: for each dataset and family, the ROC AUC at T4 of S0, of S1 and of the oracle is averaged over the family's three models and the seeds; the oracle gain is oracle - S0 and the share is (S1 - S0) / (oracle - S0). This ratio of family means weights each model by its own oracle gain; it is not the mean of the models' shares. A dataset enters H5 only if the oracle gain is at least 0.05 for both families; the others are listed, and a single model's share is shown only where its own gain reaches 0.05.

How the share reads. It is 0 for a model that does not adapt, and a monotone transform of a model's scores leaves it unchanged. An ideal learner of the S1 window reaches about 0.45 to 0.75 depending on the dataset, since the window carries the doses 2/4 and 3/4, and a share above 1 occurs when the oracle fit learns the new rule worse than the S1 fit. H5 reads the difference of the two shares, never a level. Between two families the share falls as the oracle gain rises, so the family that learns the flipped rule worse from a full-dose window tends to get the larger share. A second effect comes from the boundary's movement: rows between the B1 median and the B1 0.625 quantile of the top feature are flipped at T4 and in neither half of the S1 window, so a learner gets them right only by extrapolating the boundary across them, which a smooth learner does and a tree does not. In the stand-in run on the eleven datasets past the guard, smooth learners recovered the larger share on 9 of 11 (exact p 0.005), while trees gained more from the refit in absolute ROC AUC on 11 of 11 (p 0.001); outside the band the share difference kept its sign but not its significance (p 0.054).

A rejection is worded as "recovers a larger share of its own full-dose fit". It is read as a difference in adaptation only if, on the datasets H5 tested, both of these have the sign of H5's Hodges-Lehmann estimate at exact two-sided p < 0.05:

1. The lead in S1 - S0 at T4, TFM family mean minus booster family mean over the seeds. Otherwise the result is reported as a difference in oracle level: one family recovers a larger share without gaining more.
2. The share difference recomputed on the T4 rows outside the band (B1 median, B1 0.625 quantile] of the top feature, with S0, S1 and the oracle scored on the same rows, averaged as in H5 and with the same guard on its oracle gain. Otherwise the result is reported as extrapolation across the moving boundary.

If both fail, both are named. The two are not Holm-adjusted: they decide the wording of a rejection, not the rejection. On the stand-in run the rule reads the trees against smooth learners contrast as a difference in oracle level, and no mixed split of the six learners reaches H5's significance.

Reported next to H5: both families' oracle gains, the two quantities of the rule with their exact p, and, on the nine datasets with a control, the lead under S1, region minus control.

Exploratory, reported as such: the noise and swap arms, with their realized relabeled share and prevalence and their shares where the denominator allows; the oracle levels (which family learns the new rule better); the distance S1 - oracle in ROC AUC and log loss; the static model under prior and concept drift next to the control quantities it follows from; the lead under S1 in the covariate arm minus the control's, since H2 equals it minus H1; each family's calibration slope on the control's T4, next to H3 and H4; per-model results; slopes over T1 to T4 against the realized dose; S2; skill, ECE and Brier; Study A; and the comparison with the benchmark's random-split ordering.

## 6. Checks

Before the lock:

1. Injection gate, on the ten datasets and the concept candidates of 4.2, every seed, no model fitted. Covariate: the seed-mean realized mean of z^2 at T4 lies within 0.1 of 1.5. Prior: the realized minority count is within one row of the schedule. Concept: the relabeled counts match the schedule exactly, in the test blocks and in the oracle window (the unit tests recompute the schedule independently), and swap leaves the prevalence unchanged. Every arm: every test block and oracle window holds at least 15 rows of its smaller class after relabeling, and B1 at least 30 minority rows. A dataset that fails is dropped from that arm before the lock and listed (result in 4.2).

Before any result is read:

2. Count missing scores per model, strategy, block and seed.
3. Compare every score with its floor (logic of SDB's `scripts/check_prevalence_baseline.py`, realized block prevalence). The concept arms are exempt: a stale model is expected below the floor there.
4. Assert that every model's `y_prob[:, 1]` is the minority class.
5. Leakage tests in `tests/test_drift_windows.py`. Study B, written with the gate and run on synthetic data, since the windows are pure functions: no row is in two blocks of one scenario, B1 is never tested, no S1 or S2 window holds a row of the block it precedes, the control's S2 windows hold no row of any scenario's block they precede, and the oracle window shares no row with a test block or an S1 or S2 window of its scenario. Study A, added with its windows before any model run: every fitting and tuning index precedes every test index.
6. Control diagnostic: under none, the Hodges-Lehmann 99 % interval over the ten datasets of the lead's difference between T4 and T1, S0, is reported per outcome. If it excludes 0 the code is reviewed before anything is read; the windowing itself is guarded by check 5.
7. Join every result table by dataset, scenario and seed name, never by position.

## 7. Compute

On the ten Study B datasets the benchmark's log-loss-tuned boosters took CatBoost 0.70 h, LightGBM 0.10 h and XGBoost 0.11 h in nested CV. Per dataset and seed a booster here needs four tunings (B1 and three S2 window sizes, 201 fits each) plus 45 refits, about 850 fits on windows of 0.15 n to 0.5 n rows, against 804 fits on about 0.56 n in the benchmark: about half a benchmark run in rows fitted per seed. The TFMs took 0.35 h (TabFM, 4 fits per dataset, about 31 s per fit on MPS, each with its own load of the checkpoint), 2.24 h (TabPFN-3.5, 68 fits) and 3.50 h (TabICLv2, 148 fits). Here each needs 46 contexts and 73 block predictions per dataset and seed, on smaller contexts, so TabFM grows most. The 28 block predictions of S0 become one pass over the rows outside B1 for a TFM whose prediction of a row does not depend on the other test rows; the pilot checks that for each by comparing one pass with block-wise passes. The three concept-only datasets add, per seed, one booster tuning and five fits, and five contexts for a TFM. Study A adds nine datasets of 10,000 rows. The pilot (one Study A and two Study B datasets, one seed, all models) prints timings, that comparison and whether Drift-Resilient TabPFN runs, and no score; the budget is set after it. If it is short, the reductions apply in this order: Drift-Resilient TabPFN, the extra seeds of the small datasets, S2 tuning (B1 parameters reused), the third seed.

## 8. Deliverables

- `scripts/drift_windows.py`, committed with this plan: pools, draws, injections, the oracle window, the Study B fitting windows (`windows`, the one entry point of the runner) and the injection gate as pure functions, with `tests/test_drift_windows.py`. The Study A windows are added with the runner.
- `scripts/drift_run.py`: one model, one dataset and seed; stores per-block predictions in a checkpoint.
- `scripts/drift_analysis.py`: `results/drift_*.csv` and the figures.
- A findings section in this project's README per outcome, nulls included.

## 9. Limits

- Injected drift is synthetic. Study A is real but small and descriptive.
- The covariate arm moves test rows to the rank tails of one component and has no direction; a shift of the mean is not tested. It reweights inside B1's support and at most doubles the density of any region, so the static model never extrapolates. The tilted block is easier than the control's for most learners, since the tails of the component are close to pure in class, and the refit contrast of H2 is expected to be small; a null there is not evidence about stronger shifts.
- H3 reads only what a refit adds beyond an intercept correction, which is all a model needs under a pure prior shift. The refit's window also holds fewer (r = 0.5) or more (r = 2) minority rows than B1, so H3 includes each family's sample efficiency in the minority class; the per-level results show it.
- Concept drift is tested on one arm, a label flip above a moving threshold of one feature. Trees learn an axis-aligned threshold natively; that raises their oracle gain and, through it, lowers their share. A learner that smooths across the moving boundary recovers more than one that reproduces the window's thresholds exactly. The reading rule of section 5 names either effect when it, and not a larger absolute gain, carries the result; a real difference in adaptation that sits inside the band is reported as extrapolation.
- The oracle is a reference fit, not a ceiling: for boosters it and S1 run at hyperparameters tuned on B1's old rule, and H5 has no control term to remove that. Half of its window is the complement of T4 inside a class-stratified pool, so its class counts mirror T4's; that does not move ROC AUC.
- Three of the twelve concept datasets sit at the benchmark's ceiling under no drift (ringnorm, skin-segmentation, twonorm), and ringnorm and twonorm come from one pair of Gaussian generators, so they are not two independent designs; the leave-one-out lines show what each decides. On statlog-german-credit the top feature carries little of the label (mutual information 0.01), so the flip there runs along an almost arbitrary axis; on bank-marketing-full the step at the median alone scores 0.90 on T4. With eleven datasets H5 can also detect a difference that comes from how the two families are built rather than from drift; the reading rule decides how a rejection is worded.
- On hill-valley-with-noise the boosters are at chance in the benchmark (ROC AUC 0.54 to 0.56) and the TFMs near 1, so the boosters have nothing to lose there. The leave-one-out lines show what it decides, and the guard of H5 should remove it.
- Ten datasets for H1 to H4 and five tests: at the first Holm step (0.01) the ranks of the disagreeing datasets may sum to 3 on ten. H5 on eleven allows one disagreeing dataset up to rank 5.
- Boosters are tuned on log loss and scored by ROC AUC in H1, H2 and H5; TFMs are not tuned at all.
- All windows are small (B1 is 200 to 2,000 rows), the regime where TFMs are strongest.
- Boosters keep B1's hyperparameters in S1; a booster re-tuned on every window could do better. Under the control this shows as a non-zero S1 - S0 difference, which the difference in differences removes from H2 and H3; H5 has no control term.
- TabFM runs at 4 estimators, an eighth of its library default, and TabPFN-3.5 at half of its default.
- Study B's random pools make neighbouring rows of eeg-eye-state (one recording) fall into different blocks, B1 included, which flatters every model on that dataset, under the control as well.
