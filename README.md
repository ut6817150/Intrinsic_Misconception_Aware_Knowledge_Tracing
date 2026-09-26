# Intrinsic Misconception-Aware Knowledge Tracing

Code, annotated data, experiment notebooks, and fitted outputs for Experiment 2 of the MSc thesis
*Model User’s Skills based on Language* (Imperial College London). The experiment asks whether
intrinsic misconceptions—systematic errors in understanding conditional probability and Bayes’
theorem—can be observed in spoken reasoning and whether tracing them alongside knowledge components
(KCs) improves Bayesian Knowledge Tracing.

The thesis focuses on a four-model progression: standard BKT, KC-level evidence, typed misconception
observations, and class-aware misconception persistence. It also reports a gate-only ablation and seven
pooling functions. This repository contains those experiments **and** additional outer-model references,
alternative misconception representations, persistence variants, and a fitted persistence dial.

The application used to conduct the trial is in the separate
[Reachy_Mini_KT_Data_Gathering](https://github.com/ut6817150/Reachy_Mini_KT_Data_Gathering)
repository.

## Dataset

Twenty-six university students answered one unscored warm-up and twelve scored Bayesian reasoning
questions in a single untimed session. They explained every answer aloud to a Reachy Mini social robot.
The robot provided no instruction, correction, or feedback, so any within-session improvement arose
through self-repair.

`data/data_annotated.csv` contains 338 rows: 26 warm-up rows and 312 scored responses. The scored
responses contain 200 correct and 112 incorrect answers, 768 engaged KC cells, and 294 applicable
misconception cells. All annotation was performed manually from audited transcripts against
`data/bayes_annotation_codebook.md`.

| Channel | Columns | Labels |
|---|---|---|
| Question correctness | `question_correct` | `correct`, `wrong` |
| KC correctness | `kc1_sample_space`, `kc2_conditioning`, `kc3_joint_chain`, `kc4_total_probability`, `kc5_bayes_update` | `correct`, `wrong`, `NA` |
| Misconception observations | `conjunction`, `inverse`, `time_axis`, `denominator_neglect`, `base_rate_neglect` | `fired`, `quiet`, `NA` |

The five KCs are probability foundations, conditional probability, joint probability, total
probability, and Bayesian updating. A KC cell records whether the participant applied that concept
soundly, applied it unsoundly, or did not engage it. Arithmetic slips affect question correctness without
automatically making a KC cell wrong.

A misconception cell is `fired` only when the misconception is manifest in the verbalised reasoning,
`quiet` when the reasoning provided an opportunity for it to arise and affirmatively demonstrated its
absence, and `NA` when the reasoning did not exhibit enough information to judge it. The loader
represents the CSV’s structural `NA` values internally as empty strings.

`designed_kcs` gives the KCs required by a question’s designed solution. `adaptive_kcs` records KCs
engaged by the participant’s exhibited method outside that set, including alternative valid routes.
`annotation_rationale` gives the written justification for every row’s labels. The loader excludes Q0
from analysis.

| Misconception | Applicable | Quiet | Fired |
|---|---:|---:|---:|
| Conjunction | 22 | 22 | 0 |
| Inverse | 26 | 25 | 1 |
| Time axis | 34 | 28 | 6 |
| Denominator neglect | 91 | 88 | 3 |
| Base-rate neglect | 121 | 114 | 7 |
| **Total** | **294** | **277** | **17** |

## Models

All models use 26-fold leave-one-participant-out cross-validation. In each fold, fitted parameters are
estimated on 25 participants, then the held-out participant’s twelve responses are predicted
sequentially. Each response is predicted before its own observations update the model.

### Main progression

| Model | Implementation | Addition |
|---|---|---|
| Baseline | `scripts/model_1_1.py` | Standard multi-skill BKT. Each question-level correctness label is inherited by every designed KC, and KC predictions are pooled by the compensatory mean. |
| KC-evidence | `scripts/model_1_2_internal_chain.py` and `scripts/model_1_2_outer_chain.py` | Fits one BKT chain to each KC’s independently annotated correctness, then pools KC-correctness probabilities into question correctness through global slip and guess. |
| Misconception-observation | `Model_2_1_1_Joint` in `scripts/model_2_1_1.py` | Widens the observation alphabet so that a wrong KC cell and its applicable misconception observations form one joint symbol. |
| Misconception-observation with freeze | `scripts/model_2_2.py` | Adds class-aware persistence: a bias-class fire removes the free learning drift of its host KC for the remainder of the session. |

The outer layer pools predicted **KC correctness**, rather than mastery latents directly. For KC (k),
the internal chain converts mastery (m_k) to correctness probability
(p_k=m_k(1-s_k)+(1-m_k)g_k). The outer pooling function combines the (p_k), after which global
question-level slip and guess map the pooled value to question correctness. The global parameters are
refitted within every training fold by constrained likelihood grid search; they are not EM estimates or
parameters copied from an earlier model.

The adopted route mixture treats the route as unobserved on Q1, Q4, Q6, and Q7. It computes a
conjunctive prediction for each valid route and combines the two predictions with a fitted mixture weight.

The class-aware rule treats conjunction, inverse, time-axis, and base-rate neglect as bias-class
misconceptions. Denominator neglect is the sole skill-class misconception. A bias-class fire freezes the
learning transition of its host KC; a skill-class fire leaves that transition running. Evidence continues
to update mastery after a freeze, so a later correct KC observation can still raise the belief.

### Additional models and ablations in this repository

Several internal-chain ablations were also run during model development.

Notebook `03` evaluates twelve outer models. The seven pooling rules stated in the thesis are
`Model_1_2_AND`, `Model_1_2_MIN`, `Model_1_2_MEAN`, `Model_1_2_GENMEAN`,
`Model_1_2_TEMPAND`, `Model_1_2_LEAKY`, and the adopted `Model_1_2_MIX2`. The repository also
contains `Model_1_2_MAX`, `Model_1_2_DINO`, `Model_1_2_OWA`, `Model_1_2_LOGIT`, and
`Model_1_2_ISO` as disjunctive, ordered-weighting, learned, and calibration references.

Notebooks `04` and `05` test alternative treatments of misconception evidence:

- `Model_2_1_1_Factorized` — the registered original and predictive benchmark; multiplies cell and
  flag likelihoods as conditionally independent witnesses and uses literature-centred marginal fire rates.
- `Model_2_1_1_Factorized_No_Literature` — removes the literature calibration.
- `Model_2_1_1_Joint_No_Shrink` — removes neutral shrinkage from the canonical joint fire rates.
- `Model_2_1_1_Classic_BKT` — adds marginal flag evidence to the inherited-label M1.1 chassis.
- `Model_2_1_2` — maintains a four-cell joint belief over mastery and a separate latent misconception
  disposition.

Notebooks `06` through `08` test persistence and the placement of the gate:

- `Model_2_2_Gate_Only` — applies the freeze to KC evidence without flag likelihood factors.
- `Model_2_2_On_State_View` — mounts the freeze on the M2.1.2 latent-state chassis.
- `Model_2_2_Unratcheted` — allows a later quiet to release a frozen KC and a later fire to refreeze it.
- `Model_2_3` — replaces the hard rule with fitted bias- and skill-class transition multipliers.

## Repository structure

| Path | Contents |
|---|---|
| `data/data_annotated.csv` | Pseudonymised transcripts and all annotation channels |
| `data/bayes_annotation_codebook.md` | Annotation rules and per-question rulings |
| `scripts/data.py` | Dataset loader and structural-`NA` handling |
| `scripts/evaluator.py` | Leave-one-participant-out evaluation and metrics |
| `scripts/model_*.py` | Model implementations, ablations, and persistence helpers |
| `00_eda.ipynb` | Dataset description and annotation-channel audit |
| `01_standard_bkt_model_1_1.ipynb` | Standard multi-skill BKT baseline |
| `02_model_1_2_*.ipynb` | Internal-chain fitting and selection |
| `03_model_1_2_outer_chain.ipynb` | Pooling comparison and route mixture |
| `04_model_2_1_1_flag_observations.ipynb` | Misconception-observation model and emission ablations |
| `05_model_2_1_2_flag_state.ipynb` | Separate latent misconception-state representation |
| `06_model_2_2_class_aware_gate.ipynb` | Class-aware persistence model |
| `07_model_2_2_ablations.ipynb` | Gate-only, state-view, and unratcheted ablations |
| `08_model_2_3_persistence_dial.ipynb` | Fitted persistence multipliers |
| `cache/` | Saved folds, predictions, metrics, and model indexes |

## Results

### Question-level models and ablations

All rows below use the same 312 held-out question targets. Values come from the current notebook or
cached outputs; a dash indicates that the displayed notebook ablation table did not retain that metric.

| Model | Accuracy | F1 | AUC | Incorrect AUPRC | Log loss |
|---|---:|---:|---:|---:|---:|
| M1.1 baseline | 0.7051 | 0.8099 | 0.5973 | 0.5164 | 0.6283 |
| M1.1 with marginal flags | 0.7147 | — | 0.6103 | 0.5323 | 0.6221 |
| M1.2 MIX2 KC-evidence | 0.6859 | 0.7860 | 0.6741 | 0.5747 | 0.6070 |
| M2.1.1 joint observation | 0.7019 | 0.7974 | 0.6870 | 0.5827 | 0.6029 |
| M2.1.1 factorized, calibrated | 0.6923 | — | 0.6907 | 0.5907 | 0.6003 |
| M2.1.1 factorized, no literature calibration | 0.6987 | — | 0.6867 | 0.5858 | 0.6031 |
| M2.1.1 joint, no shrinkage | 0.7051 | — | 0.6837 | 0.5779 | 0.6051 |
| M2.1.2 latent state | 0.6955 | 0.7939 | 0.6658 | 0.5651 | 0.6112 |
| M2.2 gate only | 0.7051 | 0.7991 | 0.6829 | 0.5858 | 0.6031 |
| M2.2 gate on M2.1.2 state view | 0.6987 | 0.7957 | 0.6729 | 0.5725 | 0.6077 |
| M2.2 joint observation with freeze | 0.7051 | 0.7974 | 0.6980 | 0.5787 | 0.5961 |
| M2.2 unratcheted | 0.7051 | 0.7974 | 0.6995 | 0.5827 | 0.5958 |
| M2.3 fitted persistence dial | 0.7051 | 0.7974 | 0.6981 | 0.5795 | 0.5959 |

The thesis reports the main progression as AUC 0.5975 for the baseline, 0.6741 for KC evidence, 0.6870
for misconception observations, and 0.6980 after class-aware persistence. Its paired participant-bootstrap
comparisons are +0.0766 `[+0.0027, +0.1774]`, +0.0129 `[−0.0057, +0.0356]`, and +0.0110
`[−0.0011, +0.0277]`, respectively. The first increment is clearly separated from zero; the two
misconception-side increments are inconclusive. The full stack improves by +0.1006
`[+0.0216, +0.2096]` over the baseline.

The submitted thesis baseline differs slightly from the current cached predictions: the thesis reports
AUC 0.5975 and incorrect-response AUPRC 0.5163, whereas the cache produces 0.5973 and 0.5164. The
table uses current repository outputs and the paragraph above preserves the submitted values.

The joint M2.1.1 form is adopted for measurement fidelity even though the factorized benchmark has an
approximately 0.004 AUC edge: the factorized form treats a fired flag and its necessarily wrong host cell
as independent evidence and therefore counts the failure twice. M2.1.2 behaves as designed on trap-free
probes but is weaker on pooled question prediction. The unratcheted model is effectively tied with the
hard ratchet. In M2.3, all 26 folds select a zero bias-class multiplier, while the skill multiplier is
unidentified because KC4’s transition is already close to zero; the fitted dial adds no material value over
M2.2.

Only 51 of the 294 applicable misconception cells can add emission evidence beyond KC correctness,
and only 14 bias-class fires across seven participants can trigger persistence. Conclusions about the
misconception additions are consequently limited by sparse evidence, a single annotator, a small and
specialised cohort, and the domain-specific misconception catalogue.

## Running the experiments

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install jupyterlab
jupyter lab
```

Run the notebooks in numerical order. Notebook `02` writes the adopted internal-chain folds under
`cache/`. Later notebooks inject those cached chains rather than refitting them, but still refit their focal
model’s flag tables, pooling weight, global slip and guess, and any model-specific outer parameters within
every training fold. They load stored predictions for earlier comparison models and write their own
fitted parameters and predictions to `cache/` in their final persistence cells.

To fit and score the baseline directly:

```python
from scripts.data import load_data
from scripts.evaluator import Evaluator
from scripts.model_1_1 import Model_1_1

df = load_data("data/data_annotated.csv")
kwargs = {"n_restarts": 3}
print(Evaluator(Model_1_1, df, model_kwargs=kwargs).run().metrics)
```

Models that consume the internal chains take them as a keyword argument:

```python
from scripts.model_1_2_outer_chain import Model_1_2_MIX2, load_internal_chains

chains = load_internal_chains("cache/model_1_2_internal_chain/Model_1_2_Internal_Slip_And_Guess")
kwargs = {"n_restarts": 3, "chain_cache": chains}
print(Evaluator(Model_1_2_MIX2, df, model_kwargs=kwargs).run().metrics)
```

`scripts/evaluator.py` pools held-out predictions and computes accuracy, F1, balanced accuracy, AUC,
log loss, and AUPRC with the incorrect response as the positive class.

## Ethics and data protection

The study, *Teaching a Robot Probability Theory*, was approved by the Head of Department on
11 August 2026 and by Imperial College London’s Research Governance and Integrity Team on
12 August 2026 under SETREC reference 8481955. Participants provided signed consent before any
audio or video was recorded and could withdraw without giving a reason.

The released data is pseudonymised by participant ID and contains annotated transcripts without the
underlying audio or video recordings. Identifiable research data is not included in this repository.
