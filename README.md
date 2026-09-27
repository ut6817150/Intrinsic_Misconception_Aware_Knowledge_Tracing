# Intrinsic Misconception-Aware Knowledge Tracing

This repository contains the code, annotated data, experiment notebooks, and fitted outputs for
Experiment 2 of the MSc thesis *Model User’s Skills based on Language* (Imperial College London).
The experiment asks whether spoken reasoning reveals systematic misconceptions about conditional
probability and Bayes’ theorem, and whether those observations improve Bayesian Knowledge Tracing
(BKT).

The central experiment follows four models: a standard BKT baseline, a model trained on independently
annotated knowledge-component evidence, a model that also observes misconceptions, and a model that
uses misconception class to control learning persistence. The repository also contains supporting model
variants and ablations used during development.

The code used to collect the trial data is maintained separately in
[Reachy_Mini_KT_Data_Gathering](https://github.com/ut6817150/Reachy_Mini_KT_Data_Gathering).

## Repository structure

The numbered notebooks follow the order in which the data and models are processed. Each notebook
uses the implementation beside it in the table and saves reusable outputs under `cache/`:

| Stage | Notebook and implementation | Contents |
|---|---|---|
| Data preparation | `data/`, `scripts/data.py` | Pseudonymised transcripts, annotations, the annotation codebook, and dataset loading |
| Data checks | `00_data_and_annotation.ipynb` | Dataset structure and annotation-channel checks |
| Baseline | `01_baseline.ipynb`, `scripts/model_1_1.py` | Standard multi-skill BKT fitted from question correctness |
| KC-evidence: internal chains | `02_kc_evidence_internal_chains.ipynb`, `scripts/model_1_2_internal_chain.py` | Per-KC BKT chains fitted from annotated KC correctness |
| KC-evidence: question model | `03_kc_evidence.ipynb`, `scripts/model_1_2_outer_chain.py` | The selected `MIX2` route-mixture pool and global question-level emissions |
| Misconception-observation | `04_misconception_observation.ipynb`, `scripts/model_2_1_1.py` | Typed misconception observations added to the KC emission model |
| Misconception-state variant | `05_misconception_state_variant.ipynb`, `scripts/model_2_1_2.py` | An alternative representation with a separate latent misconception disposition |
| Class-aware persistence | `06_class_aware_persistence.ipynb`, `scripts/model_2_2.py` | A persistent freeze of the learning transition after a bias-class fire |
| Persistence variants | `07_class_aware_persistence_variants.ipynb`, `scripts/model_2_2_ablations.py` | Gate-only, state-view, and releasable-freeze variants |
| Fitted persistence variant | `08_fitted_persistence_variant.ipynb`, `scripts/model_2_3.py` | Fitted bias- and skill-class transition multipliers |
| Shared evaluation | `scripts/evaluator.py`, `cache/` | Leave-one-participant-out evaluation, fitted folds, predictions, and metrics |

## Report alignment

The report names map to the code as follows:

| Name used in the report | Code implementation | Notebook |
|---|---|---|
| Baseline | `Model_1_1` | `01_baseline.ipynb` |
| KC-evidence | `Model_1_2_MIX2` | `02_kc_evidence_internal_chains.ipynb` then `03_kc_evidence.ipynb` |
| KC-evidence with freeze | `Model_2_2_Gate_Only` | `07_class_aware_persistence_variants.ipynb` |
| Misconception-observation | `Model_2_1_1_Joint` | `04_misconception_observation.ipynb` |
| Misconception-observation with freeze | `Model_2_2` | `06_class_aware_persistence.ipynb` |

The remaining classes implement emission alternatives, the latent misconception-state view, alternative
gate placements, a releasable freeze, and fitted persistence multipliers. Several internal-chain ablations
were also run during model development.

Within the KC-evidence model, each internal chain converts its mastery probability into a KC-correctness
probability, `p_k = m_k(1 - s_k) + (1 - m_k)g_k`. `MIX2` pools these correctness probabilities rather
than the mastery latents. The global question-level guess, slip, and route-mixture weight are refitted in
each training fold by constrained likelihood search; they are not fitted by EM or copied from an earlier
model.

## Data and annotation

Twenty-six university students completed one unscored warm-up and twelve scored Bayesian reasoning
questions in a single untimed session. They explained each answer aloud to a Reachy Mini social robot.
The robot gave no instruction, correction, or feedback, so any change within a session arose through
self-repair.

`data/data_annotated.csv` contains 338 rows: 26 warm-up responses and 312 scored responses. Of the
scored responses, 200 are correct and 112 are incorrect. The analysis excludes the warm-up question,
Q0.

Each scored response contains three annotation channels:

| Channel | Columns | Labels |
|---|---|---|
| Question correctness | `question_correct` | `correct`, `wrong` |
| KC correctness | `kc1_sample_space`, `kc2_conditioning`, `kc3_joint_chain`, `kc4_total_probability`, `kc5_bayes_update` | `correct`, `wrong`, `NA` |
| Misconception observations | `conjunction`, `inverse`, `time_axis`, `denominator_neglect`, `base_rate_neglect` | `fired`, `quiet`, `NA` |

The five knowledge components (KCs) represent probability foundations, conditional probability, joint
probability, total probability, and Bayesian updating. KC correctness records whether the participant
used an engaged concept soundly. It remains `NA` when that concept was not engaged. An arithmetic slip
can therefore make the final answer wrong without making a KC observation wrong.

A misconception is `fired` when it is visible in the reasoning, `quiet` when the reasoning gives it an
opportunity to appear and demonstrates its absence, and `NA` when the response does not provide enough
evidence to judge it. Structural `NA` values are represented internally as empty strings. The observed
misconception counts are:

| Misconception | Class | Applicable | Quiet | Fired |
|---|---|---:|---:|---:|
| Conjunction | Bias | 22 | 22 | 0 |
| Inverse | Bias | 26 | 25 | 1 |
| Time axis | Bias | 34 | 28 | 6 |
| Denominator neglect | Skill | 91 | 88 | 3 |
| Base-rate neglect | Bias | 121 | 114 | 7 |
| **Total** |  | **294** | **277** | **17** |

Only 51 of the 294 applicable misconception cells add emission evidence beyond KC correctness. The
class-aware persistence rule can be triggered by 14 bias-class fires, observed across seven participants.
A bias-class fire freezes the free learning transition of its host KC for the rest of the session; the
skill-class misconception leaves that transition active. Evidence continues to update mastery after a
freeze.

The CSV also records the intended KCs in `designed_kcs`, any additional KCs used by the participant in
`adaptive_kcs`, and the justification for each row’s labels in `annotation_rationale`. Full annotation
rules and per-question rulings are in `data/bayes_annotation_codebook.md`.

## Results

The five configurations reported in the thesis were evaluated on the same 312 held-out question targets
using 26-fold leave-one-participant-out cross-validation. The internal KC chains and the selected `MIX2`
outer layer are reported together as the KC-evidence model:

| Model | Code | Accuracy | F1 | AUC (95% CI) | Incorrect AUPRC (95% CI) |
|---|---|---:|---:|---:|---:|
| Baseline | `Model_1_1` | 0.7051 | 0.8099 | 0.5975 `[0.4895, 0.6751]` | 0.5163 `[0.3526, 0.6404]` |
| KC-evidence | `Model_1_2_MIX2` | 0.6859 | 0.7860 | 0.6741 `[0.6123, 0.7257]` | 0.5747 `[0.4325, 0.6854]` |
| KC-evidence with freeze | `Model_2_2_Gate_Only` | 0.7051 | 0.7991 | 0.6829 `[0.6199, 0.7369]` | 0.5858 `[0.4406, 0.6968]` |
| Misconception-observation | `Model_2_1_1_Joint` | 0.7019 | 0.7974 | 0.6870 `[0.6223, 0.7448]` | 0.5827 `[0.4382, 0.6949]` |
| Misconception-observation with freeze | `Model_2_2` | 0.7051 | 0.7974 | 0.6980 `[0.6315, 0.7585]` | 0.5787 `[0.4492, 0.6955]` |

Incorrect responses are the positive class for AUPRC; their prevalence gives a no-skill AUPRC of
0.359.

## Running the experiments

Create an environment and start JupyterLab:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install jupyterlab
jupyter lab
```

Run the notebooks in numerical order. Notebook `02` saves the adopted internal KC chains under
`cache/`. Each later notebook reuses those folds, refits its own focal parameters within the training
participants, and saves its fitted parameters and held-out predictions back to `cache/`.

The baseline and internal BKT chains fit their standard BKT parameters using EM, with guess and slip
constrained to at most 0.3. Models with an outer question layer fit its global guess, slip, and any pooling
parameters by constrained likelihood search inside each cross-validation fold.

## Ethics and data protection

The study, *Teaching a Robot Probability Theory*, was approved by the Head of Department on
11 August 2026 and by Imperial College London’s Research Governance and Integrity Team on
12 August 2026 under SETREC reference 8481955. Participants provided signed consent before audio or
video recording and could withdraw without giving a reason.

The released data is pseudonymised by participant ID and contains annotated transcripts without the
underlying audio or video recordings. Identifiable research data is not included in this repository.
