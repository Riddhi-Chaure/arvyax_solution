# ArvyaX Emotional Wellness Prediction System
### ArvyaX ML Internship Assignment — RevoltronX
**Theme: From Understanding Humans → To Guiding Them**

---

## What This System Does

Users attend immersive ambience sessions (forest, ocean, rain, mountain, café) and write short journal reflections afterward. This system reads those reflections alongside lightweight contextual signals (sleep, stress, energy, time of day) and:

1. **Understands** the user's emotional state and intensity
2. **Decides** what action they should take and when
3. **Knows when it is unsure** — flags uncertain predictions honestly
4. **Generates** a short supportive message explaining the recommendation

---

## Project Structure

```
arvyax_solution/
├── data/
│   ├── train.csv                  # original training data (1200 rows)
│   ├── test.csv                   # original test data (120 rows)
│   ├── train_clean.csv            # cleaned + feature-engineered train
│   └── test_clean.csv             # cleaned + feature-engineered test
├── models/
│   ├── clf_state.pkl              # emotional state classifier
│   ├── clf_intensity.pkl          # intensity classifier
│   ├── le_state.pkl               # label encoder for state classes
│   ├── tfidf.pkl                  # fitted TF-IDF vectorizer
│   ├── encoders.pkl               # fitted label encoders for metadata
│   ├── X_train_full.npz           # combined feature matrix (train)
│   ├── X_test_full.npz            # combined feature matrix (test)
│   ├── X_train_text_only.npz      # text-only matrix (ablation)
│   └── X_train_meta_only.npz      # metadata-only matrix (ablation)
├── src/
│   ├── preprocess.py              # cleaning, feature engineering, TF-IDF
│   ├── train.py                   # model training, ablation, feature importance
│   ├── predict.py                 # inference pipeline → predictions.csv
│   ├── decision_engine.py         # rule-based what_to_do + when_to_do
│   └── uncertainty.py             # confidence scores + uncertain_flag
├── outputs/
│   └── predictions.csv            # final submission file
├── README.md
├── ERROR_ANALYSIS.md
└── EDGE_PLAN.md
```

---

## Setup Instructions

### Requirements
- Python 3.8+
- pip

### Step 1 — Create virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### Step 2 — Install dependencies
```bash
pip install pandas numpy scikit-learn xgboost nltk scipy joblib
```

### Step 3 — Download NLTK stopwords
```python
python -c "import nltk; nltk.download('stopwords')"
```

### Step 4 — Place data files
```
data/train.csv   ← Sample_arvyax_reflective_dataset_xlsx_-_Dataset_120.csv
data/test.csv    ← arvyax_test_inputs_120_xlsx_-_Sheet1.csv
```

---

## How to Run

Run scripts in this exact order:

```bash
# Step 1: Clean data + build feature matrices
python src/preprocess.py

# Step 2: Train models + ablation study
python src/train.py

# Step 3: Generate predictions
python src/predict.py
```

Output file: `outputs/predictions.csv`

---

## Approach

### This is NOT a standard classification problem

Real-world wellness data is messy:
- Journal texts average only **10.9 words** — very short, low signal
- `face_emotion_hint` is **missing for 10.25%** of training rows
- **232 out of 1200** training entries (19%) are flagged as very short texts
- **65 entries** show signal conflict (happy face + high stress)
- `reflection_quality = conflicted` indicates the user contradicts themselves
- Intensity labels are subjective ordinal judgments, not measured values

A system that only optimizes accuracy and ignores these realities would fail in production.

---

## Feature Engineering

### Text Features
Raw journal text is cleaned and vectorized using TF-IDF:
- Lowercased, punctuation removed
- Stopwords removed except negation words (`not`, `no`, `never`) — these flip emotional meaning
- TF-IDF with 500 features, unigrams + bigrams (`ngram_range=(1,2)`)
- Bigrams like `"racing thoughts"`, `"not calm"`, `"felt settled"` carry meaning single words miss

### Metadata Features
Categorical columns (`ambience_type`, `time_of_day`, `previous_day_mood`, `face_emotion_hint`, `reflection_quality`) are label-encoded.

Numerical columns used directly: `duration_min`, `sleep_hours`, `energy_level`, `stress_level`.

### Engineered Features
| Feature | Description |
|---|---|
| `word_count` | Number of words in journal entry |
| `char_count` | Character count |
| `is_short` | 1 if word_count <= 5 (uncertain prediction likely) |
| `signal_conflict` | 1 if happy_face + stress >= 4 (contradiction) |
| `was_negative_yesterday` | 1 if previous_day_mood was overwhelmed/restless/mixed |
| `stress_energy_gap` | stress_level - energy_level (burnout signal) |

### Missing Value Strategy
- `sleep_hours` → filled with median (robust to outliers)
- `previous_day_mood` → filled with `'unknown'` (model learns missingness as a category)
- `face_emotion_hint` → filled with `'none'` (treated as absent signal)

---

## Model Choice

### Emotional State — RandomForestClassifier + CalibratedClassifierCV
- 6-class classification: calm, focused, mixed, neutral, overwhelmed, restless
- RandomForest chosen for robustness to short noisy text and mixed signal types
- CalibratedClassifierCV (sigmoid, cv=5) wraps it to produce reliable probability scores
- **Validation F1 (macro): 0.64 | Accuracy: 63%**

### Intensity — RandomForestClassifier + CalibratedClassifierCV
- Treated as **5-class classification** (not regression)
- Intensity labels 1–5 are subjective ordinal judgments — not mathematically equidistant
- `class_weight='balanced'`, `cv=3`, `method='isotonic'`
- **Validation F1 (macro): 0.21**
- Low F1 expected — intensity is genuinely hard to infer from short vague text

---

## Ablation Study Results

| Configuration | State F1 | Intensity F1 |
|---|---|---|
| text only | 0.598 | 0.220 |
| metadata only | 0.170 | 0.172 |
| text + metadata | 0.570 | 0.200 |

**Key finding:** Text-only slightly outperforms the combined model for state prediction. This is because label-encoded categorical metadata without feature scaling can introduce noise into tree splits with only 1200 training samples. Metadata alone is weak (0.170), confirming text carries the primary signal. Future work: apply StandardScaler to numerical features before combining.

---

## Feature Importance (Top findings)

The top features for emotional state prediction were emotional vocabulary words:
`nothing`, `tasks`, `drained`, `lighter`, `organized`, `calmer`, `quiet`

- **Text contributes 95.8%** of total feature importance
- **Metadata contributes 4.2%**
- `char_count` appeared in top 10 — confirming short texts produce weaker predictions
- `stress_level` and `energy_level` were the most important metadata features

---

## Decision Engine

Rule-based logic maps predicted outputs to actionable recommendations.

### What to do
| State | Condition | Action |
|---|---|---|
| overwhelmed / restless | intensity >= 4 | box_breathing |
| calm | energy >= 3 | deep_work |
| calm | energy < 3 | light_planning |
| focused | energy >= 3 | deep_work |
| neutral | stress >= 3 | grounding |
| mixed | stress >= 4 | grounding |
| any | energy <= 1 | rest (override) |

### When to do it
Timing is based on time of day + urgency:
- `now` — urgent negative state (overwhelmed/restless + intensity >= 4)
- `within_15_min` — morning or afternoon, moderate state
- `later_today` — afternoon, positive state
- `tonight` — evening, calm/focused
- `tomorrow_morning` — night, non-urgent

---

## Uncertainty Modeling

A prediction is flagged `uncertain_flag = 1` when **any** of these are true:
1. Model confidence (max class probability) < 0.50
2. Text is very short (`is_short = 1`, word_count <= 5)
3. Face emotion contradicts stress level (`signal_conflict = 1`)
4. Reflection quality is `'conflicted'`

### Test set results
- **75.8% of predictions flagged as uncertain**
- Average confidence: 0.451
- This is honest — test journal texts are short and ambiguous

A high uncertain rate is not a failure. It means the system knows its own limits and will avoid overconfident recommendations on weak inputs.

---

## Robustness

| Scenario | Handling |
|---|---|
| Very short text ("ok", "fine") | `is_short=1` → uncertain_flag=1, fallback to `grounding` |
| Missing `face_emotion_hint` | Filled with `'none'` — treated as absent signal |
| Missing `sleep_hours` | Filled with training median |
| Missing `previous_day_mood` | Filled with `'unknown'` — model learns this category |
| Conflicting signals | `signal_conflict` feature + uncertain_flag triggered |
| Unseen category in test | LabelEncoder returns -1 gracefully |

---

## Output Format

`outputs/predictions.csv` contains 120 rows with these columns:

| Column | Type | Description |
|---|---|---|
| id | int | Test sample ID |
| predicted_state | string | One of: calm, focused, mixed, neutral, overwhelmed, restless |
| predicted_intensity | int | 1–5 |
| confidence | float | 0–1, model's certainty |
| uncertain_flag | int | 0 or 1 |
| what_to_do | string | Recommended action |
| when_to_do | string | Timing recommendation |
| supportive_message | string | Human-like explanation |

---

## Known Limitations

- Intensity prediction is weak (F1: 0.21) — short texts provide insufficient signal for fine-grained intensity discrimination
- 75.8% uncertain flag rate on test set — reflects genuine data ambiguity, not a model bug
- TF-IDF does not capture semantic similarity ("exhausted" and "drained" treated as unrelated)
- Label noise in training data — some emotional states are inherently subjective

---

## Evaluation Summary

| Metric | Value |
|---|---|
| Emotional State CV F1 (macro) | 0.534 |
| Emotional State Validation F1 | 0.640 |
| Emotional State Accuracy | 63% |
| Intensity CV F1 (macro) | 0.206 |
| Uncertain predictions (test) | 75.8% |
| Total test predictions | 120 |