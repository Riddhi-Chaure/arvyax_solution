# ERROR_ANALYSIS.md
## ArvyaX Emotional Wellness Prediction System

---

## Overview

This document analyzes failure cases observed during model validation and prediction.
Errors are drawn from the 20% validation split (240 samples) and known data patterns
identified during exploration.

The model achieved 63% accuracy on emotional state prediction. The remaining 37%
of errors are analyzed below across 10 distinct failure types.

---

## How Errors Were Identified

```python
# Validation split analysis
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_state, test_size=0.2, random_state=42, stratify=y_state
)
clf_state.fit(X_tr, y_tr)
y_pred = clf_state.predict(X_val)

val_df = train.iloc[val_indices].copy()
val_df['pred_state'] = le_state.inverse_transform(y_pred)
errors = val_df[val_df['emotional_state'] != val_df['pred_state']]
```

---

## Confusion Matrix (Validation Set — 240 samples)

```
             calm  focused  mixed  neutral  overwhelmed  restless
calm           28        5      3        1            3         3
focused         4       26      3        0            1         5
mixed           5        7     23        0            2         1
neutral         4        6      1       23            1         5
overwhelmed     0        4      1        1           26         6
restless        5        7      1        0            3        26
```

Key observations:
- `focused` is most confused with `restless` (5 cases) and `calm` (4 cases)
- `neutral` leaks into `focused` (6 cases) and `restless` (5 cases)
- `mixed` is absorbed into `focused` (7 cases) — mixed has no strong vocabulary

---

## Failure Case 1 — Very Short Text

**Type:** Insufficient input signal

**Example:**
```
journal_text    : "kinda calm ..."
emotional_state : calm
predicted_state : neutral
word_count      : 2
is_short        : 1
```

**What went wrong:**
TF-IDF produces a near-zero vector for 2-word inputs. The model falls back almost
entirely on metadata, which is not strong enough to distinguish calm from neutral
(both have low stress, moderate energy). The word "kinda" adds hedging but TF-IDF
doesn't capture sentiment modifiers well.

**Why it matters:**
19% of training entries (232 rows) are flagged as short texts. This is a systemic
failure pattern, not an edge case.

**How to improve:**
- Route short texts (word_count <= 5) to a separate fallback classifier trained
  only on metadata
- Add sentiment polarity score as a feature (positive/negative/neutral float)
- Flag these with uncertain_flag=1 and recommend conservative actions

---

## Failure Case 2 — Contradictory Signals

**Type:** Face emotion vs stress contradiction

**Example:**
```
journal_text      : "felt okay after the session i guess"
face_emotion_hint : happy_face
stress_level      : 5
emotional_state   : overwhelmed
predicted_state   : calm
signal_conflict   : 1
```

**What went wrong:**
The model weighted `happy_face` and the mildly positive text ("felt okay") over the
high stress metadata. This is a known failure — 65 such contradictions exist in
training data. The model learned that happy_face correlates with calm/focused, and
that pattern overrode the stress signal here.

**Why it matters:**
In wellness systems, acting on the wrong signal has real consequences. Telling an
overwhelmed user to do deep work because they wrote "okay" is actively harmful.

**How to improve:**
- When `signal_conflict=1`, override predicted action to a neutral safe option
  (grounding or box_breathing) regardless of predicted state
- Weight stress_level more heavily than face_emotion_hint in decision engine
- Train a separate conflict-aware model on the 65 conflict rows

---

## Failure Case 3 — "Mixed" Absorbed into Other Classes

**Type:** Weak class vocabulary

**Example:**
```
journal_text    : "the session was interesting. not sure how i feel about it."
emotional_state : mixed
predicted_state : focused
reflection_quality : conflicted
```

**What went wrong:**
"Mixed" has no strong vocabulary of its own. Words like "interesting", "not sure"
appear across multiple emotional states. The model sees uncertainty markers and
maps them to focused (which also has hedging language in some entries). Mixed had
the lowest recall in the confusion matrix.

**Why it matters:**
Mixed is arguably the most important state to detect accurately — users in a mixed
state need a specific type of guidance (grounding, journaling) rather than action.

**How to improve:**
- Add explicit mixed-state markers: "not sure", "conflicted", "both", "idk", "unclear"
  as high-weight features
- Consider merging mixed with neutral for training, then splitting via rule-based logic
  using reflection_quality == 'conflicted'

---

## Failure Case 4 — Neutral vs Focused Confusion

**Type:** Near-neighbor class overlap

**Example:**
```
journal_text    : "things felt clearer after the session. ready to get back to it."
emotional_state : neutral
predicted_state : focused
stress_level    : 2
energy_level    : 3
```

**What went wrong:**
"Clearer" and "ready" are strong focused-state words in the TF-IDF vocabulary.
The label "neutral" was assigned by an annotator seeing moderate scores, but the
text itself reads as focused. This is a labeling ambiguity, not a model error.

**Why it matters:**
Neutral and focused have nearly identical recommended actions (journaling vs
deep_work). The decision engine outcome is similar, so real-world impact is low.
However, it inflates error rates unfairly.

**How to improve:**
- Apply label smoothing during training to reduce penalty for near-neighbor errors
- In decision engine, treat neutral + focused similarly when energy >= 3
- Consider a 5-class problem by merging neutral into focused with low intensity

---

## Failure Case 5 — Night-Time Fatigue Distortion

**Type:** Time-of-day bias in writing quality

**Example:**
```
journal_text    : "tired. session was fine. going to sleep."
time_of_day     : night
emotional_state : calm
predicted_state : restless
sleep_hours     : 4.5
stress_level    : 3
```

**What went wrong:**
Short, fragmented night-time writing patterns are misread as restless. The model
learned that fragmented syntax + low word_count often correlates with restless
entries. But at night, it also correlates with simple tiredness and calm.

**Why it matters:**
Night entries make up a meaningful portion of the dataset. Misclassifying
calm-but-tired as restless would wrongly recommend box_breathing when rest is correct.

**How to improve:**
- Add interaction feature: is_short AND time_of_day == 'night' → high probability of calm
- Reduce weight of syntax-based features for night entries
- Separate night-time model or time-of-day stratified training

---

## Failure Case 6 — Positive Language Masking Anxiety

**Type:** Surface tone vs underlying state mismatch

**Example:**
```
journal_text    : "things are moving forward. just have a lot on my plate right now."
emotional_state : overwhelmed
predicted_state : focused
stress_level    : 4
energy_level    : 2
```

**What went wrong:**
"Moving forward" is a strongly positive phrase that TF-IDF maps to focused/calm.
The second clause ("a lot on my plate") signals overwhelm, but the positive opening
phrase dominates the vector representation.

**Why it matters:**
Users often mask negative emotions with positive framing — a well-known phenomenon
in mental health contexts. A system that takes surface tone at face value misses
these users.

**How to improve:**
- Add contrast-detection: sentences containing "but", "yet", "however", "just" after
  positive openers should reduce confidence in positive state prediction
- Add a stress-override rule: if stress >= 4 AND energy <= 2, cap confidence on
  positive states (calm/focused) regardless of text tone

---

## Failure Case 7 — Reflection Quality Ignored by Model

**Type:** Available signal not fully utilized

**Example:**
```
journal_text       : "felt good but also kind of on edge. hard to say really."
reflection_quality : conflicted
emotional_state    : restless
predicted_state    : mixed
```

**What went wrong:**
This is actually a near-correct prediction (restless vs mixed are close), but the
deeper issue is that reflection_quality='conflicted' should have triggered maximum
uncertainty. The model's confidence was 0.38 — below threshold — so uncertain_flag
was correctly set to 1. However, the action recommended was still 'grounding' when
'box_breathing' may have been more appropriate for the actual restless state.

**Why it matters:**
conflicted reflection quality is a strong meta-signal that the label itself may be
wrong. The model should treat these rows as inherently unreliable.

**How to improve:**
- For reflection_quality='conflicted', force uncertain_flag=1 regardless of confidence
  (already partially implemented)
- In decision engine, when uncertain_flag=1 AND reflection_quality='conflicted',
  default to the safest action: grounding or box_breathing

---

## Failure Case 8 — Previous Mood Recovery Not Modeled

**Type:** Missing temporal pattern

**Example:**
```
journal_text       : "feeling a bit better today actually"
previous_day_mood  : overwhelmed
emotional_state    : calm
predicted_state    : neutral
```

**What went wrong:**
The phrase "feeling a bit better" is a recovery signal — it only makes sense in the
context of having been worse. The model has no way to interpret "better" as calm
because it doesn't know the baseline. Without the previous_day_mood context being
directly linked to the text interpretation, the recovery signal is lost.

**Why it matters:**
Recovery states are important to identify correctly — a user moving from overwhelmed
to calm needs reinforcement and light activity, not journaling or grounding.

**How to improve:**
- Add a 'mood_delta' feature: map previous_day_mood and predicted state to a numeric
  scale, compute the difference
- Add binary feature: 'recovery_language' = 1 if text contains "better", "improved",
  "more settled", "clearer than yesterday"

---

## Failure Case 9 — Domain Slang and Informal Language

**Type:** Out-of-vocabulary informal expressions

**Example:**
```
journal_text    : "was totally vibing with the rain sounds ngl"
emotional_state : calm
predicted_state : focused
word_count      : 9
```

**What went wrong:**
"Vibing" and "ngl" (not gonna lie) are informal expressions not well-represented
in the TF-IDF vocabulary trained on this small corpus. TF-IDF treats "vibing" as
a rare token with low weight. The model falls back on "rain" (ambience) and
metadata, which predicts focused due to moderate energy and low stress.

**Why it matters:**
Younger users or casual writers use informal language frequently. A model that
fails on informal text will systematically misclassify a specific user demographic.

**How to improve:**
- Add a small informal-to-formal mapping: {"vibing": "calm", "ngl": "", "lowkey": "somewhat"}
- Add sentiment lexicon features (VADER or custom wellness lexicon) as additional
  numeric features alongside TF-IDF

---

## Failure Case 10 — Intensity Collapse to Class 4

**Type:** Systematic model bias

**Example:**
```
journal_text         : "still feeling a bit off. not great not terrible."
true_intensity       : 2
predicted_intensity  : 4
confidence_intensity : low
```

**What went wrong:**
The intensity classifier predicted class 4 for 72 out of 120 test samples (60%).
This is a systematic bias toward the majority prediction rather than true
discrimination between intensity levels. The model learned that predicting 4 is
"safe" because intensity labels in training are noisy and subjective.

**Why it matters:**
Intensity drives urgency in the decision engine. Overestimating intensity as 4
causes the system to recommend immediate actions (box_breathing, now) when a
lighter touch (journaling, later_today) would be more appropriate.

**How to improve:**
- Increase `min_samples_leaf` to force more splits in low-intensity regions
- Use ordinal regression instead of flat classification to respect the 1-5 ordering
- Add more text features specifically correlated with extremes:
  intensity 1 = "slightly", "barely", "a little"
  intensity 5 = "extremely", "completely", "overwhelmingly"
- Consider treating intensity as a regression target with MAE loss

---

## Summary Table

| # | Error Type | Frequency | Impact | Fix Priority |
|---|---|---|---|---|
| 1 | Very short text | High (19% of data) | High | P1 |
| 2 | Contradictory signals | Medium (65 rows) | High | P1 |
| 3 | Mixed class absorption | Medium | Medium | P2 |
| 4 | Neutral vs focused confusion | Medium | Low | P3 |
| 5 | Night-time fatigue distortion | Medium | Medium | P2 |
| 6 | Positive language masking anxiety | Low-Medium | High | P1 |
| 7 | Reflection quality underutilized | Low | Medium | P2 |
| 8 | Recovery pattern not modeled | Low | Medium | P3 |
| 9 | Informal language OOV | Low | Low | P3 |
| 10 | Intensity collapse to class 4 | High (60% test) | Medium | P1 |

---

## Top 3 Improvements for Next Version

**1. Stress-energy override in decision engine**
When stress >= 4 AND energy <= 2, force action = box_breathing or rest regardless
of predicted state. This handles cases 2 and 6 without retraining.

**2. Short-text fallback classifier**
Train a metadata-only RandomForest specifically on short-text rows (word_count <= 5).
Route all short inputs there instead of the main model.

**3. Intensity as ordinal regression**
Replace the flat 5-class intensity classifier with an ordinal regression model
(e.g. mord library or threshold-based approach). This respects the natural ordering
of intensity levels and reduces the collapse-to-4 behavior.