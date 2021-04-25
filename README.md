# Entity Matching (Product Record Linkage)

A machine-learning pipeline that solves the **entity matching** problem: given
two product catalogues, identify which records from each refer to the *same*
real-world product. Built in **Python** with pandas, scikit-learn, and classic
string-similarity features.

Entity matching (a.k.a. record linkage / deduplication) is a core data-cleaning
task — the same product often appears with slightly different titles, brands, or
formatting across sources, and the goal is to reconcile them accurately without
comparing every possible pair.

## Approach

The pipeline follows the standard blocking → featurization → classification
recipe:

1. **Blocking** — comparing every left record against every right record is
   quadratic and wasteful. Candidate pairs are first restricted to records that
   share the same **brand**, dramatically shrinking the search space before any
   expensive comparison.
2. **Feature engineering** — each candidate pair is turned into a numeric
   feature vector using two similarity measures across the key attributes
   (`title`, `category`, `brand`, `modelno`, `price`):
   - **Jaccard similarity** on tokenised text (set overlap)
   - **Levenshtein edit distance** on raw strings
3. **Classification** — a **Random Forest** classifier (`class_weight="balanced"`
   to handle the match/non-match imbalance) is trained on labelled pairs and
   predicts which candidate pairs are true matches.
4. **Output** — predicted matches already present in the training set are
   removed, and the remaining new matches are written to `output.csv`.

## Stack

`pandas` · `numpy` · `scikit-learn` (RandomForestClassifier) ·
`python-Levenshtein`

## Running it

```bash
pip install pandas numpy scikit-learn python-Levenshtein
python solution.py
```

Expects `data/ltable.csv`, `data/rtable.csv`, and `data/train.csv`; produces
`output.csv` with the predicted matching pairs.

## What it demonstrates

Applied machine learning on a real data-integration problem — blocking for
scalability, feature engineering with text-similarity metrics, handling class
imbalance, and end-to-end use of the scikit-learn training/prediction workflow.

> Coursework project for CS 4400 (Introduction to Database Systems).
