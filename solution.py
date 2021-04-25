"""Entity matching for product records.

Given two product catalogues (``ltable``/``rtable``) and a set of labelled
example pairs (``train``), predict which cross-catalogue record pairs refer to
the same real-world product. The pipeline is: block candidate pairs by brand,
turn each pair into similarity features, train a Random Forest, and write the
predicted matches to ``output.csv``.
"""

from os.path import join

import numpy as np
import pandas as pd
import Levenshtein as lev
from sklearn.ensemble import RandomForestClassifier


def pairs2LR(ltable, rtable, candset):
    """Join a list of (left_id, right_id) pairs into one side-by-side table.

    Left columns are suffixed ``_l`` and right columns ``_r`` so both records
    in a candidate pair sit on the same row.
    """
    ltable.index = ltable.id
    rtable.index = rtable.id
    pairs = np.array(candset)
    tpls_l = ltable.loc[pairs[:, 0], :]
    tpls_r = rtable.loc[pairs[:, 1], :]
    tpls_l.columns = [col + "_l" for col in tpls_l.columns]
    tpls_r.columns = [col + "_r" for col in tpls_r.columns]
    tpls_l.reset_index(inplace=True, drop=True)
    tpls_r.reset_index(inplace=True, drop=True)
    return pd.concat([tpls_l, tpls_r], axis=1)


def block_by_brand(ltable, rtable):
    """Restrict candidate pairs to records that share the same brand.

    Blocking avoids the quadratic cost of comparing every left record with
    every right record by only pairing those with a matching brand.
    """
    ltable['brand'] = ltable['brand'].astype(str)
    rtable['brand'] = rtable['brand'].astype(str)

    brands = set(ltable["brand"].values).union(set(rtable["brand"].values))

    brand2ids_l = {b.lower(): [] for b in brands}
    brand2ids_r = {b.lower(): [] for b in brands}
    for _, x in ltable.iterrows():
        brand2ids_l[x["brand"].lower()].append(x["id"])
    for _, x in rtable.iterrows():
        brand2ids_r[x["brand"].lower()].append(x["id"])

    candset = []
    for brd in brands:
        for l_id in brand2ids_l[brd]:
            for r_id in brand2ids_r[brd]:
                candset.append([l_id, r_id])
    return candset


def jaccard_similarity(row, attr):
    """Token-set Jaccard similarity between the left and right value of ``attr``."""
    x = set(row[attr + "_l"].lower().split())
    y = set(row[attr + "_r"].lower().split())
    return len(x.intersection(y)) / max(len(x), len(y))


def levenshtein_distance(row, attr):
    """Character-level edit distance between the left and right value of ``attr``."""
    x = row[attr + "_l"].lower()
    y = row[attr + "_r"].lower()
    return lev.distance(x, y)


def feature_engineering(LR):
    """Build a feature matrix of Jaccard + Levenshtein scores per attribute."""
    LR = LR.astype(str)
    attrs = ["title", "category", "brand", "modelno", "price"]
    features = []
    for attr in attrs:
        features.append(LR.apply(jaccard_similarity, attr=attr, axis=1))
        features.append(LR.apply(levenshtein_distance, attr=attr, axis=1))
    return np.array(features).T


def main():
    ltable = pd.read_csv(join('data', "ltable.csv"))
    rtable = pd.read_csv(join('data', "rtable.csv"))
    train = pd.read_csv(join('data', "train.csv"))

    # Block, then featurise the surviving candidate pairs.
    candset = block_by_brand(ltable, rtable)
    print("number of pairs originally", ltable.shape[0] * rtable.shape[0])
    print("number of pairs after blocking", len(candset))
    candset_df = pairs2LR(ltable, rtable, candset)
    candset_features = feature_engineering(candset_df)

    # Featurise the labelled training pairs and fit the classifier.
    training_pairs = list(map(tuple, train[["ltable_id", "rtable_id"]].values))
    training_df = pairs2LR(ltable, rtable, training_pairs)
    training_features = feature_engineering(training_df)
    training_label = train.label.values

    rf = RandomForestClassifier(class_weight="balanced", random_state=0)
    rf.fit(training_features, training_label)
    y_pred = rf.predict(candset_features)

    # Keep predicted matches, dropping any pair already present in training.
    matching_pairs = candset_df.loc[y_pred == 1, ["id_l", "id_r"]]
    matching_pairs = list(map(tuple, matching_pairs.values))

    matching_pairs_in_training = training_df.loc[training_label == 1, ["id_l", "id_r"]]
    matching_pairs_in_training = set(map(tuple, matching_pairs_in_training.values))

    pred_pairs = [pair for pair in matching_pairs
                  if pair not in matching_pairs_in_training]
    pred_df = pd.DataFrame(np.array(pred_pairs), columns=["ltable_id", "rtable_id"])
    pred_df.to_csv("output.csv", index=False)


if __name__ == "__main__":
    main()
