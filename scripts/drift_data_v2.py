"""drift_data_v2.py: datasets of plan_v2.md, from SDB or OpenML, in SDB's conventions.

SDB's loader reads its own files. Here OpenML frames get the same treatment: columns that parse as numbers more
than 90 % of the time are numeric, the rest are strings, label 1 is the minority class (the runner's
convention), all-missing columns are dropped, rows that repeat features and label are dropped before anything else,
and a file above the cap is a uniform seeded subsample (SDB's rule, seed 0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.drift_windows import minority_is_one
from sdb import load_data_df

CAP = 10_000
SEED = 0


def prepare(df: pd.DataFrame, target: str, cap: int | None = CAP, stratified: bool = False,
            seed: int = SEED) -> tuple[pd.DataFrame, np.ndarray, list[str], dict]:
    """(X, y, categorical columns, counts) from a raw frame whose column `target` holds two labels.
    cap is the row limit after deduplication; stratified draws the cap class by class (Study C)."""
    df = df.dropna(axis=1, how="all")
    X = df.drop(columns=[target]).copy()
    for c in X:
        if X[c].dtype.name == "category":
            X[c] = X[c].astype(object)
        if X[c].dtype == object:
            num = pd.to_numeric(X[c], errors="coerce")
            if num.notna().mean() > 0.9:
                X[c] = num
        elif X[c].dtype == bool:
            X[c] = X[c].astype(float)
    labels = sorted(df[target].astype(str).unique())
    if len(labels) != 2:
        raise NotBinary(f"{len(labels)} classes")
    y = df[target].astype(str).map({lab: i for i, lab in enumerate(labels)}).to_numpy().astype(int)
    return clean(X, y, X.select_dtypes(include="object").columns.tolist(), cap, stratified, seed)


def clean(X: pd.DataFrame, y: np.ndarray, cat_cols: list[str], cap: int | None, stratified: bool = False,
          seed: int = SEED) -> tuple[pd.DataFrame, np.ndarray, list[str], dict]:
    """Drop rows that repeat features and label, cap the rows, make label 1 the minority class."""
    keep = ~pd.concat([X.astype(str), pd.Series(y, index=X.index, name="__y")], axis=1).duplicated().to_numpy()
    counts = dict(rows_raw=len(X), rows_dedup=int(keep.sum()))
    X, y = X[keep].reset_index(drop=True), y[keep]
    if cap is not None and len(X) > cap:
        rng = np.random.default_rng(seed)
        if stratified:
            idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(y == k), round(cap * (y == k).mean()), replace=False)
                                          for k in (0, 1)]))
        else:
            idx = rng.choice(len(X), cap, replace=False)
        X, y = X.iloc[idx].reset_index(drop=True), y[idx]
    y = minority_is_one(y)
    X[cat_cols] = X[cat_cols].astype(str)
    return X, y, cat_cols, counts | dict(rows=len(X), features=X.shape[1], categorical=len(cat_cols),
                                         minority=float(y.mean()))


def load_openml(did: int, **kw):
    import openml
    ds = openml.datasets.get_dataset(did, download_data=True, download_qualities=False, download_features_meta_data=False)
    df, *_ = ds.get_data(dataset_format="dataframe")
    return prepare(df, ds.default_target_attribute, **kw)


class NotBinary(ValueError):
    pass


def load_sdb(name: str):
    """SDB's frame, columns and types as in every v1 run (its 10,000-row cap comes first), then deduplicated."""
    X, y, cat_cols = load_data_df(name)
    if len(np.unique(y)) != 2:
        raise NotBinary(f"{len(np.unique(y))} classes")
    return clean(X, y, cat_cols, cap=None)
