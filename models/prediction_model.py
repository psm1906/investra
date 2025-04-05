# models/prediction_model.py

import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder
import os

# Globals for in‑memory reuse
_model = None
_encoder = None
_feature_columns = None
_median_values = None

def train_model(
    property_data_path: str = "../data/ames/AmesHousing.csv",
    macro_data_path:    str = "../data/us_market/macro.csv",
    model_out_path:     str = "risk_model.pkl",
    encoder_out_path:   str = "risk_encoder.pkl",
    feature_meta_path:  str = "model_columns.pkl"
) -> lgb.LGBMClassifier:
    """
    Train the LightGBM risk model and save artifacts under models/.
    """
    # 1) Load
    prop_df  = pd.read_csv(property_data_path)
    macro_df = pd.read_csv(macro_data_path)

    # 2) Merge on YrSold → Year
    data = prop_df.merge(macro_df, how="left", left_on="YrSold", right_on="Year")
    if "Year" in data: data.drop("Year", axis=1, inplace=True)

    # 3) Create binary target: top 20% SalePrice = HighRisk
    if "SalePrice" not in data:
        raise KeyError("`SalePrice` missing from property data")
    thresh = data["SalePrice"].quantile(0.80)
    data["HighRisk"] = (data["SalePrice"] >= thresh).astype(int)

    # 4) Features X / target y
    X = data.drop(columns=["SalePrice", "HighRisk"], errors="ignore")
    # drop any ID-like columns
    X = X.loc[:, ~X.columns.str.lower().isin(["id", "idx", "index"])]
    y = data["HighRisk"]

    # 5) Impute missing
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    global _median_values
    _median_values = {c: X[c].median() for c in num_cols}
    for c in num_cols:
        X[c].fillna(_median_values[c], inplace=True)
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cat_cols:
        X[c].fillna("Missing", inplace=True)

    # 6) One‑hot encode
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    X_cat = pd.DataFrame()
    if cat_cols:
        encoder.fit(X[cat_cols])
        arr = encoder.transform(X[cat_cols])
        names = encoder.get_feature_names_out(cat_cols)
        X_cat = pd.DataFrame(arr, columns=names, index=X.index)
    X_num = X.drop(columns=cat_cols, errors="ignore")
    if "YrSold" in X_num: X_num.drop("YrSold", axis=1, inplace=True)
    X_full = pd.concat([X_num, X_cat], axis=1)

    # 7) Train LightGBM
    model = lgb.LGBMClassifier(objective="binary", random_state=42)
    model.fit(X_full, y)

    # 8) Save artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(model,       os.path.join("models", model_out_path))
    joblib.dump(encoder,     os.path.join("models", encoder_out_path))
    feature_info = {
        "columns":          X_full.columns.tolist(),
        "median_values":    _median_values,
        "categorical_cols": cat_cols
    }
    joblib.dump(feature_info, os.path.join("models", feature_meta_path))

    # Cache in memory
    global _model, _encoder, _feature_columns
    _model           = model
    _encoder         = encoder if cat_cols else None
    _feature_columns = X_full.columns.tolist()

    return model


def predict_risk_score(
    input_df: pd.DataFrame,
    macro_data_path: str = "../data/us_market/macro.csv"
) -> pd.Series:
    """
    Given new property rows in a DataFrame, return a 0–100 risk score.
    """
    global _model, _encoder, _feature_columns, _median_values

    # 1) Load model & metadata if not already in memory
    if _model is None:
        _model = joblib.load("models/risk_model.pkl")
    feat_info = joblib.load("models/model_columns.pkl")
    cols       = feat_info["columns"]
    medians    = feat_info["median_values"]
    cat_cols   = feat_info["categorical_cols"]
    if _encoder is None and cat_cols:
        _encoder = joblib.load("models/risk_encoder.pkl")

    # 2) Merge macro if requested
    df = input_df.copy()
    if macro_data_path:
        macro_df = pd.read_csv(macro_data_path)
        df = df.merge(macro_df, how="left", left_on="YrSold", right_on="Year")
        if "Year" in df: df.drop("Year", axis=1, inplace=True)

    # 3) Ensure all expected columns exist
    for c in cols:
        if c not in df:
            df[c] = np.nan

    # 4) Impute missing
    for c, m in medians.items():
        if c in df:
            df[c].fillna(m, inplace=True)
    for c in cat_cols:
        if c in df:
            df[c].fillna("Missing", inplace=True)

    # 5) One‑hot encode & assemble feature matrix
    df_cat = pd.DataFrame()
    if cat_cols:
        arr  = _encoder.transform(df[cat_cols])
        names= _encoder.get_feature_names_out(cat_cols)
        df_cat = pd.DataFrame(arr, columns=names, index=df.index)
    df_num = df.drop(columns=cat_cols + (["YrSold"] if "YrSold" in df else []), errors="ignore")
    df_full= pd.concat([df_num, df_cat], axis=1)
    df_full= df_full.reindex(columns=cols, fill_value=0)

    # 6) Predict probabilities → scale to 0–100
    probs = _model.predict_proba(df_full)[:,1]
    return pd.Series(probs * 100, index=df_full.index, name="risk_score")


if __name__ == "__main__":
    # When run directly, train the model
    print("Training RiskRadar model…")
    train_model(
        property_data_path="../data/ames/AmesHousing.csv",
        macro_data_path   ="../data/us_market/macro.csv",
        model_out_path   ="risk_model.pkl",
        encoder_out_path ="risk_encoder.pkl",
        feature_meta_path="model_columns.pkl"
    )
    print("✅ Done. Artifacts saved under models/")  