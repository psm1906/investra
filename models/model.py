import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import os

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# Globals for reuse after training
_model = None
_encoder = None
_feature_columns = None
_median_values = None
_prob_threshold = None  # training‑set 80th‑percentile cutoff of predicted probability (for scaling to 0–100)


def train_model(
    property_data_path: str = None,
    macro_data_path:    str = None,
    model_out_path:     str = "risk_model.pkl",
    encoder_out_path:   str = "risk_encoder.pkl",
    feature_meta_path:  str = "model_columns.pkl"
):
    """
    Train the LightGBM model on property + macro data to predict a binary "HighRisk" label,
    then we scale the output probabilities to produce a 0–100 risk score at inference time.

    Steps:
      1) Load & merge property data with macro data by year.
      2) Define "HighRisk" = top 20% most expensive homes by SalePrice.
      3) Train-test split (stratified).
      4) Impute numeric medians, one-hot encode categoricals.
      5) Train a LightGBM classifier.
      6) Evaluate on test set (accuracy, AUC, F1).
      7) Determine 80th percentile of the train set’s predicted probabilities (prob_threshold).
      8) Save model, encoder, and feature info to disk for inference.
    """
    # ------------------------------------------------------------------
    # 1) Load housing + macro data
    # ------------------------------------------------------------------
    if property_data_path is None:
        property_data_path = "../data/us_housing/housing.csv"
    if macro_data_path is None:
        macro_data_path = "../data/us_market/Annual_Macroeconomic_Factors.csv"

    prop_df  = pd.read_csv(property_data_path)
    macro_df = pd.read_csv(macro_data_path)

    # ------------------------------------------------------------------
    # 2) Prepare macro factors (merge by year from Date)
    # ------------------------------------------------------------------
    # Convert "Date" -> year
    macro_df["Year"] = pd.to_datetime(macro_df["Date"]).dt.year

    # Rename columns for consistency
    # Make sure these match the exact column names in your CSV
    macro_df = macro_df.rename(columns={
        "House_Price_Index":        "HousePriceIndex",
        "Stock_Price_Index":        "StockPriceIndex",
        "Consumer_Price_Index":     "CPI",
        "Unemployment_Rate":        "UnemploymentRate",
        "Mortgage_Rate":            "MortgageRate",
        "Real_GDP":                 "RealGDP",
        "Real_Disposable_Income":   "RealDisposableIncome",
    })
    # We'll keep *all* relevant macro columns. Adjust as needed.
    macro_cols = [
        "Year", "HousePriceIndex", "StockPriceIndex", "CPI",
        "Population", "UnemploymentRate", "RealGDP",
        "MortgageRate", "RealDisposableIncome"
    ]
    # Drop duplicates just in case
    macro_df = macro_df[macro_cols].drop_duplicates()

    # Merge on YrSold = Year
    data = prop_df.merge(
        macro_df,
        how="left",        # you can switch to 'inner' if you only want matching years
        left_on="YrSold",
        right_on="Year"
    )

    # Drop the extra "Year" column from macro if you wish
    if "Year" in data.columns:
        data.drop(columns=["Year"], inplace=True)

    # Must have SalePrice
    if "SalePrice" not in data.columns:
        raise KeyError("Property data must include 'SalePrice' to define HighRisk.")

    # ------------------------------------------------------------------
    # 3) Define binary HighRisk = top 20% of SalePrice
    # ------------------------------------------------------------------
    price_threshold = data["SalePrice"].quantile(0.80)
    data["HighRisk"] = (data["SalePrice"] >= price_threshold).astype(int)

    # ------------------------------------------------------------------
    # 4) Train-test split (stratified on HighRisk)
    # ------------------------------------------------------------------
    train_df, test_df = train_test_split(
        data,
        test_size=0.20,
        random_state=42,
        stratify=data["HighRisk"]
    )

    y_train = train_df["HighRisk"].values
    y_test  = test_df["HighRisk"].values

    # We do NOT feed 'SalePrice' or 'HighRisk' into the model
    X_train = train_df.drop(columns=["SalePrice", "HighRisk"], errors="ignore")
    X_test  = test_df.drop(columns=["SalePrice", "HighRisk"], errors="ignore")

    # Also drop 'Id' if present
    for col in ["Id", "index"]:
        if col in X_train.columns:
            X_train.drop(columns=[col], inplace=True)
        if col in X_test.columns:
            X_test.drop(columns=[col], inplace=True)

    # ------------------------------------------------------------------
    # 5) Impute numeric columns with median
    # ------------------------------------------------------------------
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    global _median_values
    _median_values = {col: X_train[col].median() for col in numeric_cols}

    X_train[numeric_cols] = X_train[numeric_cols].fillna(
        pd.Series(_median_values)
    )
    X_test[numeric_cols] = X_test[numeric_cols].fillna(
        pd.Series(_median_values)
    )

    # ------------------------------------------------------------------
    # 6) One-hot encode all object/categorical columns
    # ------------------------------------------------------------------
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    # Fill missing in categoricals with "Missing"
    X_train[cat_cols] = X_train[cat_cols].fillna("Missing")
    X_test[cat_cols]  = X_test[cat_cols].fillna("Missing")

    encoder = None
    if cat_cols:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
        encoder.fit(X_train[cat_cols])

        X_cat_train = encoder.transform(X_train[cat_cols])
        X_cat_test  = encoder.transform(X_test[cat_cols])

        cat_col_names = encoder.get_feature_names_out(cat_cols)
        X_cat_train = pd.DataFrame(X_cat_train, columns=cat_col_names, index=X_train.index)
        X_cat_test  = pd.DataFrame(X_cat_test, columns=cat_col_names, index=X_test.index)
    else:
        X_cat_train = pd.DataFrame(index=X_train.index)
        X_cat_test  = pd.DataFrame(index=X_test.index)

    # Combine numeric + encoded
    X_train_num = X_train.drop(columns=cat_cols, errors="ignore")
    X_test_num  = X_test.drop(columns=cat_cols, errors="ignore")

    X_full_train = pd.concat([X_train_num, X_cat_train], axis=1)
    X_full_test  = pd.concat([X_test_num,  X_cat_test],  axis=1)

    # ------------------------------------------------------------------
    # 7) Train LightGBM Classifier
    # ------------------------------------------------------------------
    model = lgb.LGBMClassifier(
        objective="binary",
        random_state=42,
        num_leaves=64
        # add hyperparams as needed
    )
    model.fit(X_full_train, y_train)

    # ------------------------------------------------------------------
    # 8) Evaluate performance on test
    # ------------------------------------------------------------------
    test_probs = model.predict_proba(X_full_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, test_preds)
    auc = roc_auc_score(y_test, test_probs)
    f1  = f1_score(y_test, test_preds)

    print("=== Test Set Performance ===")
    print(f"Accuracy: {acc:.3f}")
    print(f"ROC AUC:  {auc:.3f}")
    print(f"F1 Score: {f1:.3f}")

    # ------------------------------------------------------------------
    # 9) Determine the 80th percentile of the train set’s predicted probabilities
    # ------------------------------------------------------------------
    train_probs = model.predict_proba(X_full_train)[:, 1]
    global _prob_threshold
    _prob_threshold = np.quantile(train_probs, 0.80)

    # ------------------------------------------------------------------
    # 10) Save artifacts
    # ------------------------------------------------------------------
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, os.path.join("models", model_out_path))

    if encoder:
        joblib.dump(encoder, os.path.join("models", encoder_out_path))

    feature_info = {
        "columns":        X_full_train.columns.tolist(),
        "median_values":  _median_values,
        "cat_cols":       cat_cols,
        "prob_threshold": _prob_threshold
    }
    joblib.dump(feature_info, os.path.join("models", feature_meta_path))

    # Store globally for inference
    global _model, _encoder, _feature_columns
    _model           = model
    _encoder         = encoder
    _feature_columns = X_full_train.columns.tolist()

    print("Model training completed. Artifacts saved to ./models/")
    return model


def predict_risk_score(df_input: pd.DataFrame, macro_data_path: str = None) -> pd.Series:
    """
    Given new property rows in df_input, return a 0–100 risk score
    by applying the previously trained model & transformations.
    """
    global _model, _encoder, _feature_columns, _median_values, _prob_threshold

    # Load model from disk if not in memory
    if _model is None:
        _model = joblib.load(os.path.join("models", "risk_model.pkl"))

    feat_info = joblib.load(os.path.join("models", "model_columns.pkl"))
    if _encoder is None and feat_info["cat_cols"]:
        _encoder = joblib.load(os.path.join("models", "risk_encoder.pkl"))

    _median_values  = feat_info["median_values"]
    cat_cols        = feat_info["cat_cols"]
    _prob_threshold = feat_info["prob_threshold"]
    if _feature_columns is None:
        _feature_columns = feat_info["columns"]

    df = df_input.copy()

    # If the user didn't provide macro_data_path, default to the same used in training
    if macro_data_path is None:
        macro_data_path = "../data/us_market/Annual_Macroeconomic_Factors.csv"

    # Merge macro if YrSold is present
    if "YrSold" in df.columns:
        macro_df = pd.read_csv(macro_data_path)
        macro_df["Year"] = pd.to_datetime(macro_df["Date"]).dt.year

        macro_df = macro_df.rename(columns={
            "House_Price_Index":        "HousePriceIndex",
            "Stock_Price_Index":        "StockPriceIndex",
            "Consumer_Price_Index":     "CPI",
            "Unemployment_Rate":        "UnemploymentRate",
            "Mortgage_Rate":            "MortgageRate",
            "Real_GDP":                 "RealGDP",
            "Real_Disposable_Income":   "RealDisposableIncome",
        })

        macro_cols = [
            "Year", "HousePriceIndex", "StockPriceIndex", "CPI",
            "Population", "UnemploymentRate", "RealGDP",
            "MortgageRate", "RealDisposableIncome"
        ]
        macro_df = macro_df[macro_cols].drop_duplicates()

        df = df.merge(macro_df, how="left", left_on="YrSold", right_on="Year")
        for col in ["Year", "Date"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True, errors="ignore")

    # Impute numeric
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in _median_values:
            df[col] = df[col].fillna(_median_values[col])
        else:
            df[col] = df[col].fillna(df[col].median())

    # Impute/encode categorical
    for c in cat_cols:
        if c not in df.columns:
            df[c] = "Missing"
        df[c] = df[c].fillna("Missing")

    if _encoder:
        arr = _encoder.transform(df[cat_cols])
        cat_names = _encoder.get_feature_names_out(cat_cols)
        df_cat_encoded = pd.DataFrame(arr, columns=cat_names, index=df.index)
    else:
        df_cat_encoded = pd.DataFrame(index=df.index)

    df_num = df.drop(columns=cat_cols, errors="ignore")
    df_full = pd.concat([df_num, df_cat_encoded], axis=1)

    # Reindex to match training columns
    df_full = df_full.reindex(columns=_feature_columns, fill_value=0)

    # Predict probabilities of HighRisk
    probs = _model.predict_proba(df_full)[:, 1]

    # Scale probabilities to 0–100 based on the training 80th percentile
    if _prob_threshold and _prob_threshold > 0:
        scaled = probs / _prob_threshold
    else:
        scaled = probs

    scaled = np.clip(scaled, 0.0, 1.0)  # clamp 0..1
    risk_scores = scaled * 100.0
    return pd.Series(risk_scores, index=df.index, name="risk_score")


def predict_risk_score_from_ui(user_input: dict) -> float:
    """
    Example of how you might map high-level UI fields (Bedrooms, Condition, etc.)
    to the columns your model expects, then call predict_risk_score on a single row.
    """
    row_data = {}

    row_data["YrSold"] = user_input.get("YrSold", 2010)

    if "SqFt" in user_input:
        row_data["GrLivArea"] = user_input["SqFt"]
    if "Bedrooms" in user_input:
        row_data["BedroomAbvGr"] = user_input["Bedrooms"]
    if "Bathrooms" in user_input:
        row_data["FullBath"] = user_input["Bathrooms"]
    if "YearBuilt" in user_input:
        row_data["YearBuilt"] = user_input["YearBuilt"]

    cond_map = {"Excellent": 9, "Good": 7, "Fair": 5, "Poor": 3}
    if "Condition" in user_input:
        row_data["OverallQual"] = cond_map.get(user_input["Condition"], 5)

    if "Neighborhood" in user_input:
        row_data["Neighborhood"] = user_input["Neighborhood"]
    else:
        row_data["Neighborhood"] = "NAmes"

    # If user says SingleFamily, Townhouse, etc. → MSZoning
    type_map = {"SingleFamily": "RL", "Townhouse": "RM", "Condo": "RM"}
    if "PropertyType" in user_input:
        row_data["MSZoning"] = type_map.get(user_input["PropertyType"], "RL")
    else:
        row_data["MSZoning"] = "RL"

    # Optionally override macro
    if "MortgageRate" in user_input:
        row_data["MortgageRate"] = user_input["MortgageRate"]
    if "UnemploymentRate" in user_input:
        row_data["UnemploymentRate"] = user_input["UnemploymentRate"]
    if "CPI" in user_input:
        row_data["CPI"] = user_input["CPI"]

    # Build a single-row DataFrame
    df = pd.DataFrame([row_data])
    score = predict_risk_score(df)
    return float(score.iloc[0])


# ----------------------------------------------------------------------
# If run as script, train the model and do a short synthetic test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("==== Training the RiskRadar LightGBM model ====")
    train_model(
        property_data_path="../data/us_housing/housing.csv",     # put your actual housing.csv path
        macro_data_path="../data/us_market/Annual_Macroeconomic_Factors.csv"  # your actual macro CSV path
    )

    print("\n==== Quick Synthetic Testing ====")
    df_test = pd.DataFrame([
        {
            "YrSold":       2008,
            "MSZoning":     "RL",
            "OverallQual":  5,
            "GrLivArea":    1400,
            "Neighborhood": "NAmes"
        },
        {
            "YrSold":       2009,
            "MSZoning":     "FV",
            "OverallQual":  9,
            "GrLivArea":    3000,
            "Neighborhood": "StoneBr"
        }
    ])

    scores = predict_risk_score(df_test)
    for idx, row in df_test.iterrows():
        print(f"Row {idx} => {dict(row)} | RiskScore={scores[idx]:.1f}")