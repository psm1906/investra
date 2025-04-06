import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import os

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# Get the absolute path to the directory of this file (i.e. backend/)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# We'll store models in backend/models/
_MODEL_DIR = os.path.join(_SCRIPT_DIR, "models")

# Globals for reuse after training
_model = None
_encoder = None
_feature_columns = None
_median_values = None
_prob_threshold = None  # training-set 80th-percentile cutoff (for scaling 0–100)

def train_model(
    property_data_path: str = None,
    macro_data_path:    str = None
):
    """
    Train a LightGBM model to predict a binary "HighRisk" label (top 20% by SalePrice),
    then store the model + one-hot encoder + feature info to disk under backend/models/.
    """

    global _model, _encoder, _feature_columns, _median_values, _prob_threshold

    # Default CSV paths if none provided
    if property_data_path is None:
        property_data_path = "data/us_housing/housing.csv"
    if macro_data_path is None:
        macro_data_path = "data/us_market/Annual_Macroeconomic_Factors.csv"

    # 1) Load housing + macro data
    prop_df  = pd.read_csv(property_data_path)
    macro_df = pd.read_csv(macro_data_path)

    # 2) Merge macro by year
    macro_df["Year"] = pd.to_datetime(macro_df["Date"]).dt.year
    macro_df = macro_df.rename(columns={
        "House_Price_Index":      "HousePriceIndex",
        "Stock_Price_Index":      "StockPriceIndex",
        "Consumer_Price_Index":   "CPI",
        "Unemployment_Rate":      "UnemploymentRate",
        "Mortgage_Rate":          "MortgageRate",
        "Real_GDP":               "RealGDP",
        "Real_Disposable_Income": "RealDisposableIncome",
    })
    macro_cols = [
        "Year", "HousePriceIndex", "StockPriceIndex", "CPI",
        "Population", "UnemploymentRate", "RealGDP",
        "MortgageRate", "RealDisposableIncome"
    ]
    macro_df = macro_df[macro_cols].drop_duplicates()

    data = prop_df.merge(
        macro_df,
        how="left",
        left_on="YrSold",
        right_on="Year"
    )
    if "Year" in data.columns:
        data.drop(columns=["Year"], inplace=True)

    # Must have SalePrice to define HighRisk
    if "SalePrice" not in data.columns:
        raise KeyError("Property data must include 'SalePrice' to define HighRisk.")

    # 3) HighRisk = top 20% of SalePrice
    price_threshold = data["SalePrice"].quantile(0.80)
    data["HighRisk"] = (data["SalePrice"] >= price_threshold).astype(int)

    # 4) Train/test split (stratified on HighRisk)
    train_df, test_df = train_test_split(
        data,
        test_size=0.20,
        random_state=42,
        stratify=data["HighRisk"]
    )
    y_train = train_df["HighRisk"].values
    y_test  = test_df["HighRisk"].values

    X_train = train_df.drop(columns=["SalePrice", "HighRisk"], errors="ignore")
    X_test  = test_df.drop(columns=["SalePrice", "HighRisk"], errors="ignore")

    for col in ["Id", "index"]:
        if col in X_train.columns:
            X_train.drop(columns=[col], inplace=True)
        if col in X_test.columns:
            X_test.drop(columns=[col], inplace=True)

    # 5) Impute numeric columns with median
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    _median_values = {col: X_train[col].median() for col in numeric_cols}
    X_train[numeric_cols] = X_train[numeric_cols].fillna(pd.Series(_median_values))
    X_test[numeric_cols]  = X_test[numeric_cols].fillna(pd.Series(_median_values))

    # 6) One-hot encode object/categorical columns
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    X_train[cat_cols] = X_train[cat_cols].fillna("Missing")
    X_test[cat_cols]  = X_test[cat_cols].fillna("Missing")

    encoder = None
    if cat_cols:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoder.fit(X_train[cat_cols])
        X_cat_train = encoder.transform(X_train[cat_cols])
        X_cat_test  = encoder.transform(X_test[cat_cols])

        cat_col_names = encoder.get_feature_names_out(cat_cols)
        X_cat_train = pd.DataFrame(X_cat_train, columns=cat_col_names, index=X_train.index)
        X_cat_test  = pd.DataFrame(X_cat_test, columns=cat_col_names, index=X_test.index)
    else:
        X_cat_train = pd.DataFrame(index=X_train.index)
        X_cat_test  = pd.DataFrame(index=X_test.index)

    X_train_num = X_train.drop(columns=cat_cols, errors="ignore")
    X_test_num  = X_test.drop(columns=cat_cols, errors="ignore")

    X_full_train = pd.concat([X_train_num, X_cat_train], axis=1)
    X_full_test  = pd.concat([X_test_num,  X_cat_test],  axis=1)

    # 7) Train LightGBM
    model = lgb.LGBMClassifier(objective="binary", random_state=42, num_leaves=64)
    model.fit(X_full_train, y_train)

    # 8) Evaluate
    test_probs = model.predict_proba(X_full_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, test_preds)
    auc = roc_auc_score(y_test, test_probs)
    f1  = f1_score(y_test, test_preds)
    print("=== Test Set Performance ===")
    print(f"Accuracy: {acc:.3f}")
    print(f"ROC AUC:  {auc:.3f}")
    print(f"F1 Score: {f1:.3f}")

    # 9) 80th percentile cutoff
    train_probs   = model.predict_proba(X_full_train)[:, 1]
    _prob_threshold = np.quantile(train_probs, 0.80)

    # 10) Save artifacts in backend/models/
    # Make sure the folder exists
    os.makedirs(_MODEL_DIR, exist_ok=True)

    # Build absolute file paths for the pickles
    model_out_path     = os.path.join(_MODEL_DIR, "risk_model.pkl")
    encoder_out_path   = os.path.join(_MODEL_DIR, "risk_encoder.pkl")
    feature_meta_path  = os.path.join(_MODEL_DIR, "model_columns.pkl")

    joblib.dump(model, model_out_path)
    if encoder:
        joblib.dump(encoder, encoder_out_path)

    feature_info = {
        "columns":        X_full_train.columns.tolist(),
        "median_values":  _median_values,
        "cat_cols":       cat_cols,
        "prob_threshold": _prob_threshold
    }
    joblib.dump(feature_info, feature_meta_path)

    # Store globally for runtime/inference
    _model           = model
    _encoder         = encoder
    _feature_columns = X_full_train.columns.tolist()

    print("=== Model training completed ===")
    print(f"Model saved to:      {model_out_path}")
    print(f"Encoder saved to:    {encoder_out_path}")
    print(f"Feature info saved:  {feature_meta_path}")

    return model


def predict_risk_score(
    df_input: pd.DataFrame,
    macro_data_path: str = None
) -> pd.Series:
    """
    Given new property rows in df_input, return a 0–100 risk score
    by applying the previously trained model & transformations.
    """

    global _model, _encoder, _feature_columns, _median_values, _prob_threshold

    # If not loaded, load from disk
    if _model is None:
        # Build absolute paths again
        model_path      = os.path.join(_MODEL_DIR, "risk_model.pkl")
        encoder_path    = os.path.join(_MODEL_DIR, "risk_encoder.pkl")
        feat_info_path  = os.path.join(_MODEL_DIR, "model_columns.pkl")

        _model = joblib.load(model_path)
        feat_info = joblib.load(feat_info_path)

        _median_values  = feat_info["median_values"]
        cat_cols        = feat_info["cat_cols"]
        _prob_threshold = feat_info["prob_threshold"]
        _feature_columns = feat_info["columns"]

        # If we actually had an encoder
        if cat_cols:  
            _encoder = joblib.load(encoder_path)

    # If the user didn't provide a macro_data_path, default to the same used in training
    if macro_data_path is None:
        macro_data_path = "data/us_market/Annual_Macroeconomic_Factors.csv"

    df = df_input.copy()

    # Merge macro if YrSold is present
    if "YrSold" in df.columns:
        macro_df = pd.read_csv(macro_data_path)
        macro_df["Year"] = pd.to_datetime(macro_df["Date"]).dt.year
        macro_df = macro_df.rename(columns={
            "House_Price_Index":      "HousePriceIndex",
            "Stock_Price_Index":      "StockPriceIndex",
            "Consumer_Price_Index":   "CPI",
            "Unemployment_Rate":      "UnemploymentRate",
            "Mortgage_Rate":          "MortgageRate",
            "Real_GDP":               "RealGDP",
            "Real_Disposable_Income": "RealDisposableIncome",
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
            # fallback for new numeric columns
            df[col] = df[col].fillna(df[col].median())

    # Impute + encode categoricals
    feat_info = joblib.load(os.path.join(_MODEL_DIR, "model_columns.pkl"))
    cat_cols  = feat_info["cat_cols"]  # same cat cols from training
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

    # Scale probabilities to 0–100 using the 80th percentile from training
    if _prob_threshold and _prob_threshold > 0:
        scaled = probs / _prob_threshold
    else:
        scaled = probs

    scaled = np.clip(scaled, 0.0, 1.0)  # clamp to [0,1]
    return pd.Series(scaled * 100.0, index=df.index, name="risk_score")


def predict_risk_score_from_ui(user_input: dict) -> float:
    """
    Shows how you might map simple UI keys (Bedrooms, Condition, etc.)
    to the columns your model expects, then call `predict_risk_score`
    on a single row.
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

    row_data["Neighborhood"] = user_input.get("Neighborhood", "NAmes")

    # SingleFamily, Townhouse, Condo => typical MSZoning mapping:
    type_map = {"SingleFamily": "RL", "Townhouse": "RM", "Condo": "RM"}
    row_data["MSZoning"] = type_map.get(user_input.get("PropertyType"), "RL")

    # Optionally override macro
    if "MortgageRate" in user_input:
        row_data["MortgageRate"] = user_input["MortgageRate"]
    if "UnemploymentRate" in user_input:
        row_data["UnemploymentRate"] = user_input["UnemploymentRate"]
    if "CPI" in user_input:
        row_data["CPI"] = user_input["CPI"]

    df = pd.DataFrame([row_data])
    score_series = predict_risk_score(df)
    return float(score_series.iloc[0])


# This is the function app.py is importing/calling in /analyze
def analyze_investment_risk_api(clerk_user_id: str, property_details: dict) -> dict:
    """
    Just a thin wrapper around predict_risk_score_from_ui() so we can
    pass in a user ID + property_details and get back a JSON-friendly result.
    """
    risk_score = predict_risk_score_from_ui(property_details)
    return {
        "user_id": clerk_user_id,
        "property_details": property_details,
        "risk_score": risk_score
    }


if __name__ == "__main__":
    print("=== Training the RiskRadar LightGBM model ===")

    # Trigger training, which writes PKLs to backend/models/
    train_model(
        property_data_path="data/us_housing/housing.csv",
        macro_data_path="data/us_market/Annual_Macroeconomic_Factors.csv"
    )

    print("\n=== Quick Synthetic Testing ===")
    df_test = pd.DataFrame([
        {"YrSold": 2008, "MSZoning": "RL", "OverallQual": 5, "GrLivArea": 1400, "Neighborhood": "NAmes"},
        {"YrSold": 2009, "MSZoning": "FV", "OverallQual": 9, "GrLivArea": 3000, "Neighborhood": "StoneBr"}
    ])

    scores = predict_risk_score(df_test)
    for idx, row in df_test.iterrows():
        print(f"Row {idx} => {dict(row)} | RiskScore={scores[idx]:.1f}")