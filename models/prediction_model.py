# prediction_model.py

import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder

# Global variables to hold the trained model and encoder once loaded (for reuse in a persistent environment)
_model = None
_encoder = None
_feature_columns = None
_median_values = None

def train_model(property_data_path: str, macro_data_path: str, 
                model_out_path: str = "risk_model.pkl",
                encoder_out_path: str = "risk_encoder.pkl",
                feature_meta_path: str = "model_columns.pkl") -> lgb.LGBMClassifier:
    """
    Train the LightGBM risk model using property and macroeconomic data, and save the model and encoders.
    
    Parameters:
        property_data_path (str): File path to the CSV containing property-level data (e.g., Ames housing data).
        macro_data_path (str): File path to the CSV with macroeconomic indicators, keyed by year (to join on YrSold).
        model_out_path (str): Filename to save the trained LightGBM model (pickle format).
        encoder_out_path (str): Filename to save the fitted OneHotEncoder for categorical features.
        feature_meta_path (str): Filename to save feature metadata (like final column names and imputation values).
    
    Returns:
        model (lgb.LGBMClassifier): The trained LightGBM model (also saved to disk).
    """
    # 1. Load datasets
    property_df = pd.read_csv(property_data_path)
    macro_df = pd.read_csv(macro_data_path)
    
    # 2. Merge macro indicators into property data on the year sold
    # Assuming macro_df has a column 'Year' that corresponds to property_df['YrSold']
    data = property_df.merge(macro_df, how='left', left_on='YrSold', right_on='Year')
    if 'Year' in data.columns:
        data.drop(columns=['Year'], inplace=True)  # drop redundant year column after merge
    
    # 3. Generate the binary risk target (top 20% SalePrice -> high risk)
    if 'SalePrice' not in data.columns:
        raise ValueError("SalePrice column not found in property data")
    # Determine the 80th percentile SalePrice value
    threshold = data['SalePrice'].quantile(0.80)
    # Create a new binary column 'HighRisk': 1 if SalePrice is in top 20%, else 0
    data['HighRisk'] = (data['SalePrice'] >= threshold).astype(int)
    
    # 4. Prepare feature set X and target y
    # Drop SalePrice and any identifier columns from features
    X = data.drop(columns=['SalePrice', 'HighRisk'], errors='ignore')
    # If there's an ID column (like 'Id' in Ames data), drop it as well
    X = X.loc[:, ~X.columns.str.lower().isin(['id', 'idx', 'index'])]
    y = data['HighRisk']
    
    # 5. Clean missing data
    # Fill missing numeric values with the median of that column (computed from training data)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    global _median_values
    _median_values = {col: X[col].median() for col in numeric_cols}
    for col in numeric_cols:
        X[col].fillna(_median_values[col], inplace=True)
    # Fill missing categorical values with a placeholder 'Missing'
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_cols:
        X[col].fillna('Missing', inplace=True)
    
    # 6. One-hot encode categorical features using a consistent template
    # We'll use OneHotEncoder to capture the training categories and reuse it for predictions&#8203;:contentReference[oaicite:7]{index=7}
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    # Fit the encoder on training categorical columns
    X_cat = pd.DataFrame()  # empty frame if no categoricals
    if len(categorical_cols) > 0:
        encoder.fit(X[categorical_cols])
        # Transform the categorical data to one-hot encoded numpy array
        X_cat_array = encoder.transform(X[categorical_cols])
        # Convert to DataFrame with proper column names
        # get_feature_names_out gives names like <col>_<value>
        cat_feature_names = encoder.get_feature_names_out(categorical_cols)
        X_cat = pd.DataFrame(X_cat_array, columns=cat_feature_names, index=X.index)
    # If needed, drop original categorical columns from X (they will be replaced by one-hot columns)
    X_num = X.drop(columns=categorical_cols, errors='ignore')
    # Optionally, drop YrSold if it was only used for merging macro (to avoid using raw year as a feature)
    if 'YrSold' in X_num.columns:
        # We drop YrSold as the macro features represent year effects; ensures model doesn't overly rely on year itself
        X_num.drop(columns=['YrSold'], inplace=True)
    # Combine numeric features and encoded categorical features
    X_full = pd.concat([X_num, X_cat], axis=1)
    
    # 7. Train the LightGBM model
    model = lgb.LGBMClassifier(objective='binary', random_state=42)
    model.fit(X_full, y)
    
    # 8. Save the trained model and preprocessing artifacts to disk
    joblib.dump(model, model_out_path)  # Save the LightGBM model&#8203;:contentReference[oaicite:8]{index=8}
    if len(categorical_cols) > 0:
        joblib.dump(encoder, encoder_out_path)  # Save the fitted OneHotEncoder for later use&#8203;:contentReference[oaicite:9]{index=9}
    # Save the list of feature columns and numeric imputation values for consistent inference
    feature_info = {
        "columns": X_full.columns.tolist(),
        "median_values": _median_values,
        "categorical_columns": categorical_cols
    }
    joblib.dump(feature_info, feature_meta_path)
    
    # Also store in global variables for immediate use without re-loading
    global _model, _encoder, _feature_columns
    _model = model
    _encoder = encoder if len(categorical_cols) > 0 else None
    _feature_columns = X_full.columns.tolist()
    
    return model

def predict_risk_score(input_data: pd.DataFrame, macro_data_path: str = None) -> pd.Series:
    """
    Predict the 0-100 risk score for new property data using the trained model.
    
    Parameters:
        input_data (pd.DataFrame): DataFrame containing one or more property records with the same features used in training 
                                   (except SalePrice and HighRisk). It should include categorical fields like Neighborhood, etc., 
                                   and numeric fields like GrLivArea, YearBuilt, YrSold, etc.
        macro_data_path (str, optional): Path to the macro indicators CSV, if needed for merging. If not provided, it assumes 
                                         the macro features are already present in input_data or were saved in training artifacts.
    
    Returns:
        pd.Series: Risk score(s) between 0 and 100 for each input record (float values, where 100 means highest risk).
    """
    global _model, _encoder, _feature_columns, _median_values
    # Load model and encoder from disk if not already loaded in memory
    if _model is None:
        _model = joblib.load("risk_model.pkl")
    # Load encoder and feature info
    feature_info = joblib.load("model_columns.pkl")
    saved_columns = feature_info["columns"]
    saved_medians = feature_info["median_values"]
    train_categorical = feature_info["categorical_columns"]
    if _encoder is None and train_categorical:
        # Only load the encoder if there were categorical features used in training
        _encoder = joblib.load("risk_encoder.pkl")
    _feature_columns = saved_columns
    _median_values = saved_medians
    
    # 1. Merge macro data if a macro CSV path is provided
    data = input_data.copy()
    if macro_data_path is not None:
        macro_df = pd.read_csv(macro_data_path)
        data = data.merge(macro_df, how='left', left_on='YrSold', right_on='Year')
        if 'Year' in data.columns:
            data.drop(columns=['Year'], inplace=True)
    # (If macro_data_path is None, we assume input_data already contains the necessary macro columns or none are needed)
    
    # 2. Ensure all required columns are present (fill missing columns with NaN if any missing)
    # If the input is missing any columns that the model expects (except target), add them with NaN
    for col in _feature_columns:
        if col not in data.columns:
            data[col] = np.nan
    
    # 3. Clean missing data in the input using the same strategy as training
    # Fill numeric NaN with medians from training
    for col, median_val in _median_values.items():
        if col in data.columns:
            data[col].fillna(median_val, inplace=True)
    # Fill categorical NaN with 'Missing'
    for col in train_categorical:
        if col in data.columns:
            data[col].fillna('Missing', inplace=True)
    
    # 4. One-hot encode categorical features using the saved encoder to ensure identical dummy columns
    X_new = data.copy()
    # Separate categorical and numeric as per training
    X_new_cat = pd.DataFrame()
    if train_categorical:
        # We assume the encoder was fit on these train_categorical features
        # Use the loaded encoder to transform new data (this will produce the same dummy columns as training&#8203;:contentReference[oaicite:10]{index=10})
        cat_array = _encoder.transform(X_new[train_categorical])
        cat_cols = _encoder.get_feature_names_out(train_categorical)
        X_new_cat = pd.DataFrame(cat_array, columns=cat_cols, index=X_new.index)
    # Drop original categorical columns and YrSold (if it was dropped during training) from X_new
    X_new_num = X_new.drop(columns=train_categorical + (['YrSold'] if 'YrSold' in X_new.columns else []), errors='ignore')
    # Combine numeric and dummy features
    X_new_full = pd.concat([X_new_num, X_new_cat], axis=1)
    # Reorder the columns to match the training feature order exactly&#8203;:contentReference[oaicite:11]{index=11}
    X_new_full = X_new_full.reindex(columns=_feature_columns, fill_value=0)
    
    # 5. Predict using the loaded model
    # LightGBM model expects the input features in the same order/structure as training
    # Get probability of class 1 (HighRisk) for each sample
    probs = _model.predict_proba(X_new_full)[:, 1]  # probability of being high risk
    risk_scores = probs * 100  # scale to 0-100
    # Return as pandas Series for convenience (align with input index)
    return pd.Series(risk_scores, index=X_new_full.index, name="risk_score")