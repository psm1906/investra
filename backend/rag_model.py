import os
import json
import google.generativeai as genai
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import uuid
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# ---------------------------------------------------------------
# GEMINI MODEL CONFIGURATION (ensure gemini API key is set)
# ---------------------------------------------------------------
# It is assumed that somewhere (e.g. in app.py) you call:
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# And that your model is available as follows:
gemini_pro_model = genai.GenerativeModel('gemini-2.0-flash')

# ---------------------------------------------------------------
# (Other functions such as train_model(), predict_risk_score(), etc.
#  remain as before.)
# ---------------------------------------------------------------


def generate_investment_analysis(risk_score: float, property_details: dict, user_profile: dict = None) -> str:
    """
    Use Gemini to produce a detailed, location-aware investment analysis using the inputs parsed from the UI.
    The output must be formatted strictly into the following sections:
    - Summary: A formal, objective summary of the investment opinion.
    - View Full Report: The complete analysis with detailed local market research.
    - Pros: A bullet-point list of advantages.
    - Cons: A bullet-point list of disadvantages.
    - High Risks: A bullet-point list of major risks.
    - Verdict: A final recommendation on whether to invest, hold off, or negotiate further.

    All inputs (e.g. property details) come from the UI. In addition, use the provided location to conduct in-depth research on local market conditions, including rental rates, property tax implications, neighborhood quality, and demographic trends.
    """
    prompt = f"""
You are a seasoned real estate investment expert with access to current market research and local economic data. Using the property location details provided below, conduct thorough local market research to evaluate local rental rates, property tax implications, neighborhood quality, demographic trends, and other economic factors that may affect the investment. Analyze the following property details and risk score, which have been parsed from the UI, and provide a comprehensive investment analysis in the exact format below. The analysis must be formal and objective without using conversational or first-person language.

Output your answer in the exact format:

Summary:
[Provide a brief, formal summary of the investment opinion.]

View Full Report:
[Provide the complete analysis including detailed local market research, property characteristics, and risk considerations.]

Pros:
- [List advantages of the property as bullet points.]

Cons:
- [List disadvantages of the property as bullet points.]

High Risks:
- [Highlight the major risks as bullet points.]

Verdict:
[Conclude with a final recommendation on whether to invest, hold off, or negotiate further.]

Property Details (parsed from UI):
- Location: {property_details.get('location', 'N/A')}
- Property Type: {property_details.get('PropertyType', property_details.get('propertyType', 'N/A'))}
- Year Built: {property_details.get('YearBuilt', property_details.get('yearBuilt', 'N/A'))}
- Square Footage: {property_details.get('SqFt', property_details.get('squareFootage', 'N/A'))}
- Bedrooms: {property_details.get('Bedrooms', property_details.get('bedrooms', 'N/A'))}
- Bathrooms: {property_details.get('Bathrooms', property_details.get('bathrooms', 'N/A'))}
- Condition: {property_details.get('Condition', property_details.get('condition', 'N/A'))}

Risk Score: {risk_score:.2f} (on a 0–100 scale where higher indicates greater risk)
"""
    try:
        response = gemini_pro_model.generate_content(prompt)
        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text.strip()
        else:
            return "Unable to generate AI analysis at this time."
    except Exception as e:
        print("Error generating investment analysis:", e)
        return "Error generating AI analysis."
    
def analyze_investment_risk_api(clerk_user_id: str, property_details: dict) -> dict:
    """
    Analyze the property risk and generate an AI investment analysis.
    """
    risk_score = predict_risk_score_from_ui(property_details)
    
    # Optionally, retrieve user financial data if available (for personalization)
    user_profile = None
    
    ai_recommendation = generate_investment_analysis(risk_score, property_details, user_profile)
    
    return {
        "user_id": clerk_user_id,
        "property_details": property_details,
        "risk_score": risk_score,
        "ai_recommendation": ai_recommendation
    }

def predict_risk_score(
    df_input: pd.DataFrame,
    macro_data_path: str = None,
    model_path: str = "backend/models/risk_model.pkl",
    encoder_path: str = "backend/models/risk_encoder.pkl",
    feat_info_path: str = "backend/models/model_columns.pkl"
) -> pd.Series:
    """
    Given new property rows in df_input, return a 0–100 risk score by applying the
    previously trained model and necessary transformations.
    """
    _model = None
    _encoder = None
    _feature_columns = None
    _median_values = None
    _prob_threshold = None

    # Load model artifacts if not already in memory.
    if _model is None:
        _model = joblib.load(model_path)

    feat_info = joblib.load(feat_info_path)

    if _encoder is None and feat_info["cat_cols"]:
        _encoder = joblib.load(encoder_path)

    _median_values = feat_info["median_values"]
    cat_cols       = feat_info["cat_cols"]
    _prob_threshold= feat_info["prob_threshold"]

    if _feature_columns is None:
        _feature_columns = feat_info["columns"]

    df = df_input.copy()

    # Merge macro data if YrSold exists.
    if macro_data_path is None:
        macro_data_path = "data/us_market/Annual_Macroeconomic_Factors.csv"
    if "YrSold" in df.columns:
        macro_df = pd.read_csv(macro_data_path)
        macro_df["Year"] = pd.to_datetime(macro_df["Date"]).dt.year
        macro_df = macro_df.rename(columns={
            "House_Price_Index":    "HousePriceIndex",
            "Stock_Price_Index":    "StockPriceIndex",
            "Consumer_Price_Index": "CPI",
            "Unemployment_Rate":    "UnemploymentRate",
            "Mortgage_Rate":        "MortgageRate",
            "Real_GDP":             "RealGDP",
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

    # Impute numeric columns.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in _median_values:
            df[col] = df[col].fillna(_median_values[col])
        else:
            df[col] = df[col].fillna(df[col].median())

    # Handle categorical columns.
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
    df_full = df_full.reindex(columns=_feature_columns, fill_value=0)

    # Predict probabilities and scale risk score.
    probs = _model.predict_proba(df_full)[:, 1]
    scaled = probs / _prob_threshold if _prob_threshold and _prob_threshold > 0 else probs
    scaled = np.clip(scaled, 0.0, 1.0)
    risk_scores = scaled * 100.0

    return pd.Series(risk_scores, index=df.index, name="risk_score")

def predict_risk_score_from_ui(user_input: dict) -> float:
    """
    Maps UI input keys to the columns expected by the model,
    calls predict_risk_score() on a single-row DataFrame, and returns the risk score.
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
    type_map = {"SingleFamily": "RL", "Townhouse": "RM", "Condo": "RM"}
    row_data["MSZoning"] = type_map.get(user_input.get("PropertyType"), "RL")
    if "MortgageRate" in user_input:
        row_data["MortgageRate"] = user_input["MortgageRate"]
    if "UnemploymentRate" in user_input:
        row_data["UnemploymentRate"] = user_input["UnemploymentRate"]
    if "CPI" in user_input:
        row_data["CPI"] = user_input["CPI"]
    df = pd.DataFrame([row_data])
    score_series = predict_risk_score(df)
    return float(score_series.iloc[0])

# The predict_risk_score() function and other model/training functions remain unchanged.
# ...
# (For brevity, the rest of the file is not shown here but remains as before.)

if __name__ == "__main__":
    print("==== Training the RiskRadar LightGBM model ====")
    train_model(
        property_data_path="data/us_housing/housing.csv",
        macro_data_path="data/us_market/Annual_Macroeconomic_Factors.csv",
        model_out_path="models/risk_model.pkl",
        encoder_out_path="models/risk_encoder.pkl",
        feature_meta_path="models/model_columns.pkl"
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