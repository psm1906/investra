import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb
import pickle

# ----------------------------
# 1. Data Loading and Cleaning
# ----------------------------

# Load the Ames housing dataset (ensure the CSV is in your working directory)
ames = pd.read_csv('ames.csv')

# Fill missing numeric values with the median of each column
for col in ames.select_dtypes(include=[np.number]).columns:
    ames[col].fillna(ames[col].median(), inplace=True)

# Fill missing categorical values with the mode (most frequent value) for each column
for col in ames.select_dtypes(include=['object']).columns:
    ames[col].fillna(ames[col].mode()[0], inplace=True)

# Remove outliers: for example, filter out properties with extremely high living area
# (Ames dataset is known to have some outliers in 'GrLivArea'; adjust threshold as needed)
ames = ames[ames['GrLivArea'] < 4000]

# ----------------------------
# 2. Merge with Macroeconomic Data
# ----------------------------

# Load the macroeconomic dataset (ensure 'macro.csv' is in your working directory)
# Expecting macro.csv to have columns: 'Year', 'InterestRate', 'Inflation'
macro = pd.read_csv('macro.csv')
macro.fillna(method='ffill', inplace=True)  # forward-fill missing values

# Merge macro data with Ames on the sale year. 'YrSold' is the sale year in Ames.
ames = ames.merge(macro, left_on='YrSold', right_on='Year', how='left')

# Quick check: Print the first few rows to verify merge
print("Merged data preview:")
print(ames.head())

# ----------------------------
# 3. Feature Selection & Risk Target Definition
# ----------------------------

# Select key features for risk prediction:
# - 'Neighborhood': provides location information.
# - 'GrLivArea': above-ground living area (size).
# - 'YearBuilt': property age.
# - 'OverallQual': quality/condition rating.
# - 'LotArea': size of the lot.
# - 'InterestRate' and 'Inflation': macroeconomic factors.
selected_features = ['Neighborhood', 'GrLivArea', 'YearBuilt', 'OverallQual', 'LotArea', 'InterestRate', 'Inflation']

# For our risk target, create a binary label.
# One approach: label properties in the top 20% of SalePrice as "high risk" (1) and others as "low risk" (0)
threshold = ames['SalePrice'].quantile(0.8)
ames['RiskLabel'] = (ames['SalePrice'] >= threshold).astype(int)

# Prepare feature matrix X and target vector y
X = ames[selected_features]
y = ames['RiskLabel']

# ----------------------------
# 4. Encoding and Preprocessing
# ----------------------------

# Encode categorical features. Here, one-hot encode 'Neighborhood'
X = pd.get_dummies(X, columns=['Neighborhood'], drop_first=True)

# (Optional) If desired, you could scale the macro variables here using StandardScaler,
# but LightGBM generally works well without scaling.

# ----------------------------
# 5. Splitting Data into Training and Testing Sets
# ----------------------------

# Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# 6. Train LightGBM Model
# ----------------------------

# Initialize a LightGBM classifier using default parameters for rapid prototyping.
model = lgb.LGBMClassifier(random_state=42)
model.fit(X_train, y_train)

# ----------------------------
# 7. Evaluate the Model
# ----------------------------

# Get predictions and predicted probabilities
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # probability for class "1" (high risk)

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\nModel Evaluation:")
print("Test Accuracy: {:.2f}%".format(accuracy * 100))
print("Test ROC AUC: {:.3f}".format(roc_auc))

# Convert predicted probabilities to a risk score on a 0-100 scale (for user-friendliness)
risk_scores = (y_prob * 100).round(1)
print("\nSample Risk Scores (0-100 scale):")
print(risk_scores[:10])

# ----------------------------
# 8. Save the Trained Model
# ----------------------------

# Save the model to a file for later integration (e.g., with a Node.js backend)
with open('risk_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel training complete and saved to 'risk_model.pkl'")