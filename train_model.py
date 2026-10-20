"""
Compoundex - Country & Geopolitical Risk Model
===============================================
Risk area: Country & Geopolitical Risk

Compoundex teaches students to build discounted cash flow (DCF) valuations.
A core input to any DCF is the country risk premium added to the discount
rate of a company operating in a given country.

This model predicts whether a country will experience ELEVATED geopolitical
risk over the next 12 months - a materially adverse event such as sovereign
stress, conflict escalation, sanctions action or a disorderly political
transition - severe enough that students should raise the country risk
premium when valuing companies exposed to that country.

Outputs:
    fintech.xlsx     synthetic dataset (no index column)
    fintech.joblib   trained model
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Fixed seed so the dataset is always identical
np.random.seed(42)

N_OBSERVATIONS = 1000

# ----------------------------------------------------------------------
# 1. Generate the three country-risk predictor variables
# ----------------------------------------------------------------------

# Discrete: number of distinct international sanctions actions taken against
# the country in the last five years.
sanctions_events_last_5_years = np.random.poisson(lam=1.5, size=N_OBSERVATIONS)

# Continuous: 5-year sovereign credit default swap spread, in basis points.
# The market's own price of the country's default risk.
sovereign_cds_spread_bps = np.round(
    np.random.gamma(shape=2.0, scale=90.0, size=N_OBSERVATIONS), 2
)

# Continuous: political stability index, World Bank WGI scale,
# roughly -2.5 (fragile) to +2.5 (very stable).
political_stability_index = np.round(
    np.clip(np.random.normal(loc=0.0, scale=0.9, size=N_OBSERVATIONS), -2.5, 2.5), 3
)

# ----------------------------------------------------------------------
# 2. Build the target so it genuinely depends on the predictors
# ----------------------------------------------------------------------
# More sanctions and wider CDS spreads raise risk; greater political
# stability lowers it.
log_odds = (
    -2.20
    + 0.350 * sanctions_events_last_5_years
    + 0.0060 * sovereign_cds_spread_bps
    - 0.900 * political_stability_index
)
probability = 1 / (1 + np.exp(-log_odds))
elevated_country_risk = np.random.binomial(1, probability)

df = pd.DataFrame({
    "sanctions_events_last_5_years": sanctions_events_last_5_years,
    "sovereign_cds_spread_bps": sovereign_cds_spread_bps,
    "political_stability_index": political_stability_index,
    "elevated_country_risk": elevated_country_risk,
})

df.to_excel("fintech.xlsx", index=False)
print(f"Saved fintech.xlsx  ({len(df)} rows)")
print(f"Elevated-risk rate: {df['elevated_country_risk'].mean():.1%}\n")

# ----------------------------------------------------------------------
# 3. Train / test split and logistic regression
# ----------------------------------------------------------------------
feature_names = [
    "sanctions_events_last_5_years",
    "sovereign_cds_spread_bps",
    "political_stability_index",
]
X = df[feature_names]
y = df["elevated_country_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("=== CLASSIFICATION REPORT - TRAIN ===")
print(classification_report(y_train, model.predict(X_train)))
print("=== CLASSIFICATION REPORT - TEST ===")
print(classification_report(y_test, model.predict(X_test)))

print("Model coefficients:")
for name, coef in zip(feature_names, model.coef_[0]):
    print(f"  {name:<32} {coef:+.4f}")
print(f"  {'intercept':<32} {model.intercept_[0]:+.4f}\n")

# ----------------------------------------------------------------------
# 4. Persist the model
# ----------------------------------------------------------------------
joblib.dump(model, "fintech.joblib")
print("Saved fintech.joblib")
