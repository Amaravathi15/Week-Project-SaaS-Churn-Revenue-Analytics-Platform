# ============================================================
# MODULE 9 — CHURN RISK SCORING
# ============================================================

import pandas as pd
import numpy as np

print("MODULE 9 — Churn Risk Scoring")


# ============================================================
# 1. LOAD CLEANED DATASETS
# ============================================================


customers = pd.read_csv(
    "cleaned_saas_customers.csv"
)

subscriptions = pd.read_csv(
    "cleaned_saas_subscriptions.csv"
)

usage = pd.read_csv(
    "cleaned_saas_usage.csv"
)

tickets = pd.read_csv(
    "cleaned_saas_tickets.csv"
)

print("=" * 70)
print("MODULE 9 — CHURN RISK SCORING")
print("=" * 70)

print("\nDatasets loaded successfully.")


print("Customers:", customers.shape)
print("Subscriptions:", subscriptions.shape)
print("Usage:", usage.shape)
print("Tickets:", tickets.shape)


# ============================================================
# 2. CONVERT NUMERIC COLUMNS
# ============================================================

subscriptions["MRR"] = pd.to_numeric(
    subscriptions["MRR"],
    errors="coerce"
)

subscriptions["Seats"] = pd.to_numeric(
    subscriptions["Seats"],
    errors="coerce"
)

usage["Logins"] = pd.to_numeric(
    usage["Logins"],
    errors="coerce"
)

usage["ActiveUsers"] = pd.to_numeric(
    usage["ActiveUsers"],
    errors="coerce"
)

usage["APICalls"] = pd.to_numeric(
    usage["APICalls"],
    errors="coerce"
)

usage["SessionMinutes"] = pd.to_numeric(
    usage["SessionMinutes"],
    errors="coerce"
)

tickets["ResolutionHours"] = pd.to_numeric(
    tickets["ResolutionHours"],
    errors="coerce"
)

tickets["SatisfactionScore"] = pd.to_numeric(
    tickets["SatisfactionScore"],
    errors="coerce"
)

print("Numeric conversion completed.")



# ============================================================
# 3. CREATE CUSTOMER-LEVEL METRICS
# ============================================================

# -------------------------
# Subscription metrics
# -------------------------

subscription_metrics = (
    subscriptions
    .groupby("CustomerID")
    .agg(
        Total_MRR=("MRR", "sum"),
        Total_Seats=("Seats", "sum")
    )
    .reset_index()
)


# -------------------------
# Usage metrics
# -------------------------

usage_metrics = (
    usage
    .groupby("CustomerID")
    .agg(
        Total_Logins=("Logins", "sum"),
        Average_ActiveUsers=("ActiveUsers", "mean"),
        Total_SessionMinutes=("SessionMinutes", "sum"),
        Total_APICalls=("APICalls", "sum")
    )
    .reset_index()
)


# -------------------------
# Ticket metrics
# -------------------------

ticket_metrics = (
    tickets
    .groupby("CustomerID")
    .agg(
        Ticket_Count=("TicketID", "nunique"),
        Average_Satisfaction=("SatisfactionScore", "mean")
    )
    .reset_index()
)


# ============================================================
# 4. CREATE CUSTOMER RISK TABLE
# ============================================================

risk_data = customers[
    ["CustomerID", "CompanyName"]
].copy()

risk_data = risk_data.merge(
    subscription_metrics,
    on="CustomerID",
    how="left"
)

risk_data = risk_data.merge(
    usage_metrics,
    on="CustomerID",
    how="left"
)

risk_data = risk_data.merge(
    ticket_metrics,
    on="CustomerID",
    how="left"
)


# Customers without activity get zero activity

numeric_columns = [
    "Total_MRR",
    "Total_Seats",
    "Total_Logins",
    "Average_ActiveUsers",
    "Total_SessionMinutes",
    "Total_APICalls",
    "Ticket_Count",
    "Average_Satisfaction"
]

risk_data[numeric_columns] = (
    risk_data[numeric_columns]
    .fillna(0)
)


print("\nCustomer risk table created.")
print(risk_data.head())


# ============================================================
# 5. CREATE RISK SIGNALS
# ============================================================

# Median thresholds are used so that the risk rules
# are based on the behaviour of this dataset.

login_threshold = risk_data["Total_Logins"].median()

active_user_threshold = (
    risk_data["Average_ActiveUsers"].median()
)

session_threshold = (
    risk_data["Total_SessionMinutes"].median()
)

ticket_threshold = (
    risk_data["Ticket_Count"].median()
)

satisfaction_threshold = (
    risk_data["Average_Satisfaction"].median()
)


print("\nRisk thresholds:")
print("Login threshold:", login_threshold)
print("Active-user threshold:", active_user_threshold)
print("Session-minute threshold:", session_threshold)
print("Ticket threshold:", ticket_threshold)
print("Satisfaction threshold:", satisfaction_threshold)


# ============================================================
# 6. CALCULATE INDIVIDUAL RISK SIGNALS
# ============================================================

# Signal 1 — Low login activity

risk_data["Low_Login_Risk"] = np.where(
    risk_data["Total_Logins"] < login_threshold, 1,0)


# Signal 2 — Low active-user activity

risk_data["Low_ActiveUser_Risk"] = np.where(
    risk_data["Average_ActiveUsers"] < active_user_threshold,1,0)


# Signal 3 — Low session activity

risk_data["Low_Session_Risk"] = np.where(
    risk_data["Total_SessionMinutes"] < session_threshold,1, 0)


# Signal 4 — High ticket volume

risk_data["High_Ticket_Risk"] = np.where(
    risk_data["Ticket_Count"] > ticket_threshold,1,0)


# Signal 5 — Low satisfaction

# Only customers with recorded satisfaction
# are considered for this signal.

risk_data["Low_Satisfaction_Risk"] = np.where(
    (
        (risk_data["Average_Satisfaction"] > 0)
        &
        (
            risk_data["Average_Satisfaction"]
            < satisfaction_threshold
        )
    ),1,0)



# ============================================================
# 7. CREATE WEIGHTED RISK SCORE
# ============================================================

# Each signal contributes points.
#
# Low login       = 2 points
# Low active user = 1 point
# Low sessions    = 2 points
# High tickets    = 1 point
# Low satisfaction= 2 points
#
# Maximum = 8 points

risk_data["Risk_Score"] = (
    risk_data["Low_Login_Risk"] * 2
    +
    risk_data["Low_ActiveUser_Risk"] * 1
    +
    risk_data["Low_Session_Risk"] * 2
    +
    risk_data["High_Ticket_Risk"] * 1
    +
    risk_data["Low_Satisfaction_Risk"] * 2
)


print("\nRisk score distribution:")
print(
    risk_data["Risk_Score"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 8. CLASSIFY CUSTOMERS BY RISK
# ============================================================

risk_data["Risk_Level"] = np.select(
    [
        risk_data["Risk_Score"] >= 6,
        risk_data["Risk_Score"] >= 3
    ],
    [
        "High Risk",
        "Medium Risk"
    ],
    default="Low Risk"
)


print("\nRisk level counts:")
print(
    risk_data["Risk_Level"]
    .value_counts()
)


# ============================================================
# 9. RANK CUSTOMERS BY RISK
# ============================================================

risk_data = risk_data.sort_values(
    by=[
        "Risk_Score",
        "Total_MRR"
    ],
    ascending=[
        False,
        False
    ]
).reset_index(drop=True)


# Rank 1 = highest risk

risk_data["Risk_Rank"] = (
    risk_data.index + 1
)

# ============================================================
# 10. SHOW HIGHEST-RISK CUSTOMERS
# ============================================================

print("\nTOP 20 HIGHEST-RISK CUSTOMERS")

print(
    risk_data[
        [
            "Risk_Rank",
            "CustomerID",
            "CompanyName",
            "Total_MRR",
            "Risk_Score",
            "Risk_Level"
        ]
    ].head(20)
)

# ============================================================
# 11. CALCULATE MRR IN HIGHEST-RISK GROUP
# ============================================================

high_risk_customers = risk_data[
    risk_data["Risk_Level"] == "High Risk"
].copy()


high_risk_mrr = (
    high_risk_customers["Total_MRR"]
    .sum()
)


total_mrr = (
    risk_data["Total_MRR"]
    .sum()
)


if total_mrr > 0:
    high_risk_mrr_percentage = (
        high_risk_mrr / total_mrr
    ) * 100
else:
    high_risk_mrr_percentage = 0
    

print("\n============================================================")
print("HIGHEST-RISK GROUP")
print("============================================================")

print(
    "High-risk customers:",
    len(high_risk_customers)
)

print(
    "MRR in high-risk group:",
    round(high_risk_mrr, 2)
)

print(
    "Percentage of total MRR:",
    round(high_risk_mrr_percentage, 2),
    "%"
)


# ============================================================
# 12. RISK SUMMARY
# ============================================================

risk_summary = (
    risk_data
    .groupby("Risk_Level")
    .agg(
        Customer_Count=("CustomerID", "count"),
        Total_MRR=("Total_MRR", "sum"),
        Average_MRR=("Total_MRR", "mean"),
        Average_Risk_Score=("Risk_Score", "mean")
    )
    .round(2)
)

# Put risk levels in logical order

risk_order = [
    "Low Risk",
    "Medium Risk",
    "High Risk"
]

risk_summary = (
    risk_summary
    .reindex(risk_order)
)

print("\nRISK SUMMARY")
print(risk_summary)


# 13. SAVE CUSTOMER RISK RANKING
# ============================================================

risk_data.to_csv(
    "module9_customer_churn_risk.csv",
    index=False
)

# ============================================================
# 14. SAVE RISK SUMMARY
# ============================================================

risk_summary.to_csv(
    "module9_risk_summary.csv"
)

print("\n============================================================")
print("MODULE 9 FILES SAVED")
print("============================================================")

print(
    "1. module9_customer_churn_risk.csv")

print(
    "2. module9_risk_summary.csv")


# ============================================================
# 15. FINAL MODULE 9 STATEMENT
# ============================================================

print("\nMODULE 9 CONCLUSION")

print(
    f"The highest-risk group contains "
    f"{len(high_risk_customers)} customers "
    f"with total MRR of "
    f"{high_risk_mrr:.2f}."
)

print(
    f"This represents "
    f"{high_risk_mrr_percentage:.2f}% "
    f"of total customer MRR."
)