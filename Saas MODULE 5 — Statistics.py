
# MODULE 5 — STATISTICS

import pandas as pd
import numpy as np
from scipy import stats


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
print("MODULE 5 — STATISTICS")
print("=" * 70)

print("\nDatasets loaded successfully.")




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


# ============================================================
# 3. CREATE CUSTOMER-LEVEL METRICS
# ============================================================

customer_mrr = (
    subscriptions
    .groupby("CustomerID")
    .agg(
        Total_MRR=("MRR", "sum"),
        Total_Seats=("Seats", "sum")
    )
    .reset_index()
)

customer_usage = (
    usage
    .groupby("CustomerID")
    .agg(
        Total_Logins=("Logins", "sum"),
        Total_ActiveUsers=("ActiveUsers", "sum"),
        Total_APICalls=("APICalls", "sum"),
        Total_SessionMinutes=("SessionMinutes", "sum")
    )
    .reset_index()
)


# Merge customer-level metrics

customer_stats = customers[
    ["CustomerID", "CompanyName", "Industry", "Country"]
].merge(
    customer_mrr,
    on="CustomerID",
    how="left"
).merge(
    customer_usage,
    on="CustomerID",
    how="left"
)



# Customers without records receive zero activity/revenue

metric_columns = [
    "Total_MRR",
    "Total_Seats",
    "Total_Logins",
    "Total_ActiveUsers",
    "Total_APICalls",
    "Total_SessionMinutes"
]

for column in metric_columns:
    customer_stats[column] = customer_stats[column].fillna(0)


print("\nCustomer-level statistical dataset created.")
print("Shape:", customer_stats.shape)

# 4. DESCRIPTIVE STATISTICS
# ============================================================

key_metrics = [
    "Total_MRR",
    "Total_Seats",
    "Total_Logins",
    "Total_ActiveUsers",
    "Total_APICalls",
    "Total_SessionMinutes"
]

descriptive_stats = customer_stats[key_metrics].describe().T

descriptive_stats["median"] = (
    customer_stats[key_metrics].median()
)

descriptive_stats["variance"] = (
    customer_stats[key_metrics].var()
)

descriptive_stats["std"] = (
    customer_stats[key_metrics].std()
)

descriptive_stats = descriptive_stats[
    [
        "count",
        "mean",
        "std",
        "min",
        "25%",
        "median",
        "50%",
        "75%",
        "max",
        "variance"
    ]
]

print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)

print(descriptive_stats.round(2))


# ============================================================
# 5. TWO RANDOM SAMPLES
# ============================================================

# Sample 1
sample_1 = customer_stats.sample(
    n=50,
    random_state=42
)

# Sample 2
sample_2 = customer_stats.sample(
    n=50,
    random_state=7
)

# 6. COMPARE SAMPLE MEANS WITH POPULATION MEANS
# ============================================================

sample_comparison = pd.DataFrame({
    "Metric": key_metrics,

    "Population_Mean": [
        customer_stats[column].mean()
        for column in key_metrics
    ],

    "Sample_1_Mean": [
        sample_1[column].mean()
        for column in key_metrics
    ],

    "Sample_2_Mean": [
        sample_2[column].mean()
        for column in key_metrics
    ]
})

sample_comparison["Sample_1_Difference"] = (
    sample_comparison["Sample_1_Mean"]
    - sample_comparison["Population_Mean"]
)

sample_comparison["Sample_2_Difference"] = (
    sample_comparison["Sample_2_Mean"]
    - sample_comparison["Population_Mean"]
)

print("\n" + "=" * 70)
print("SAMPLE MEANS VS POPULATION MEANS")
print("=" * 70)

print(sample_comparison.round(2))

# ============================================================
# 7. CREATE CHURN / RETENTION STATUS
# ============================================================

# A customer is considered churned if they have a subscription
# with Status = Churned.

churned_customers = (
    subscriptions[
        subscriptions["Status"]
        .astype("string")
        .str.strip()
        .str.lower()
        == "churned"
    ]["CustomerID"]
    .dropna()
    .unique()
)

customer_stats["Customer_Status"] = np.where(
    customer_stats["CustomerID"].isin(churned_customers),
    "Churned",
    "Retained"
)


print("\n" + "=" * 70)
print("CUSTOMER STATUS")
print("=" * 70)

print(
    customer_stats["Customer_Status"]
    .value_counts()
)

# ============================================================
# 8. CHURN HYPOTHESIS
# ============================================================

print("\n" + "=" * 70)
print("CHURN HYPOTHESIS")
print("=" * 70)

print("""
H0 (Null Hypothesis):
Churned and retained customers have the same average
number of logins.

H1 (Alternative Hypothesis):
Churned customers have fewer average logins than
retained customers.
""")

# ============================================================
# 9. SEPARATE CHURNED AND RETAINED CUSTOMERS
# ============================================================

churned = customer_stats[
    customer_stats["Customer_Status"] == "Churned"
]["Total_Logins"]

retained = customer_stats[
    customer_stats["Customer_Status"] == "Retained"
]["Total_Logins"]


print("Churned customers:", len(churned))
print("Retained customers:", len(retained))

print(
    "\nAverage logins — Churned:",
    round(churned.mean(), 2)
)

print(
    "Average logins — Retained:",
    round(retained.mean(), 2)
)


# 10. INDEPENDENT TWO-SAMPLE T-TEST
# ============================================================

# Welch's t-test is used because the two groups may have
# different variances and different sample sizes.

t_statistic, p_value = stats.ttest_ind(
    churned,
    retained,
    equal_var=False,
    alternative="less"
)

print("\n" + "=" * 70)
print("CHURN HYPOTHESIS TEST")
print("=" * 70)

print("Test: Welch's independent two-sample t-test")
print("t-statistic:", round(t_statistic, 4))
print("p-value:", round(p_value, 6))

# ============================================================
# 11. INTERPRET THE P-VALUE
# ============================================================

alpha = 0.05

print("\nSignificance level:", alpha)

if p_value < alpha:

    print("""
RESULT:
Reject the null hypothesis.

The data provide statistically significant evidence that
churned customers have fewer logins than retained customers.
""")

else:

    print("""
RESULT:
Fail to reject the null hypothesis.

The data do not provide sufficient statistical evidence
that churned customers have fewer logins than retained customers.
""")


# 12. IMPORTANT P-VALUE EXPLANATION
# ============================================================

print("""
P-VALUE EXPLANATION:

The p-value measures how compatible the observed data are
with the null hypothesis.

A small p-value means that the observed difference would be
relatively unlikely if the null hypothesis were true.

A p-value below 0.05 is commonly treated as evidence against
the null hypothesis.

However, the p-value does NOT prove that churn causes lower
login activity.

It also does NOT prove that every churned customer logs in less.

Statistical significance shows evidence of a difference between
the groups, not causation.
""")


# ============================================================
# 13. EFFECT SIZE — DIFFERENCE IN MEANS
# ============================================================

mean_difference = (
    churned.mean()
    - retained.mean()
)

print("\n" + "=" * 70)
print("MEAN DIFFERENCE")
print("=" * 70)

print(
    "Churned mean logins:",
    round(churned.mean(), 2)
)

print(
    "Retained mean logins:",
    round(retained.mean(), 2)
)

print(
    "Difference (Churned - Retained):",
    round(mean_difference, 2)
)




# ============================================================
# 14. SAVE STATISTICAL OUTPUTS
# ============================================================

descriptive_stats.to_csv(
    "module5_descriptive_statistics.csv"
)

sample_comparison.to_csv(
    "module5_sample_vs_population.csv",
    index=False
)

hypothesis_result = pd.DataFrame({
    "Test": [
        "Welch Independent Two-Sample T-Test"
    ],
    "Metric": [
        "Total Logins"
    ],
    "Churned_Mean": [
        churned.mean()
    ],
    "Retained_Mean": [
        retained.mean()
    ],
    "Mean_Difference": [
        mean_difference
    ],
    "T_Statistic": [
        t_statistic
    ],
    "P_Value": [
        p_value
    ],
    "Alpha": [
        alpha
    ],
    "Significant": [
        p_value < alpha
    ]
})

hypothesis_result.to_csv(
    "module5_churn_hypothesis_test.csv",
    index=False
)

customer_stats.to_csv(
    "module5_customer_statistics.csv",
    index=False
)


# ============================================================
# 15. FINAL values 
# ============================================================

print("\n" + "=" * 70)
print("MODULE 5 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFiles saved:")
print("1. module5_descriptive_statistics.csv")
print("2. module5_sample_vs_population.csv")
print("3. module5_churn_hypothesis_test.csv")
print("4. module5_customer_statistics.csv")