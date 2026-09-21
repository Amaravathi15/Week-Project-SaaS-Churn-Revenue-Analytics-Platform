import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
# 2. CONVERT DATE COLUMNS
# ============================================================

customers["SignupDate"] = pd.to_datetime(
    customers["SignupDate"],
    errors="coerce"
)

subscriptions["StartDate"] = pd.to_datetime(
    subscriptions["StartDate"],
    errors="coerce"
)

subscriptions["EndDate"] = pd.to_datetime(
    subscriptions["EndDate"],
    errors="coerce"
)

usage["Month"] = pd.to_datetime(
    usage["Month"],
    errors="coerce"
)


# ============================================================
# 3. CREATE SIGNUP COHORT MONTH
# ============================================================

customers["CohortMonth"] = (
    customers["SignupDate"]
    .dt.to_period("M")
)

print("\nSignup cohorts created successfully.")

print(
    customers[
        ["CustomerID", "SignupDate", "CohortMonth"]
    ].head()
)

# ============================================================
# 4. CREATE CUSTOMER ACTIVITY MONTH
# ============================================================

# Usage activity is used to determine whether a customer
# was active during a particular month.

usage["ActivityMonth"] = (
     usage["Month"]
    .dt.to_period("M")
)

customer_activity = (
    usage[
        ["CustomerID", "ActivityMonth"]
    ]
    .dropna()
    .drop_duplicates()
)

print("\nCustomer activity table created.")
print(customer_activity.head())


# ============================================================
# 5. MERGE COHORT INFORMATION WITH ACTIVITY
# ============================================================

cohort_activity = customer_activity.merge(
    customers[
        ["CustomerID", "CohortMonth"]
    ],
    on="CustomerID",
    how="left"
)

cohort_activity = cohort_activity.dropna(
    subset=["CohortMonth", "ActivityMonth"]
)



# ============================================================
# 6. CALCULATE MONTHS SINCE SIGNUP
# ============================================================

cohort_activity["CohortIndex"] = (
    (
        cohort_activity["ActivityMonth"].dt.year
        - cohort_activity["CohortMonth"].dt.year
    ) * 12
    +
    (
        cohort_activity["ActivityMonth"].dt.month
        - cohort_activity["CohortMonth"].dt.month
    )
)

# Keep only activity occurring from signup month onwards

cohort_activity = cohort_activity[
    cohort_activity["CohortIndex"] >= 0
]

print("\nCohort index calculated successfully.")


# ============================================================
# 7. COUNT CUSTOMERS IN EACH COHORT
# ============================================================

cohort_sizes = (
    customers
    .groupby("CohortMonth")["CustomerID"]
    .nunique()
)

print("\nCohort sizes:")
print(cohort_sizes)


# ============================================================
# 8. COUNT ACTIVE CUSTOMERS BY COHORT AND MONTH
# ============================================================

active_customers = (
    cohort_activity
    .groupby(
        ["CohortMonth", "CohortIndex"]
    )["CustomerID"]
    .nunique()
)

print("\nActive customers by cohort:")
print(active_customers.head(10))


# ============================================================
# 9. CREATE RETENTION TABLE
# ============================================================

retention_table = (
    active_customers
    .unstack(fill_value=0)
)

# Divide active customers by original cohort size

retention_rate = retention_table.divide(
    cohort_sizes,
    axis=0
) * 100

retention_rate = retention_rate.round(2)

print("\n" + "=" * 70)
print("COHORT RETENTION TABLE (%)")
print("=" * 70)

print(retention_rate)


# ============================================================
# 10. CREATE RETENTION CURVE
# ============================================================

average_retention = retention_rate.mean(
    axis=0
)

plt.figure(figsize=(10, 6))

plt.plot(
    average_retention.index,
    average_retention.values,
    marker="o"
)

plt.title("Average Customer Retention Curve")
plt.xlabel("Months Since Signup")
plt.ylabel("Average Retention (%)")
plt.xticks(average_retention.index)
plt.grid(True)

plt.tight_layout()
plt.show()

# ============================================================
# 11. RETENTION AT SPECIFIC MONTHS
# ============================================================

retention_summary = pd.DataFrame({
    "Month": average_retention.index,
    "Average_Retention_Percentage": average_retention.values
})

print("\n" + "=" * 70)
print("AVERAGE RETENTION CURVE DATA")
print("=" * 70)

print(retention_summary.round(2))



# ============================================================
# 12. IDENTIFY BEST AND WORST COHORTS
# ============================================================

# Use Month 1 retention when available because older cohorts
# may have more observation time than newer cohorts.

month_1_retention = retention_rate[1].dropna()

if len(month_1_retention) > 0:

    best_cohort = month_1_retention.idxmax()
    worst_cohort = month_1_retention.idxmin()

    best_value = month_1_retention.max()
    worst_value = month_1_retention.min()

    print("\n" + "=" * 70)
    print("BEST AND WORST COHORTS")
    print("=" * 70)

    print(
        "Best cohort based on Month 1 retention:",
        best_cohort
    )

    print(
        "Month 1 retention:",
        round(best_value, 2),
        "%"
    )

    print(
        "\nWorst cohort based on Month 1 retention:",
        worst_cohort
    )

    print(
        "Month 1 retention:",
        round(worst_value, 2),
        "%"
    )

else:

    print(
        "\nMonth 1 retention is not available "
        "for enough cohorts to compare."
    )



# ============================================================
# 13. COMPARE RETENTION CHANGES
# ============================================================

if len(month_1_retention) > 0:

    retention_change = (
        best_value - worst_value
    )

    print(
        "\nRetention difference between best "
        "and worst cohorts:",
        round(retention_change, 2),
        "percentage points"
    )




# ============================================================
# 14. CREATE COHORT SUMMARY
# ============================================================

cohort_summary = (
    customers
    .groupby("CohortMonth")
    .agg(
        Customers=("CustomerID", "nunique")
    )
    .reset_index()
)

if 1 in retention_rate.columns:

    cohort_summary = cohort_summary.merge(
        retention_rate[1]
        .rename("Month_1_Retention"),
        left_on="CohortMonth",
        right_index=True,
        how="left"
    )

if 2 in retention_rate.columns:

    cohort_summary = cohort_summary.merge(
        retention_rate[2]
        .rename("Month_2_Retention"),
        left_on="CohortMonth",
        right_index=True,
        how="left"
    )

if 3 in retention_rate.columns:

    cohort_summary = cohort_summary.merge(
        retention_rate[3]
        .rename("Month_3_Retention"),
        left_on="CohortMonth",
        right_index=True,
        how="left"
    )

print("\n" + "=" * 70)
print("COHORT SUMMARY")
print("=" * 70)

print(cohort_summary.round(2))



# ============================================================
# 15. SAVE MODULE 6 OUTPUTS
# ============================================================

retention_rate.to_csv(
    "module6_cohort_retention_table.csv"
)

retention_summary.to_csv(
    "module6_retention_curve.csv",
    index=False
)

cohort_summary.to_csv(
    "module6_cohort_summary.csv",
    index=False
)

cohort_activity.to_csv(
    "module6_customer_cohort_activity.csv",
    index=False
)
  

print("\n" + "=" * 70)
print("MODULE 6 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFiles saved:")
print("1. module6_cohort_retention_table.csv")
print("2. module6_retention_curve.csv")
print("3. module6_cohort_summary.csv")