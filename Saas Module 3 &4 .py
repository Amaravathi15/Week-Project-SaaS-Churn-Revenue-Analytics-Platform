
#MODULE 3 — NumPy

import numpy as np 
import pandas as pd

# USING CLEANED DATASETS FROM MODULE 2

# 1. LOAD CLEANED DATASETS

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

print("Cleaned datasets loaded successfully.")

print("\nCustomers:", customers.shape)
print("Subscriptions:", subscriptions.shape)
print("Usage:", usage.shape)
print("Tickets:", tickets.shape)



# ============================================================
# 2. CONVERT REQUIRED COLUMNS TO NUMERIC
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

print("\nRequired columns converted to numeric successfully.")


# ============================================================
# 3. CONVERT MRR, SEATS AND USAGE METRICS TO NUMPY ARRAYS
# ============================================================

mrr_array = subscriptions["MRR"].dropna().to_numpy()
seats_array = subscriptions["Seats"].dropna().to_numpy()

logins_array = usage["Logins"].dropna().to_numpy()
active_users_array = usage["ActiveUsers"].dropna().to_numpy()
api_calls_array = usage["APICalls"].dropna().to_numpy()
session_minutes_array = usage["SessionMinutes"].dropna().to_numpy()

print("\n" + "=" * 70)
print("NUMPY ARRAY SIZES")
print("=" * 70)

print("MRR:", mrr_array.shape)
print("Seats:", seats_array.shape)
print("Logins:", logins_array.shape)
print("Active Users:", active_users_array.shape)
print("API Calls:", api_calls_array.shape)
print("Session Minutes:", session_minutes_array.shape)


# ============================================================
# 4. FUNCTION TO CALCULATE NUMPY STATISTICS
# ============================================================

def numpy_statistics(array, metric_name):

    print("\n" + "-" * 60)
    print(metric_name)
    print("-" * 60)

    print("Mean :", np.mean(array))
    print("Std  :", np.std(array))
    print("Min  :", np.min(array))
    print("Max  :", np.max(array))


    
 # 5. MRR STATISTICS


numpy_statistics(
    mrr_array,
    "MRR Statistics"
)

# 6. SEATS STATISTICS


numpy_statistics(
    seats_array,
    "Seats Statistics"
)

# 7. USAGE STATISTICS


numpy_statistics(
    logins_array,
    "Logins Statistics"
)

numpy_statistics(
    active_users_array,
    "Active Users Statistics"
)

numpy_statistics(
    api_calls_array,
    "API Calls Statistics"
)

numpy_statistics(
    session_minutes_array,
    "Session Minutes Statistics"
)

# 8. NORMALISE MRR USING MIN-MAX NORMALISATION
# ============================================================

mrr_min = np.min(mrr_array)
mrr_max = np.max(mrr_array)

subscriptions["MRR_Normalized"] = (
    (subscriptions["MRR"] - mrr_min)
    / (mrr_max - mrr_min)
)

print("\n" + "=" * 70)
print("MRR NORMALISATION")
print("=" * 70)

print(
    subscriptions[
        ["SubscriptionID", "CustomerID", "MRR", "MRR_Normalized"]
    ].head(10)
)


# 9. CREATE CUSTOMER-LEVEL METRICS
# ============================================================

# Total MRR per customer
customer_mrr = (
    subscriptions
    .groupby("CustomerID")["MRR"]
    .sum()
    .reset_index()
)

customer_mrr.rename(
    columns={"MRR": "Total_MRR"},
    inplace=True
)


# Total seats per customer
customer_seats = (
    subscriptions
    .groupby("CustomerID")["Seats"]
    .sum()
    .reset_index()
)

customer_seats.rename(
    columns={"Seats": "Total_Seats"},
    inplace=True
)


# Total logins per customer
customer_logins = (
    usage
    .groupby("CustomerID")["Logins"]
    .sum()
    .reset_index()
)

customer_logins.rename(
    columns={"Logins": "Total_Logins"},
    inplace=True
)


# Total active users per customer
customer_active_users = (
    usage
    .groupby("CustomerID")["ActiveUsers"]
    .sum()
    .reset_index()
)

customer_active_users.rename(
    columns={"ActiveUsers": "Total_ActiveUsers"},
    inplace=True
)


# Total API calls per customer
customer_api_calls = (
    usage
    .groupby("CustomerID")["APICalls"]
    .sum()
    .reset_index()
)

customer_api_calls.rename(
    columns={"APICalls": "Total_APICalls"},
    inplace=True
)


# ============================================================
# 10. MERGE CUSTOMER-LEVEL METRICS
# ============================================================

customer_metrics = customers[
    ["CustomerID", "CompanyName"]
].copy()

customer_metrics = customer_metrics.merge(
    customer_mrr,
    on="CustomerID",
    how="left"
)

customer_metrics = customer_metrics.merge(
    customer_seats,
    on="CustomerID",
    how="left"
)

customer_metrics = customer_metrics.merge(
    customer_logins,
    on="CustomerID",
    how="left"
)

customer_metrics = customer_metrics.merge(
    customer_active_users,
    on="CustomerID",
    how="left"
)

customer_metrics = customer_metrics.merge(
    customer_api_calls,
    on="CustomerID",
    how="left"
)

# Replace missing customer-level usage with 0.
# This is appropriate because a valid customer with no
# matching usage record has no recorded usage.

customer_metrics[
    [
        "Total_MRR",
        "Total_Seats",
        "Total_Logins",
        "Total_ActiveUsers",
        "Total_APICalls"
    ]
] = customer_metrics[
    [
        "Total_MRR",
        "Total_Seats",
        "Total_Logins",
        "Total_ActiveUsers",
        "Total_APICalls"
    ]
].fillna(0)

print("\n" + "=" * 70)
print("CUSTOMER METRICS MERGED SUCCESSFULLY")
print("=" * 70)

print(customer_metrics.head(10))
print("\nShape:", customer_metrics.shape)


# ============================================================
# 11. HIGH-VALUE ACCOUNT FLAG USING np.where()
# ============================================================

# High-value account:
# Total MRR >= overall median customer MRR

high_value_threshold = np.median(
    customer_metrics["Total_MRR"].to_numpy()
)

customer_metrics["High_Value_Flag"] = np.where(
    customer_metrics["Total_MRR"] >= high_value_threshold,
    "High Value",
    "Regular"
)

print("\n" + "=" * 70)
print("HIGH-VALUE ACCOUNT FLAG")
print("=" * 70)

print(
    "High-value threshold:",
    high_value_threshold
)

print(
    customer_metrics[
        [
            "CustomerID",
            "CompanyName",
            "Total_MRR",
            "High_Value_Flag"
        ]
    ].head(10)
)


# ============================================================
# 12. AT-RISK ACCOUNT FLAG USING np.where()
# ============================================================

# At-risk definition:
# Customer has relatively low usage compared with the
# overall customer usage distribution.
#
# Condition:
# Total_Logins < median Total_Logins
# AND Total_MRR > 0

login_threshold = np.median(
    customer_metrics["Total_Logins"].to_numpy()
)

customer_metrics["At_Risk_Flag"] = np.where(
    (
        (customer_metrics["Total_Logins"] < login_threshold)
        &
        (customer_metrics["Total_MRR"] > 0)
    ),
    "At Risk",
    "Not At Risk"
)

print("\n" + "=" * 70)
print("AT-RISK ACCOUNT FLAG")
print("=" * 70)

print(
    "Low-usage threshold:",
    login_threshold
)

print(
    customer_metrics[
        [
            "CustomerID",
            "CompanyName",
            "Total_MRR",
            "Total_Logins",
            "At_Risk_Flag"
        ]
    ].head(10)
)

# ============================================================
# 13. COMBINE ACCOUNT FLAGS
# ============================================================

customer_metrics["Account_Status"] = np.where(
    customer_metrics["High_Value_Flag"] == "High Value",
    "High Value",
    np.where(
        customer_metrics["At_Risk_Flag"] == "At Risk",
        "At Risk",
        "Regular"
    )
)

# 14. FINAL ACCOUNT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL CUSTOMER ACCOUNT SUMMARY")
print("=" * 70)

print(
    customer_metrics[
        [
            "CustomerID",
            "CompanyName",
            "Total_MRR",
            "Total_Seats",
            "Total_Logins",
            "Total_ActiveUsers",
            "Total_APICalls",
            "High_Value_Flag",
            "At_Risk_Flag",
            "Account_Status"
        ]
    ].head(20)
)


# ============================================================
# 15. COUNT ACCOUNT FLAGS
# ============================================================

print("\n" + "=" * 70)
print("ACCOUNT FLAG COUNTS")
print("=" * 70)

print("\nHigh-Value Accounts:")
print(
    customer_metrics["High_Value_Flag"].value_counts()
)

print("\nAt-Risk Accounts:")
print(
    customer_metrics["At_Risk_Flag"].value_counts()
)

print("\nFinal Account Status:")
print(
    customer_metrics["Account_Status"].value_counts()
)

# 16. SAVE MODULE 3 OUTPUT
# ============================================================

customer_metrics.to_csv(
    "module3_customer_numpy_analysis.csv",
    index=False
)

subscriptions.to_csv(
    "module3_subscriptions_numpy.csv",
    index=False
)

print("\n" + "=" * 70)
print("MODULE 3 COMPLETED SUCCESSFULLY")
print("=" * 70)




# ============================================================
# MODULE 4 — PANDAS WRANGLING & EDA
# ============================================================
# Using CLEANED datasets from Module 2


import pandas as pd
import numpy as np

# 1. LOADED CLEANED DATASETS

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


# 3. CONVERT NUMERIC COLUMNS
# ============================================================

numeric_columns = {
    "subscriptions": ["MRR", "Seats"],
    "usage": ["Logins", "ActiveUsers", "APICalls", "SessionMinutes"],
    "tickets": ["ResolutionHours", "SatisfactionScore"]
}

for column in numeric_columns["subscriptions"]:
    subscriptions[column] = pd.to_numeric(
        subscriptions[column],
        errors="coerce"
    )

for column in numeric_columns["usage"]:
    usage[column] = pd.to_numeric(
        usage[column],
        errors="coerce"
    )

for column in numeric_columns["tickets"]:
    tickets[column] = pd.to_numeric(
        tickets[column],
        errors="coerce"
    )


# CUSTOMERS — NO NUMERIC COLUMNS TO CONVERT
# ============================================================

print("\nCustomers: No numeric columns specified for conversion.")


# 4. CREATE CUSTOMER-LEVEL SUBSCRIPTION SUMMARY
# ============================================================

subscription_summary = (
    subscriptions
    .groupby("CustomerID")
    .agg(
        Total_MRR=("MRR", "sum"),
        Average_MRR=("MRR", "mean"),
        Total_Seats=("Seats", "sum"),
        Subscription_Count=("SubscriptionID", "nunique")
    )
    .reset_index()
)

print("\nSubscription summary:")
print(subscription_summary.head())


# 5. CREATE CUSTOMER-LEVEL USAGE SUMMARY
# ============================================================

usage_summary = (
    usage
    .groupby("CustomerID")
    .agg(
        Total_Logins=("Logins", "sum"),
        Average_ActiveUsers=("ActiveUsers", "mean"),
        Total_APICalls=("APICalls", "sum"),
        Total_SessionMinutes=("SessionMinutes", "sum"),
        Usage_Records=("Month", "count")
    )
    .reset_index()
)

print("\nUsage summary:")
print(usage_summary.head())


# ============================================================
# 6. CREATE CUSTOMER-LEVEL TICKET SUMMARY
# ============================================================

ticket_summary = (
    tickets
    .groupby("CustomerID")
    .agg(
        Ticket_Count=("TicketID", "nunique"),
        Average_ResolutionHours=("ResolutionHours", "mean"),
        Average_Satisfaction=("SatisfactionScore", "mean")
    )
    .reset_index()
)

print("\nTicket summary:")
print(ticket_summary.head())


# 7. MERGE ALL FOUR TABLES
# ============================================================

# Customers is the master/customer table.
# LEFT JOIN is used because every customer should remain
# in the final customer-level analytical view, even if that
# customer has no subscription, usage, or ticket records.

customer_view = customers.merge(
    subscription_summary,
    on="CustomerID",
    how="left"
)

customer_view = customer_view.merge(
    usage_summary,
    on="CustomerID",
    how="left"
)

customer_view = customer_view.merge(
    ticket_summary,
    on="CustomerID",
    how="left"
)


# Customers with no related activity receive zero
# for count/sum-based metrics.

zero_columns = [
    "Total_MRR",
    "Average_MRR",
    "Total_Seats",
    "Subscription_Count",
    "Total_Logins",
    "Average_ActiveUsers",
    "Total_APICalls",
    "Total_SessionMinutes",
    "Usage_Records",
    "Ticket_Count"
]

for column in zero_columns:
    customer_view[column] = customer_view[column].fillna(0)


print("\n" + "=" * 70)
print("CUSTOMER-LEVEL VIEW")
print("=" * 70)

print("Shape:", customer_view.shape)
print("\nColumns:")
print(customer_view.columns.tolist())

print("\nFirst 5 rows:")
print(customer_view.head())



# 8. GROUPBY — PLAN
# ============================================================

plan_analysis = (
    subscriptions
    .groupby("PlanName")
    .agg(
        Customers=("CustomerID", "nunique"),
        Subscriptions=("SubscriptionID", "nunique"),
        Total_MRR=("MRR", "sum"),
        Average_MRR=("MRR", "mean"),
        Total_Seats=("Seats", "sum"),
        Average_Seats=("Seats", "mean")
    )
    .reset_index()
)

print("\n" + "=" * 70)
print("GROUPBY — PLAN")
print("=" * 70)
print(plan_analysis)


# 9. GROUPBY — INDUSTRY
# ============================================================

industry_analysis = (
    customer_view
    .groupby("Industry")
    .agg(
        Customers=("CustomerID", "nunique"),
        Total_MRR=("Total_MRR", "sum"),
        Average_MRR=("Total_MRR", "mean"),
        Total_Seats=("Total_Seats", "sum"),
        Total_Logins=("Total_Logins", "sum"),
        Total_Tickets=("Ticket_Count", "sum")
    )
    .reset_index()
)

print("\n" + "=" * 70)
print("GROUPBY — INDUSTRY")
print("=" * 70)
print(industry_analysis)


# 10. GROUPBY — COUNTRY
# ============================================================

region_analysis = (
    customer_view
    .groupby("Country")
    .agg(
        Customers=("CustomerID", "nunique"),
        Total_MRR=("Total_MRR", "sum"),
        Average_MRR=("Total_MRR", "mean"),
        Total_Seats=("Total_Seats", "sum"),
        Total_Logins=("Total_Logins", "sum"),
        Total_Tickets=("Ticket_Count", "sum")
    )
    .reset_index()
)

print("\n" + "=" * 70)
print("GROUPBY — COUNTRY")
print("=" * 70)
print(region_analysis)


# 11. GROUPBY — ACQUISITION CHANNEL
# ============================================================

channel_analysis = (
    customer_view
    .groupby("AcquisitionChannel")
    .agg(
        Customers=("CustomerID", "nunique"),
        Total_MRR=("Total_MRR", "sum"),
        Average_MRR=("Total_MRR", "mean"),
        Total_Seats=("Total_Seats", "sum"),
        Total_Logins=("Total_Logins", "sum"),
        Total_Tickets=("Ticket_Count", "sum")
    )
    .reset_index()
)

print("\n" + "=" * 70)
print("GROUPBY — ACQUISITION CHANNEL")
print("=" * 70)
print(channel_analysis)


# 12. PIVOT TABLE — PLAN × BILLING TERM
# ============================================================

pivot_plan_billing = pd.pivot_table(
    subscriptions,
    index="PlanName",
    columns="BillingTerm",
    values="MRR",
    aggfunc="sum",
    fill_value=0
)

print("\n" + "=" * 70)
print("PIVOT — PLAN × BILLING TERM")
print("=" * 70)
print(pivot_plan_billing)


# 13. PIVOT TABLE — INDUSTRY × ACQUISITION CHANNEL
# ============================================================

pivot_industry_channel = pd.pivot_table(
    customer_view,
    index="Industry",
    columns="AcquisitionChannel",
    values="Total_MRR",
    aggfunc="sum",
    fill_value=0
)

print("\n" + "=" * 70)
print("PIVOT — INDUSTRY × ACQUISITION CHANNEL")
print("=" * 70)
print(pivot_industry_channel)



# 14. CALCULATED COLUMN — TENURE
# ============================================================

# Tenure is calculated in months from SignupDate to the
# latest available date in the dataset.

analysis_date = max(
    customers["SignupDate"].max(),
    subscriptions["StartDate"].max(),
    usage["Month"].max()
)

customer_view["Tenure_Months"] = (
    (analysis_date - customer_view["SignupDate"])
    .dt.days / 30.44
).round(2)

print("\nTenure created successfully.")


# 15. CALCULATED COLUMN — REVENUE PER SEAT
# ============================================================

customer_view["Revenue_Per_Seat"] = np.where(
    customer_view["Total_Seats"] > 0,
    customer_view["Total_MRR"] / customer_view["Total_Seats"],
    0
)

print("Revenue_Per_Seat created successfully.")


# 16. CALCULATED COLUMN — TICKETS PER MONTH
# ============================================================

customer_view["Tickets_Per_Month"] = np.where(
    customer_view["Tenure_Months"] > 0,
    customer_view["Ticket_Count"] /
    customer_view["Tenure_Months"],
    0
)

print("Tickets_Per_Month created successfully.")


# 17. CALCULATED COLUMN — USAGE TREND
# ============================================================

# Divide usage records into first half and second half
# of the available usage period.

usage_customer = usage.copy()

usage_customer["Usage_Period"] = np.where(
    usage_customer["Month"] <=
    usage_customer["Month"].median(),
    "First Half",
    "Second Half"
)

usage_trend = (
    usage_customer
    .groupby(["CustomerID", "Usage_Period"])
    .agg(
        Total_Logins=("Logins", "sum"),
        Total_ActiveUsers=("ActiveUsers", "sum"),
        Total_APICalls=("APICalls", "sum")
    )
    .reset_index()
)

usage_pivot = usage_trend.pivot(
    index="CustomerID",
    columns="Usage_Period",
    values="Total_Logins"
).fillna(0)

usage_pivot.columns.name = None

if "First Half" not in usage_pivot.columns:
    usage_pivot["First Half"] = 0

if "Second Half" not in usage_pivot.columns:
    usage_pivot["Second Half"] = 0

usage_pivot["Usage_Change"] = (
    usage_pivot["Second Half"] -
    usage_pivot["First Half"]
)

usage_pivot["Usage_Trend"] = np.where(
    usage_pivot["Usage_Change"] > 0,
    "Increasing",
    np.where(
        usage_pivot["Usage_Change"] < 0,
        "Decreasing",
        "Stable"
    )
)

customer_view = customer_view.merge(
    usage_pivot[["Usage_Trend"]],
    on="CustomerID",
    how="left"
)

customer_view["Usage_Trend"] = (
    customer_view["Usage_Trend"]
    .fillna("No Usage Data")
)

print("Usage_Trend created successfully.")


# 18. IQR OUTLIER DETECTION
# ============================================================

def detect_iqr_outliers(df, column):
    """
    Detect outliers using the IQR method.

    Q1 = 25th percentile
    Q3 = 75th percentile
    IQR = Q3 - Q1

    Lower Limit = Q1 - 1.5 × IQR
    Upper Limit = Q3 + 1.5 × IQR
    """

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)

    iqr = q3 - q1

    lower_limit = q1 - (1.5 * iqr)
    upper_limit = q3 + (1.5 * iqr)

    outlier_mask = (
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    )

    print("\n" + "-" * 60)
    print(f"IQR OUTLIER ANALYSIS — {column}")
    print("-" * 60)

    print("Q1:", round(q1, 2))
    print("Q3:", round(q3, 2))
    print("IQR:", round(iqr, 2))
    print("Lower Limit:", round(lower_limit, 2))
    print("Upper Limit:", round(upper_limit, 2))
    print("Outliers:", outlier_mask.sum())

    return outlier_mask


# ============================================================
# DETECT OUTLIERS IN IMPORTANT BUSINESS METRICS
# ============================================================

customer_view["MRR_Outlier"] = detect_iqr_outliers(
    customer_view,
    "Total_MRR"
)

customer_view["Seats_Outlier"] = detect_iqr_outliers(
    customer_view,
    "Total_Seats"
)

customer_view["Login_Outlier"] = detect_iqr_outliers(
    customer_view,
    "Total_Logins"
)

customer_view["Tickets_Outlier"] = detect_iqr_outliers(
    customer_view,
    "Ticket_Count"
)

print("\nIQR outlier detection completed successfully.")



# 19. CORRELATION MATRIX
# ============================================================

correlation_columns = [
    "Total_MRR",
    "Total_Seats",
    "Total_Logins",
    "Average_ActiveUsers",
    "Total_APICalls",
    "Total_SessionMinutes",
    "Ticket_Count",
    "Average_ResolutionHours",
    "Average_Satisfaction",
    "Tenure_Months",
    "Revenue_Per_Seat",
    "Tickets_Per_Month"
]

correlation_matrix = customer_view[
    correlation_columns
].corr()

print("\n" + "=" * 70)
print("CORRELATION MATRIX")
print("=" * 70)

print(correlation_matrix.round(3))



# 20. PRINT MEANINGFUL CORRELATIONS
# ============================================================

print("\n" + "=" * 70)
print("MEANINGFUL CORRELATIONS")
print("=" * 70)

for i in range(len(correlation_matrix.columns)):

    for j in range(i + 1, len(correlation_matrix.columns)):

        variable_1 = correlation_matrix.columns[i]
        variable_2 = correlation_matrix.columns[j]

        correlation = correlation_matrix.iloc[i, j]

        if abs(correlation) >= 0.30:

            if correlation >= 0.70:
                strength = "strong positive"

            elif correlation >= 0.50:
                strength = "moderate positive"

            elif correlation >= 0.30:
                strength = "weak positive"

            elif correlation <= -0.70:
                strength = "strong negative"

            elif correlation <= -0.50:
                strength = "moderate negative"

            else:
                strength = "weak negative"

            print(
                f"{variable_1} vs {variable_2}: "
                f"{correlation:.3f} → {strength} relationship"
            )





# 21. SAVE MODULE 4 OUTPUTS
# ============================================================

customer_view.to_csv(
    "module4_customer_level_view.csv",
    index=False
)

plan_analysis.to_csv(
    "module4_plan_analysis.csv",
    index=False
)

industry_analysis.to_csv(
    "module4_industry_analysis.csv",
    index=False
)

region_analysis.to_csv(
    "module4_region_analysis.csv",
    index=False
)

channel_analysis.to_csv(
    "module4_channel_analysis.csv",
    index=False
)

pivot_plan_billing.to_csv(
    "module4_pivot_plan_billing.csv",
    index=False
)

pivot_industry_channel.to_csv(
    "module4_pivot_industry_channel.csv",
    index=False
)

correlation_matrix.to_csv(
    "module4_correlation_matrix.csv",
    index=False
)


# ============================================================
# MODULE 4 COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("MODULE 4 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFiles saved:")
print("1. module4_customer_level_view.csv")
print("2. module4_plan_analysis.csv")
print("3. module4_industry_analysis.csv")
print("4. module4_region_analysis.csv")
print("5. module4_channel_analysis.csv")
print("6. module4_pivot_plan_billing.csv")
print("7. module4_pivot_industry_channel.csv")
print("8. module4_correlation_matrix.csv")