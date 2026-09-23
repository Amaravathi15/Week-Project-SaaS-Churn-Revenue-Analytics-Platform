# MODULE 8 — CUSTOMER SEGMENTATION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans



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
print("MODULE 8 — CUSTOMER SEGMENTATION")
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
# 3. BUILD CUSTOMER-LEVEL FEATURES
# ============================================================

# ----------------------------
# Subscription features
# ----------------------------

subscription_features = (
    subscriptions
    .groupby("CustomerID")
    .agg(
        Total_MRR=("MRR", "sum"),
        Total_Seats=("Seats", "sum")
    )
    .reset_index()
)


# ----------------------------
# Usage features
# ----------------------------

usage_features = (
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


# ----------------------------
# Ticket features
# ----------------------------

ticket_features = (
    tickets
    .groupby("CustomerID")
    .agg(
        Ticket_Count=("CustomerID", "count"),
        Avg_Satisfaction=("SatisfactionScore", "mean")
    )
    .reset_index()
)



# ============================================================
# 4. MERGE CUSTOMER-LEVEL FEATURES
# ============================================================

customer_features = customers[
    ["CustomerID", "CompanyName"]
].copy()

customer_features = customer_features.merge(
    subscription_features,
    on="CustomerID",
    how="left"
)

customer_features = customer_features.merge(
    usage_features,
    on="CustomerID",
    how="left"
)

customer_features = customer_features.merge(
    ticket_features,
    on="CustomerID",
    how="left"
)

print("\nCustomer-level feature table:")
print(customer_features.head())

print("\nShape:", customer_features.shape)

#LEFT JOIN

#We start with the customer table because every customer should remain in the segmentation analysis, 
#even if they have no recorded usage or tickets.

#For aggregated metrics such as usage or tickets, 
#a missing value means there is no recorded activity, so we can treat those aggregated measures as zero.



# ============================================================
# 5. HANDLE MISSING CUSTOMER ACTIVITY
# ============================================================

feature_columns = [
    "Total_MRR",
    "Total_Seats",
    "Total_Logins",
    "Total_ActiveUsers",
    "Total_APICalls",
    "Total_SessionMinutes",
    "Ticket_Count",
    "Avg_Satisfaction"
]

for col in feature_columns:
    customer_features[col] = customer_features[col].fillna(0)


print("\nCustomer-level features created.")

print(
    customer_features[
        ["CustomerID"] + feature_columns
    ].head()
)

print("\nMissing values in segmentation features:")
print(
    customer_features[feature_columns]
    .isna()
    .sum()
)


# ============================================================
# 6. SELECT FEATURES FOR CLUSTERING
# ============================================================

clustering_features = [
    "Total_MRR",
    "Total_Seats",
    "Total_Logins",
    "Total_ActiveUsers",
    "Total_APICalls",
    "Total_SessionMinutes",
    "Ticket_Count",
    "Avg_Satisfaction"
]

X = customer_features[
    clustering_features
].copy()

print("\nFeatures used for K-Means:")
print(clustering_features)

print("\nFeature matrix shape:", X.shape)


# 7. SCALE FEATURES
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nFeatures scaled successfully.")
print("Scaled matrix shape:", X_scaled.shape)


# ============================================================
# 8. ELBOW METHOD
# ============================================================

inertia = []

k_values = range(2, 9)

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    inertia.append(
        kmeans.inertia_
    )


# ============================================================
# 9. ELBOW PLOT
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    k_values,
    inertia,
    marker="o"
)


plt.title("Elbow Method for Customer Segmentation")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Within-Cluster Sum of Squares (Inertia)")
plt.xticks(list(k_values))
plt.grid(True)

plt.show()


# 10. DISPLAY ELBOW VALUES
# ============================================================

elbow_table = pd.DataFrame({
    "k": list(k_values),
    "Inertia": inertia
})

print("\nElbow Method Results:")
print(elbow_table)

#The Elbow Method indicated k = 4 because the reduction in inertia became substantially smaller after four clusters.
#Therefore, four clusters were selected as a practical balance between segmentation detail and model simplicity.



# ============================================================
# 11. CHOOSE K
# ============================================================

# Based on the elbow plot, select the point
# where the decrease in inertia starts slowing down.

chosen_k = 4

print(
    "\nChosen number of clusters (k):",
    chosen_k
)

print(
    "Justification: k=4 provides a practical balance "
    "between reducing within-cluster variation and "
    "keeping the customer segments easy to interpret."
)


# ============================================================
# 12 RUN K-MEANS
# ============================================================


kmeans_final = KMeans(
    n_clusters=chosen_k,
    random_state=42,
    n_init=10
)

customer_features["Cluster"] = (
    kmeans_final.fit_predict(X_scaled)
)

print("\nK-Means completed.")

print("\nNumber of customers in each cluster:")
print(
    customer_features["Cluster"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 13. ANALYSE CLUSTER PROFILES & IDENTIFY CLUSTER BEHAVIOUR
# ============================================================

cluster_profile = (
    customer_features
    .groupby("Cluster")[clustering_features]
    .mean()
    .round(2)
)

cluster_size = (
    customer_features["Cluster"]
    .value_counts()
    .sort_index()
    .rename("Customer_Count")
)

cluster_profile = cluster_profile.join(
    cluster_size
)

print("\nCluster Profiles:")
print(cluster_profile)

# ============================================================
# 14 BEHAVIOUR-BASED SEGMENT NAMING
# ============================================================


# Calculate medians across clusters

mrr_median = cluster_profile["Total_MRR"].median()
usage_median = cluster_profile["Total_Logins"].median()
ticket_median = cluster_profile["Ticket_Count"].median()
satisfaction_median = cluster_profile["Avg_Satisfaction"].median()

segment_names = {}

for cluster in cluster_profile.index:

    row = cluster_profile.loc[cluster]

    high_mrr = row["Total_MRR"] >= mrr_median
    high_usage = row["Total_Logins"] >= usage_median
    high_tickets = row["Ticket_Count"] >= ticket_median
    high_satisfaction = (
        row["Avg_Satisfaction"] >= satisfaction_median
    )

    if high_mrr and high_usage:
        name = "High-Value Power Users"

    elif high_usage and not high_mrr:
        name = "Active Growth Accounts"

    elif high_tickets and not high_satisfaction:
        name = "Support-Heavy Accounts"

    else:
        name = "Low-Engagement Accounts"

    segment_names[cluster] = name


print("\nCluster → Segment Name:")
for cluster, name in segment_names.items():
    print(cluster, "→", name)


    # ============================================================
# 15. ADD BEHAVIOUR-BASED SEGMENT NAME
# ============================================================

customer_features["Segment"] = (
    customer_features["Cluster"]
    .map(segment_names)
)

print("\nCustomer segmentation result:")
print(
    customer_features[
        [
            "CustomerID",
            "CompanyName",
            "Cluster",
            "Segment"
        ]
    ].head(20)
)


# ============================================================
# 16. FINAL SEGMENT SUMMARY
# ============================================================

segment_summary = (
    customer_features
    .groupby("Segment")
    .agg(
        Customer_Count=("CustomerID", "count"),
        Total_MRR=("Total_MRR", "sum"),
        Avg_MRR=("Total_MRR", "mean"),
        Avg_Logins=("Total_Logins", "mean"),
        Avg_ActiveUsers=("Total_ActiveUsers", "mean"),
        Avg_APICalls=("Total_APICalls", "mean"),
        Avg_SessionMinutes=("Total_SessionMinutes", "mean"),
        Avg_Tickets=("Ticket_Count", "mean"),
        Avg_Satisfaction=("Avg_Satisfaction", "mean")
    )
    .round(2)
    .reset_index()
)


# ============================================================
# 17. RETENTION RECOMMENDATIONS
# ============================================================

recommendations = {
    "High-Value Power Users":
        "Provide proactive account management, premium support and expansion opportunities to protect high-value revenue.",

    "Active Growth Accounts":
        "Encourage adoption of additional features and plans because strong usage indicates potential for account expansion.",

    "Support-Heavy Accounts":
        "Investigate recurring support issues and provide targeted onboarding or product assistance to improve satisfaction.",

    "Low-Engagement Accounts":
        "Use re-engagement campaigns, onboarding reminders and feature education to increase product usage."
}

# Add recommendation directly to customer-level data
customer_features["Retention_Recommendation"] = (
    customer_features["Segment"].map(recommendations)
)

print("\nSEGMENT RETENTION RECOMMENDATIONS")

print(
    customer_features[
        [
            "CustomerID",
            "CompanyName",
            "Segment",
            "Retention_Recommendation"
        ]
    ].head(20)
)


# ============================================================
# 18. FINAL CUSTOMER-LEVEL OUTPUT
# ============================================================

final_columns = [
    "CustomerID",
    "CompanyName",
    "Total_MRR",
    "Total_Seats",
    "Total_Logins",
    "Total_ActiveUsers",
    "Total_APICalls",
    "Total_SessionMinutes",
    "Ticket_Count",
    "Avg_Satisfaction",
    "Cluster",
    "Segment",
    "Retention_Recommendation"
]

final_segmentation = customer_features[
    final_columns
].copy()

print("\nFINAL CUSTOMER SEGMENTATION")
print(final_segmentation.head(20))



# ============================================================
# 19. SAVE MODULE 8 OUTPUTS
# ============================================================

final_segmentation.to_csv(
    "module8_customer_segmentation.csv",
    index=False
)

cluster_profile.to_csv(
    "module8_cluster_profile.csv"
)

segment_summary.to_csv(
    "module8_segment_summary.csv",
    index=False
)

elbow_table.to_csv(
    "module8_elbow_results.csv",
    index=False
)

print("\nModule 8 files saved successfully.")

print("\n1. module8_customer_segmentation.csv")
print("2. module8_cluster_profile.csv")
print("3. module8_segment_summary.csv")
print("4. module8_elbow_results.csv")
