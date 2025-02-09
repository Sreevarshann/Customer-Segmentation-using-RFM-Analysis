# Load RFM data
rfm_df = pd.read_csv('data/rfm_data.csv')

# Calculate RFM quartiles
quantiles = rfm_df.quantile(q=[0.25, 0.5, 0.75])

# Function to assign RFM scores
def rfm_score(x, p, d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.5]:
        return 2
    elif x <= d[p][0.75]:
        return 3
    else:
        return 4

# Assigning scores
rfm_df['R_Score'] = rfm_df['Recency'].apply(rfm_score, args=('Recency', quantiles,))
rfm_df['F_Score'] = rfm_df['Frequency'].apply(rfm_score, args=('Frequency', quantiles,))
rfm_df['M_Score'] = rfm_df['Monetary'].apply(rfm_score, args=('Monetary', quantiles,))

# Combine scores
rfm_df['RFM_Segment'] = rfm_df['R_Score'].astype(str) + rfm_df['F_Score'].astype(str) + rfm_df['M_Score'].astype(str)

rfm_df.to_csv('data/segmented_rfm.csv', index=False)
