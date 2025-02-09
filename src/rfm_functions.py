import pandas as pd

def calculate_rfm(df):
    """Calculate RFM metrics."""
    latest_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (latest_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'})
    return rfm

def assign_rfm_scores(rfm, quantiles):
    """Assign RFM scores based on quantiles."""
    rfm['R_Score'] = pd.qcut(rfm['Recency'], q=[0, .25, .5, .75, 1], labels=[4, 3, 2, 1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'], q=[0, .25, .5, .75, 1], labels=[1, 2, 3, 4])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=[0, .25, .5, .75, 1], labels=[1, 2, 3, 4])
    rfm['RFM_Segment'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    return rfm
