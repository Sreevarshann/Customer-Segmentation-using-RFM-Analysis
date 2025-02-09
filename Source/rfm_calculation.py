# Load cleaned data
df = pd.read_csv('data/cleaned_data.csv')

# Calculate Recency
latest_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
df['Recency'] = (latest_date - df['InvoiceDate']).dt.days

# Calculate Frequency
frequency_df = df.groupby('CustomerID').agg({'InvoiceNo': 'nunique'}).rename(columns={'InvoiceNo': 'Frequency'})

# Calculate Monetary
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
monetary_df = df.groupby('CustomerID').agg({'TotalPrice': 'sum'}).rename(columns={'TotalPrice': 'Monetary'})

# Merge RFM metrics
rfm_df = df.groupby('CustomerID').agg(Recency=('InvoiceDate', lambda x: (latest_date - x.max()).days),
                                       Frequency=('InvoiceNo', 'nunique'),
                                       Monetary=('TotalPrice', 'sum')).reset_index()

rfm_df.to_csv('data/rfm_data.csv', index=False)
