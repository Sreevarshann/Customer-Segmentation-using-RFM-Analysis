import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/customer_data.csv', encoding='ISO-8859-1')

# Inspect data for missing values
print(df.isnull().sum())

# Fill missing 'CustomerID' with mode (most frequent value)
df['CustomerID'].fillna(df['CustomerID'].mode()[0], inplace=True)

# Drop rows where 'Description' is missing
df.dropna(subset=['Description'], inplace=True)

# Convert 'InvoiceDate' to datetime format
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Save the cleaned data
df.to_csv('data/cleaned_data.csv', index=False)
