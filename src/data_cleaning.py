import pandas as pd

def load_data(file_path):
    """Load data from CSV file."""
    return pd.read_csv(file_path, encoding='ISO-8859-1')

def clean_data(df):
    """Clean and preprocess dataframe."""
    df.dropna(subset=['CustomerID', 'Description'], inplace=True)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['CustomerID'] = df['CustomerID'].astype(str)
    return df

def save_cleaned_data(df, output_path):
    """Save cleaned data to CSV."""
    df.to_csv(output_path, index=False)
