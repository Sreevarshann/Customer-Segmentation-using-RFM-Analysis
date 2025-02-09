import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_cleaning import load_data, clean_data, save_cleaned_data
from rfm_functions import calculate_rfm, assign_rfm_scores
from clustering import standardize_data, perform_kmeans, find_optimal_clusters
import logging
from pathlib import Path

class CustomerSegmentationPipeline:
    def __init__(self, input_path, output_dir):
        """
        Initialize the pipeline with input and output paths.
        
        Args:
            input_path (str): Path to input CSV file
            output_dir (str): Directory for output files
        """
        self.input_path = input_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def run_pipeline(self):
        """Execute the complete customer segmentation pipeline."""
        try:
            # 1. Data Loading and Cleaning
            self.logger.info("Starting data preprocessing...")
            df = load_data(self.input_path)
            df_cleaned = clean_data(df)
            save_cleaned_data(df_cleaned, self.output_dir / 'cleaned_data.csv')
            
            # 2. Calculate TotalPrice
            df_cleaned['TotalPrice'] = df_cleaned['Quantity'] * df_cleaned['UnitPrice']
            
            # 3. RFM Calculation
            self.logger.info("Calculating RFM metrics...")
            rfm_df = calculate_rfm(df_cleaned)
            
            # 4. RFM Scoring
            self.logger.info("Assigning RFM scores...")
            quantiles = rfm_df.quantile(q=[0.25, 0.5, 0.75])
            rfm_scored = assign_rfm_scores(rfm_df, quantiles)
            rfm_scored.to_csv(self.output_dir / 'rfm_scored.csv', index=True)
            
            # 5. Clustering
            self.logger.info("Performing clustering analysis...")
            rfm_normalized = standardize_data(rfm_scored[['Recency', 'Frequency', 'Monetary']])
            
            # Find optimal number of clusters
            wcss = find_optimal_clusters(rfm_normalized, 10)
            self.plot_elbow_curve(wcss)
            
            # Perform clustering with optimal number of clusters (k=3 based on analysis)
            clusters = perform_kmeans(rfm_normalized, n_clusters=3)
            rfm_scored['Cluster'] = clusters
            
            # 6. Save final results
            self.logger.info("Saving final results...")
            rfm_scored.to_csv(self.output_dir / 'final_segments.csv', index=True)
            
            # 7. Generate visualizations
            self.logger.info("Generating visualizations...")
            self.create_visualizations(rfm_scored)
            
            # 8. Generate segment profiles
            self.generate_segment_profiles(rfm_scored)
            
            self.logger.info("Pipeline completed successfully!")
            return rfm_scored
            
        except Exception as e:
            self.logger.error(f"Error in pipeline execution: {str(e)}")
            raise

    def plot_elbow_curve(self, wcss):
        """Plot the elbow curve for clustering."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(wcss) + 1), wcss, marker='o')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')
        plt.savefig(self.output_dir / 'elbow_curve.png')
        plt.close()

    def create_visualizations(self, data):
        """Create and save visualizations."""
        # RFM distributions by cluster
        plt.figure(figsize=(15, 5))
        
        metrics = ['Recency', 'Frequency', 'Monetary']
        for i, metric in enumerate(metrics, 1):
            plt.subplot(1, 3, i)
            sns.boxplot(x='Cluster', y=metric, data=data)
            plt.title(f'{metric} by Cluster')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cluster_distributions.png')
        plt.close()

    def generate_segment_profiles(self, data):
        """Generate and save segment profiles."""
        profiles = data.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'CustomerID': 'count'
        }).round(2)
        
        profiles.rename(columns={'CustomerID': 'Count'}, inplace=True)
        profiles.to_csv(self.output_dir / 'segment_profiles.csv')
        
        # Log segment profiles
        self.logger.info("\nSegment Profiles:")
        for cluster in profiles.index:
            self.logger.info(f"\nCluster {cluster}:")
            self.logger.info(f"Number of customers: {profiles.loc[cluster, 'Count']}")
            self.logger.info(f"Average Recency: {profiles.loc[cluster, 'Recency']:.2f} days")
            self.logger.info(f"Average Frequency: {profiles.loc[cluster, 'Frequency']:.2f} orders")
            self.logger.info(f"Average Monetary Value: ${profiles.loc[cluster, 'Monetary']:.2f}")

def main():
    # Define paths with the specified input file
    input_path = "/Users/sreevarshansathiyamurthy/Downloads/Customer-Segmentation.csv"
    output_dir = "output"
    
    # Create and run pipeline
    try:
        pipeline = CustomerSegmentationPipeline(input_path, output_dir)
        results = pipeline.run_pipeline()
        print("Customer segmentation analysis completed successfully!")
    except Exception as e:
        print(f"Error running pipeline: {str(e)}")

if __name__ == "__main__":
    main()