# Customer Segmentation using RFM Analysis

## Introduction
This comprehensive project utilizes Recency, Frequency, and Monetary (RFM) analysis to segment customers of an e-commerce dataset. The RFM model is a proven marketing model for behavior-based customer segmentation. It groups customers based on their transaction history – how recently, how often, and how much did they buy. This analysis helps businesses tailor their marketing efforts to different customer segments to increase customer retention and optimize profitability.

## Project Objectives
The main objectives of this project are to:
- **Clean and preprocess data**: Prepare the raw data for analysis by cleaning and structuring.
- **Perform RFM analysis**: Compute RFM metrics for each customer to understand their purchasing habits.
- **Segment customers**: Use clustering techniques to segment customers based on their RFM scores.
- **Profile segments**: Analyze and profile each customer segment based on their RFM characteristics.
- **Visualize findings**: Create visual representations of the data to better understand the distribution of customers.
- **Develop targeted strategies**: Provide actionable marketing strategies tailored to each customer segment.

## Methodology
### Data Preprocessing
The dataset undergoes a rigorous cleaning process where missing values are imputed, and anomalies are corrected to ensure data quality and reliability for the analysis.

### RFM Calculation
- **Recency (R)**: Days since last purchase
- **Frequency (F)**: Total number of transactions
- **Monetary (M)**: Total money spent

### Customer Segmentation
Applying K-Means clustering, we identify optimal clusters and segment the customer base into meaningful groups.

### Analysis and Insights
Post-segmentation, we delve deep into each segment to understand and document the distinctive characteristics and behaviors exhibited by the customers in each group.

## Installation
This project requires Python 3.x and several libraries.
```bash
# Clone the repository
git clone https://github.com/Sreevarshann/customer-segmentation-rfm.git
# Navigate to the project directory
cd customer-segmentation-rfm
# Install required Python packages
pip install -r requirements.txt
