
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import os

def calculate_correlations(input_csv_path, output_csv_path):
    """
    Calculates Pearson and Spearman correlation coefficients between 'Detection_Rate'
    and other features in the input CSV, then saves the results to an output CSV.
    """
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_csv_path}")
        return

    if 'Detection_Rate' not in df.columns:
        print("Error: 'Detection_Rate' column not found in the input CSV.")
        return

    correlation_results = []
    detection_rate = df['Detection_Rate']

    for column in df.columns:
        if column in ['Marker_ID', 'Detection_Rate']:
            continue  # Skip Marker_ID and Detection_Rate itself

        feature_data = df[column]

        # Ensure feature data is numeric and handle potential NaNs for correlation
        # Drop rows where either detection_rate or feature_data is NaN for a fair comparison
        temp_df = pd.DataFrame({'detection_rate': detection_rate, 'feature_data': feature_data}).dropna()

        if len(temp_df) < 2:
            print(f"Warning: Not enough data points to calculate correlation for {column}. Skipping.")
            continue

        # Pearson correlation
        pearson_corr, pearson_p = pearsonr(temp_df['detection_rate'], temp_df['feature_data'])

        # Spearman correlation
        spearman_corr, spearman_p = spearmanr(temp_df['detection_rate'], temp_df['feature_data'])

        correlation_results.append({
            'Feature': column,
            'Pearson_Correlation': pearson_corr,
            'Pearson_P_Value': pearson_p,
            'Spearman_Correlation': spearman_corr,
            'Spearman_P_Value': spearman_p
        })

    results_df = pd.DataFrame(correlation_results)

    try:
        results_df.to_csv(output_csv_path, index=False)
        print(f"Correlation results saved to {output_csv_path}")
    except Exception as e:
        print(f"Error saving results to {output_csv_path}: {e}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, 'marker_analyze_2.csv')
    output_file = os.path.join(script_dir, 'marker_pirson.csv')
    calculate_correlations(input_file, output_file)
