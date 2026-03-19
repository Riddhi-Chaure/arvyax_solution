import pandas as pd
import numpy as np

def compute_uncertainty(predictions_df):
    """
    Compute confidence scores and flag low-confidence predictions.
    
    Confidence: Scaled distance from 0.5 (max uncertainty). 
    A probability of 0.5 means a 0.0 confidence score.
    A probability of 0.0 or 1.0 means a 1.0 confidence score.
    """
    # Using normalized distance from 0.5
    # For binary classification: 2 * abs(prob - 0.5)
    # 0.5 -> 0.0 (uncertain)
    # 1.0 -> 1.0 (certain)
    # 0.0 -> 1.0 (certain)
    
    probs = predictions_df['Probability']
    confidence_scores = 2 * np.abs(probs - 0.5)
    
    # Flag cases with confidence less than a certain threshold (e.g., 0.3)
    # A confidence score of 0.3 means prob between 0.35 and 0.65.
    uncertain_flag = (confidence_scores < 0.3).astype(int)
    
    results = predictions_df.copy()
    results['ConfidenceScore'] = confidence_scores
    results['IsUncertain'] = uncertain_flag
    
    return results

def main():
    """Enhance predictions with uncertainty analysis."""
    input_path = 'outputs/predictions.csv'
    
    if not pd.api.types.is_file_like(input_path):
        import os
        if not os.path.exists(input_path):
            print(f"Error: {input_path} not found. Run predict.py first.")
            return

    # Load previously generated predictions
    preds_df = pd.read_csv(input_path)
    
    # Apply uncertainty logic
    enhanced_df = compute_uncertainty(preds_df)
    
    # Overwrite the predictions.csv with enhanced fields
    enhanced_df.to_csv(input_path, index=False)
    print("Uncertainty analysis complete. Updated outputs/predictions.csv with ConfidenceScore and IsUncertain.")

if __name__ == "__main__":
    main()
