import pandas as pd
import os

def determine_actions(enriched_df):
    """
    Decide business actions based on predictions and uncertainty.
    
    Rule-based Logic:
    1. If Churn probability is high (>= 0.7) and Confidence is high, Recommend 'Retention Call'.
    2. If Churn probability is very low (<= 0.3) and Confidence is high, Recommend 'Do Nothing'.
    3. If Prediction is uncertain (IsUncertain == 1), Recommend 'Human Review / Send Survey'.
    4. Default Recommendation: 'Monitor'.
    """
    results = enriched_df.copy()
    
    # Simple logic mapping
    actions = []
    
    for idx, row in results.iterrows():
        prob = row['Probability']
        is_uncertain = row['IsUncertain']
        
        if is_uncertain == 1:
            actions.append("Human Review / Send Survey")
        elif prob >= 0.7:
            actions.append("Direct Retention Call")
        elif prob <= 0.3:
            actions.append("Auto-Approve / No Action")
        else:
            actions.append("Monitor Engagement")
            
    results['RecommendedAction'] = actions
    return results

def main():
    """Run the decision engine on predictions."""
    input_path = 'outputs/predictions.csv'
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Ensure predict.py and uncertainty.py have run.")
        return

    # Load enriched predictions
    enriched_df = pd.read_csv(input_path)
    
    # Deciding Final Action
    final_output = determine_actions(enriched_df)
    
    # Save final output
    output_path = 'outputs/predictions.csv'
    final_output.to_csv(output_path, index=False)
    
    print(f"Decision and recommended actions saved to {output_path}.")
    print("\nSummary of Actions Recommended:")
    print(final_output['RecommendedAction'].value_counts())

if __name__ == "__main__":
    main()
