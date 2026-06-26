import os
import sys
import argparse
import json
import joblib
import pandas as pd
import numpy as np
import cv2
import cv2.aruco as aruco
import re

from features import get_marker_bits, extract_features

def parse_ids(input_str):
    """
    Parses strings like '0-100,105,107,110-120' into a sorted list of unique integers.
    """
    ids = set()
    tokens = re.split(r'[,\s]+', input_str.strip())
    for token in tokens:
        if not token:
            continue
        if '-' in token:
            parts = token.split('-')
            if len(parts) == 2:
                try:
                    start = int(parts[0])
                    end = int(parts[1])
                    if start <= end:
                        ids.update(range(start, end + 1))
                    else:
                        ids.update(range(end, start + 1))
                except ValueError:
                    print(f"Warning: could not parse range '{token}'", file=sys.stderr)
            else:
                print(f"Warning: invalid range format '{token}'", file=sys.stderr)
        else:
            try:
                ids.add(int(token))
            except ValueError:
                print(f"Warning: could not parse ID '{token}'", file=sys.stderr)
    return sorted(list(ids))

def main():
    parser = argparse.ArgumentParser(description="Evaluate ArUco markers using a trained ML model.")
    parser.add_argument("ids", type=str, help="IDs to evaluate, e.g. '0-10' or '0,1,2,5-8'")
    parser.add_argument("--model", type=str, default="model.pkl", help="Path to the saved model.pkl file. Default is 'model.pkl'.")
    parser.add_argument("--dict", type=str, default="4X4_250", help="ArUco predefined dictionary. Default is 4X4_250.")
    parser.add_argument("--limit", type=int, default=None, help="Limit output to top N candidates.")
    parser.add_argument("--output", type=str, default=None, help="Path to save the results (supports .csv, .json, .txt/log).")
    
    args = parser.parse_args()
    
    # 1. Load model
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found. Please train the model first by running train.py.", file=sys.stderr)
        sys.exit(1)
        
    model_data = joblib.load(args.model)
    pipeline = model_data["pipeline"]
    features_list = model_data["features"]
    model_type = model_data["model_type"]
    
    # 2. Resolve dictionary
    dict_name = f"DICT_{args.dict.upper()}"
    try:
        dict_id = getattr(aruco, dict_name)
    except AttributeError:
        try:
            dict_id = getattr(aruco, args.dict)
        except AttributeError:
            print(f"Error: Unknown dictionary '{args.dict}'.", file=sys.stderr)
            sys.exit(1)
            
    dictionary = aruco.getPredefinedDictionary(dict_id)
    
    # 3. Parse IDs
    target_ids = parse_ids(args.ids)
    if not target_ids:
        print("Error: No valid marker IDs specified.", file=sys.stderr)
        sys.exit(1)
        
    # 4. Generate features
    rows = []
    for marker_id in target_ids:
        try:
            bits = get_marker_bits(dictionary, marker_id)
            feats = extract_features(bits)
            feats["MarkerID"] = marker_id
            rows.append(feats)
        except Exception as e:
            print(f"Warning: failed to process ID {marker_id}: {e}", file=sys.stderr)
            
    if not rows:
        print("Error: No markers were successfully processed.", file=sys.stderr)
        sys.exit(1)
        
    df = pd.DataFrame(rows)
    X = df[features_list]
    
    # 5. Predict probabilities (Q-scores)
    # predict_proba returns probability estimates [prob_class_0, prob_class_1]
    df["Q"] = pipeline.predict_proba(X)[:, 1]
    
    # Apply hard constraint: BlackComponents must be exactly 1.
    # If a marker has fragmented black components (BlackComponents > 1), its Q score is forced to 0.0
    df["Q"] = np.where(df["BlackComponents"] == 1, df["Q"], 0.0)
    
    # Sort by Q score descending
    df = df.sort_values("Q", ascending=False)
    
    if args.limit is not None:
        df = df.head(args.limit)
        
    # 6. Prepare and Print results
    output_lines = []
    output_lines.append(f"\n--- Marker Quality Evaluation Results (Model: {model_type}) ---")
    output_lines.append(f"{'ID':<6} | {'Q Score':<8} | {'Ones':<4} | {'T':<4} | {'Chess':<5} | {'Isolated':<4} | {'Cmax':<4} | {'BlackComp':<9} | {'EdgeWhite':<9}")
    output_lines.append("-" * 85)
    for idx, row in df.iterrows():
        output_lines.append(f"{int(row['MarkerID']):<6} | {row['Q']:<8.4f} | {int(row['Ones']):<4} | {int(row['Transitions']):<4} | {int(row['Chessboard']):<5} | {int(row['Isolated']):<4} | {int(row['Cmax']):<4} | {int(row['BlackComponents']):<9} | {int(row['EdgeWhiteCells']):<9}")
        
    ordered_ids = [str(int(mid)) for mid in df["MarkerID"]]
    output_lines.append("\n--- Ordered IDs (Descending by ML Quality) ---")
    output_lines.append(",".join(ordered_ids))
    
    # Print to screen
    for line in output_lines:
        print(line)
        
    # 7. Save results if --output is specified
    if args.output:
        # Create output directory if it doesn't exist
        out_dir = os.path.dirname(args.output)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        ext = os.path.splitext(args.output)[1].lower()
        if ext == '.csv':
            df_out = df[["MarkerID", "Q", "Ones", "Transitions", "Chessboard", "Isolated", "Cmax", "BlackComponents", "EdgeWhiteCells"]].copy()
            df_out.rename(columns={"Q": "Q_Score", "Transitions": "T", "Chessboard": "Chess", "BlackComponents": "BlackComp", "EdgeWhiteCells": "EdgeWhite"}, inplace=True)
            df_out.to_csv(args.output, index=False)
            print(f"\nResults saved to CSV: {args.output}")
        elif ext == '.json':
            df_out = df[["MarkerID", "Q", "Ones", "Transitions", "Chessboard", "Isolated", "Cmax", "BlackComponents", "EdgeWhiteCells"]].copy()
            df_out.rename(columns={"Q": "Q_Score", "Transitions": "T", "Chessboard": "Chess", "BlackComponents": "BlackComp", "EdgeWhiteCells": "EdgeWhite"}, inplace=True)
            
            # Convert table rows to a list of dictionaries with standard Python types
            table_data = df_out.to_dict(orient="records")
            for row_dict in table_data:
                for k, v in row_dict.items():
                    if isinstance(v, (np.integer, np.int64)):
                        row_dict[k] = int(v)
                    elif isinstance(v, (np.floating, np.float64)):
                        row_dict[k] = float(v)
            
            # List of ordered marker IDs as integers
            ordered_ids_ints = [int(mid) for mid in df["MarkerID"]]
            
            # Create customized JSON representation
            # Format each row dict on a single line
            table_lines = []
            for row in table_data:
                row_str = json.dumps(row)
                table_lines.append("        " + row_str)
                
            table_json = "[\n" + ",\n".join(table_lines) + "\n    ]"
            ordered_ids_json = json.dumps(ordered_ids_ints)
            
            json_string = (
                "{\n"
                f'    "table": {table_json},\n'
                f'    "ordered_ids": {ordered_ids_json}\n'
                "}"
            )
            
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(json_string + '\n')
            print(f"\nResults saved to JSON: {args.output}")
        else:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write('\n'.join(output_lines) + '\n')
            print(f"\nResults saved to text file: {args.output}")

if __name__ == "__main__":
    main()
