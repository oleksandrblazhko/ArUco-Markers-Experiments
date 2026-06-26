import os
import json
import joblib
import pandas as pd
import numpy as np
import cv2
import cv2.aruco as aruco

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from features import get_marker_bits, extract_features

def main():
    # 1. Load data
    json_path = "../marker_detection_rate.json"
    if not os.path.exists(json_path):
        json_path = "../utils/marker_detection_rate.json"
    if not os.path.exists(json_path):
        json_path = "marker_detection_rate.json"
    
    if not os.path.exists(json_path):
        # Try finding anywhere in the workspace
        print("Looking for marker_detection_rate.json...")
        for root, dirs, files in os.walk(".."):
            if "marker_detection_rate.json" in files:
                json_path = os.path.join(root, "marker_detection_rate.json")
                break
                
    if not os.path.exists(json_path):
        print(f"Error: Could not find marker_detection_rate.json. Please place it in the working directory.")
        return
        
    print(f"Loading data from {json_path}...")
    with open(json_path, "r") as f:
        data = json.load(f)
        
    # Predefined dictionary 4x4_1000
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    
    # 2. Extract features
    rows = []
    for item in data:
        marker_id = item.get("marker_ID") or item.get("marker_id")
        dr = item.get("Detection_Rate") or item.get("detection_rate") or item.get("Detection_rate")
        if marker_id is None or dr is None:
            continue
            
        try:
            bits = get_marker_bits(dictionary, marker_id)
            feats = extract_features(bits)
            feats["MarkerID"] = marker_id
            feats["DetectionRate"] = dr
            rows.append(feats)
        except Exception as e:
            print(f"Warning: failed to process ID {marker_id}: {e}")
            
    df = pd.DataFrame(rows)
    print(f"Extracted features for {len(df)} markers.")
    
    # Define features and target (DetectionRate >= 95% is class 1, otherwise 0)
    features_list = [
        "Ones", "OnesDev", "Transitions", "Chessboard", "Isolated", 
        "Cmax", "Homogeneous", "Perimeter", "EdgeWhiteCells", "Components",
        "BlackComponents", "TotalComponents"
    ]
    
    X = df[features_list]
    y = (df["DetectionRate"] >= 95.0).astype(int)
    
    # 3. Model selection using 5-fold cross-validation ROC-AUC
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42))
    ])
    
    rf_pipeline = Pipeline([
        ("clf", RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42))
    ])
    
    auc_lr = cross_val_score(lr_pipeline, X, y, cv=5, scoring="roc_auc").mean()
    auc_rf = cross_val_score(rf_pipeline, X, y, cv=5, scoring="roc_auc").mean()
    
    print(f"Logistic Regression 5-fold CV ROC-AUC: {auc_lr:.4f}")
    print(f"Random Forest 5-fold CV ROC-AUC:       {auc_rf:.4f}")
    
    if auc_rf > auc_lr:
        best_pipeline = rf_pipeline
        model_type = "RandomForestClassifier"
        print(f"Selected Model: Random Forest (ROC-AUC: {auc_rf:.4f})")
    else:
        best_pipeline = lr_pipeline
        model_type = "LogisticRegression"
        print(f"Selected Model: Logistic Regression (ROC-AUC: {auc_lr:.4f})")
        
    # Fit the best model on the entire dataset
    best_pipeline.fit(X, y)
    
    # 4. Save model
    model_file = "model.pkl"
    joblib.dump({
        "pipeline": best_pipeline,
        "features": features_list,
        "model_type": model_type
    }, model_file)
    print(f"Saved best model and metadata to '{model_file}'.")
    
    # If Logistic Regression, show coefficient importances for explainability
    if model_type == "LogisticRegression":
        clf = best_pipeline.named_steps["clf"]
        scaler = best_pipeline.named_steps["scaler"]
        print("\nLogistic Regression Coefficients (standardized features):")
        for name, coef in zip(features_list, clf.coef_[0]):
            print(f"  {name:<16}: {coef:.4f}")
        print(f"  Intercept       : {clf.intercept_[0]:.4f}")
    else:
        clf = best_pipeline.named_steps["clf"]
        print("\nRandom Forest Feature Importances:")
        for name, imp in zip(features_list, clf.feature_importances_):
            print(f"  {name:<16}: {imp:.4f}")

if __name__ == "__main__":
    main()
