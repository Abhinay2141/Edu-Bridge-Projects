"""
ML Transaction Fraud Detection System
Author: College Assignment Submission
Description: A Random Forest Classifier to identify fraudulent transactions based on user metadata.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def main():
    print("==================================================")
    print("          PROJECT 5: ML - FRAUD DETECTION         ")
    print("==================================================")
    
    # 1. Load Dataset
    data_path = "dataset.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found in local project directory.")
        return
        
    print(f"Loading dataset: {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Dataset Loaded Successfully! Size: {df.shape[0]} records.")
    
    # Show baseline fraud ratio
    fraud_counts = df["Is_Fraud"].value_counts()
    print("\nTransaction Classes Distribution:")
    print(f" - Legitimate (Class 0): {fraud_counts[0]} ({fraud_counts[0]/len(df):.2%})")
    print(f" - Fraudulent (Class 1): {fraud_counts[1]} ({fraud_counts[1]/len(df):.2%})")
    
    print("\nFirst 5 Records:")
    print(df.head())
    
    # 2. Data Preprocessing & Categorical Encoding
    print("\nPerforming One-Hot Encoding on categorical variables ('Transaction_Type', 'Device_Type')...")
    # One-hot encode the categorical text features
    df_encoded = pd.get_dummies(df, columns=["Transaction_Type", "Device_Type"], dtype=int)
    print("Encoding completed! Encoded Columns:")
    print(list(df_encoded.columns))
    
    # 3. Feature Selection & Target Splitting
    X = df_encoded.drop(columns=["Is_Fraud"])
    y = df_encoded["Is_Fraud"]
    
    # 4. Train-Test Split (using stratify to ensure balanced target class ratios)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"\nTrain set size: {len(X_train)} samples | Test set size: {len(X_test)} samples")
    
    # 5. Model Training: Random Forest Classifier
    print("\nTraining Random Forest Classifier (100 estimators)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    print("Model training completed successfully!")
    
    # 6. Feature Importance Evaluation
    print("\nEvaluating Feature Importances (MDI):")
    importances = model.feature_importances_
    feat_names = list(X.columns)
    importance_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)
    print(importance_df.head(6))
    
    # 7. Model Evaluation
    print("\nEvaluating model performance on Test Set...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n" + "="*40)
    print("               MODEL PERFORMANCE")
    print("="*40)
    print(f"Accuracy Score : {accuracy:.2%}")
    print(f"Precision Score: {precision:.2%}  (Out of predicted fraud, how many were actual fraud)")
    print(f"Recall Score   : {recall:.2%}  (Out of actual fraud, how many were detected)")
    print(f"F1-Score       : {f1:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print("             Predicted Legit   Predicted Fraud")
    print(f"Actual Legit     {cm[0][0]:<5}             {cm[0][1]:<5}    (TN / FP)")
    print(f"Actual Fraud     {cm[1][0]:<5}             {cm[1][1]:<5}    (FN / TP)")
    print("="*40)
    
    # 8. Sample Predictions on Custom Inputs
    print("\nTesting Custom Transaction Inference:")
    
    # Prepare dummy samples with same structures
    custom_samples = [
        {
            "Amount": 45.0, "Time": 14, "Account_Age": 24,
            "Transaction_Type_Online": 0, "Transaction_Type_Retail": 1, "Transaction_Type_Transfer": 0,
            "Device_Type_Desktop": 1, "Device_Type_Mobile": 0, "Device_Type_Tablet": 0
        },
        {
            "Amount": 2800.0, "Time": 3, "Account_Age": 1,
            "Transaction_Type_Online": 0, "Transaction_Type_Retail": 0, "Transaction_Type_Transfer": 1,
            "Device_Type_Desktop": 0, "Device_Type_Mobile": 1, "Device_Type_Tablet": 0
        }
    ]
    
    custom_df = pd.DataFrame(custom_samples)
    # Ensure correct columns order matches X
    custom_df = custom_df[X.columns]
    
    preds = model.predict(custom_df)
    probs = model.predict_proba(custom_df)
    
    inputs_desc = [
        "Retail purchase ($45.00) at 2 PM, Desktop device, Account age: 2 years",
        "Late night Transfer ($2800.00) at 3 AM, Mobile device, Account age: 1 month"
    ]
    
    for desc, pred, prob in zip(inputs_desc, preds, probs):
        print(f"\nTransaction: {desc}")
        label = "FRAUD" if pred == 1 else "LEGITIMATE"
        conf = prob[pred]
        print(f"Predicted Class: {label} (Confidence: {conf:.2%})")
        
    print("\n==================================================")
    print("Explanation:")
    print("The system models fraud classification using Random Forest (Decision Tree ensembles).")
    print("Categorical strings are one-hot encoded to numerical binary variables.")
    print("Precision measures safety from false positives (protects user from false locks).")
    print("Recall measures safety from false negatives (protects bank from fraud leakage).")
    print("Random Forest balances these objectives by splitting samples across multiple trees.")
    print("==================================================")

if __name__ == "__main__":
    main()
