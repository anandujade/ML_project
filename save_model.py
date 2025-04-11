import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv('malware_dataset (4).csv')

# Preprocessing (assuming necessary steps are included)
X = df.drop(columns=['Target_Compromised'])  # Features
y = df['Target_Compromised']  # Target

# Splitting data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the trained model
model_filename = "rf_model.pkl"
joblib.dump(rf_model, model_filename)
print(f"Model saved as {model_filename}")
