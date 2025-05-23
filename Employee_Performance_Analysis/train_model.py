import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("employee_data.csv")

# Extract relevant features
df_cleaned = df[['Total Hours Worked', 'Task Completed (%)']]
df_cleaned['Productivity'] = np.where(df_cleaned['Total Hours Worked'] >= 6, 1, 0)

# Split data into training and testing sets
X = df_cleaned[['Total Hours Worked', 'Task Completed (%)']]
y = df_cleaned['Productivity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Gradient Boosting model
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model and scaler
with open("employee_performance_model.pkl", 'wb') as f:
    pickle.dump(model, f)
with open("scaler.pkl", 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Model trained and saved successfully.")
