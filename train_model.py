import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ---------------------------
# Load dataset
# ---------------------------
df = pd.read_csv("heart_disease_health_indicators.csv")

# ---------------------------
# Split features & target
# ---------------------------
X = df.drop("HeartDiseaseorAttack", axis=1)
y = df["HeartDiseaseorAttack"]

# ---------------------------
# Train-test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Scaling
# ---------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ---------------------------
# Model
# ---------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# ---------------------------
# Save model and scaler
# ---------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ model.pkl and scaler.pkl saved successfully!")

