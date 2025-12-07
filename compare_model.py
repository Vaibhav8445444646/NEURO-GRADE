import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("student_performance_updated_1000.csv")

# Drop irrelevant columns
df.drop(['StudentID', 'Name'], axis=1, inplace=True)

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Encode categorical and boolean variables
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['ParentalSupport'] = df['ParentalSupport'].astype('category').cat.codes
df['Online Classes Taken'] = df['Online Classes Taken'].fillna(False).astype(bool).astype(int)
df['ExtracurricularActivities'] = df['ExtracurricularActivities'].fillna(False).astype(bool).astype(int)

# Normalize numerical features
scaler = MinMaxScaler()
df[['Study Hours', 'Attendance (%)']] = scaler.fit_transform(df[['Study Hours', 'Attendance (%)']])

# Drop any remaining NaNs
df.dropna(inplace=True)

# Split features and target
X = df.drop('FinalGrade', axis=1)
y = df['FinalGrade']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Neural Network": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
}

# Train and evaluate
r2_scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2_scores[name] = r2_score(y_test, y_pred)

# Plot comparison
plt.figure(figsize=(8, 5))
plt.bar(r2_scores.keys(), r2_scores.values(), color=['skyblue', 'lightgreen', 'salmon'])
plt.ylabel("R² Score")
plt.title("Model Comparison on Student Score Prediction")
plt.ylim(-0.5, 1.0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()