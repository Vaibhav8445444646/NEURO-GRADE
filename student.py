import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import sys
import glob

def find_csv_path():
    """Locate a CSV path to load."""
    if len(sys.argv) > 1:
        candidate = sys.argv[1]
        if os.path.isabs(candidate) and os.path.exists(candidate):
            return candidate
        else:
            rel = os.path.join(os.getcwd(), candidate)
            if os.path.exists(rel):
                return rel
        raise FileNotFoundError(f"Provided path does not exist: {candidate}")

    common_names = [
        'StudentPerformance.csv',
        'student_performance_updated_1000.csv',
        'student_performance.csv'
    ]
    here = os.getcwd()
    for name in common_names:
        p = os.path.join(here, name)
        if os.path.exists(p):
            return p

    csvs = glob.glob(os.path.join(here, '*.csv'))
    if csvs:
        csvs.sort()
        return csvs[0]

    files = os.listdir(here)
    raise FileNotFoundError(
        f"No CSV file found in {here}. Files present: {files}.\n"
        "You can provide a path as the first argument: python student.py <path/to/file.csv>"
    )

# Load dataset
csv_path = find_csv_path()
print(f"Loading CSV from: {csv_path}")
df = pd.read_csv(csv_path)
print("Original Data:")
print(df.head())

# Drop irrelevant columns
df.drop(['StudentID', 'Name'], axis=1, inplace=True)

# Handle missing values (numeric only)
df.fillna(df.mean(numeric_only=True), inplace=True)

# Encode categorical and boolean variables
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['ParentalSupport'] = df['ParentalSupport'].astype('category').cat.codes
df['Online Classes Taken'] = df['Online Classes Taken'].fillna(False).astype(bool).astype(int)

# Normalize numerical features
scaler = MinMaxScaler()
df[['Study Hours', 'Attendance (%)']] = scaler.fit_transform(df[['Study Hours', 'Attendance (%)']])

# Drop any remaining NaNs
df.dropna(inplace=True)
print("Preprocessed Data:")
print(df.head())

# Split data
X = df.drop('FinalGrade', axis=1)
y = df['FinalGrade']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Save model
joblib.dump(model, 'score_model.pkl')
print("Model saved as 'score_model.pkl'")

# Visualization
plt.scatter(y_test, y_pred, alpha=0.7, color='teal')
plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.show()
print("Training features:", X.columns.tolist())