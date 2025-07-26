import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

X, y = fetch_california_housing(return_X_y=True)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = joblib.load("model.joblib")

y_pred = model.predict(X_test)
print("R² Score inside Docker:", r2_score(y_test, y_pred))
