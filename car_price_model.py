
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("car_data.csv")

print("First few rows of dataset:")
display(df.head())


df = df.drop("Car_Name", axis=1)


le = LabelEncoder()
df["Fuel_Type"] = le.fit_transform(df["Fuel_Type"])
df["Seller_Type"] = le.fit_transform(df["Seller_Type"])
df["Transmission"] = le.fit_transform(df["Transmission"])



X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestRegressor()
model.fit(X_train, y_train)



y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)

print(f"Model Accuracy (R2 Score): {accuracy:.2f}")



plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Price")
plt.show()


sample_data = pd.DataFrame({
    "Year": [2018],
    "Present_Price": [10.5],
    "Kms_Driven": [25000],
    "Fuel_Type": [le.fit_transform(["Petrol"])[0]],
    "Seller_Type": [le.fit_transform(["Dealer"])[0]],
    "Transmission": [le.fit_transform(["Manual"])[0]],
    "Owner": [0]
})

predicted_price = model.predict(sample_data)[0]
print(f"Predicted Selling Price for Sample Car: {predicted_price:.2f} lakhs")
