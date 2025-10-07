# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 7/10/25
### Name:Dhivya Dharshini B
### Reg:212223240031



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# -------------------------------
# Step 1: Load Data
# -------------------------------
data = pd.read_csv("gold_price_data.csv")

if 'Date' in data.columns:
    # Case 1: If dataset has a Date column
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')
    ts = data.set_index('Date')['Value']
else:

    ts = data['Value']
    ts.index = range(len(ts))  

print("\n--- GIVEN DATA (First 10 rows) ---")
print(ts.head(10))

# -------------------------------
# Step 2: Stationarity Test
# -------------------------------
result = adfuller(ts.dropna())
print("\nADF Statistic:", result[0])
print("p-value:", result[1])
print("Critical Values:", result[4])

# -------------------------------
# Step 3: Train-Test Split
# -------------------------------
train_size = int(len(ts) * 0.8)
train, test = ts.iloc[:train_size], ts.iloc[train_size:]

# -------------------------------
# Step 4: ACF & PACF
# -------------------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plot_acf(ts.dropna(), lags=30, ax=plt.gca())
plt.title("Autocorrelation Function (ACF)")

plt.subplot(1,2,2)
plot_pacf(ts.dropna(), lags=30, ax=plt.gca(), method="ywm")
plt.title("Partial Autocorrelation Function (PACF)")

plt.tight_layout()
plt.show()

# -------------------------------
# Step 5: Fit AutoRegressive Model
# -------------------------------
model = AutoReg(train, lags=5).fit()
print("\n--- MODEL SUMMARY ---")
print(model.summary())

# -------------------------------
# Step 6: Predictions
# -------------------------------
preds = model.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

# -------------------------------
# Step 7: Error Evaluation
# -------------------------------
error = mean_squared_error(test, preds)
print("\nMean Squared Error:", error)

print("\n--- FINAL PREDICTION (First 10 values) ---")
final_df = pd.DataFrame({"Actual": test.values, "Predicted": preds.values}, index=test.index)
print(final_df.head(10))

# -------------------------------
# Step 8: Plot Actual vs Predicted
# -------------------------------
plt.figure(figsize=(10,5))
plt.plot(test.index, test, label='Actual', linewidth=2)
plt.plot(test.index, preds, label='Predicted', color='red', linestyle="--")
plt.legend()
plt.title("Sales Data: Actual vs Predicted (AR Model)")
plt.xlabel("Date")
plt.ylabel("Value")
plt.show()

```

# OUTPUT:

### GIVEN DATA
<img width="378" height="296" alt="image" src="https://github.com/user-attachments/assets/fba4bc5a-e942-4645-a8f8-1e453c6d2db0" />

### PACF - ACF
<img width="1209" height="549" alt="image" src="https://github.com/user-attachments/assets/22d2b085-62c0-48e0-ae85-aaf73325d248" />


### PREDICTION
<img width="647" height="551" alt="image" src="https://github.com/user-attachments/assets/91309d80-4693-410b-bf21-a11610be0375" />

### FINIAL PREDICTION
<img width="822" height="663" alt="image" src="https://github.com/user-attachments/assets/f8039127-7517-4003-9df8-c728051d720d" />

# RESULT:
Thus we have successfully implemented the auto regression function using python.
