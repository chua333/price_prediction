import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# data reading and data cleaning, EDA part
df = pd.read_csv("laptop_price.csv", encoding="latin-1")

df = df.drop("Product", axis=1)

df = df.join(pd.get_dummies(df.Company))
df = df.drop("Company", axis=1)

df = df.join(pd.get_dummies(df.TypeName))
df = df.drop("TypeName", axis=1)

df["ScreenResolution"] = df.ScreenResolution.str.split(" ").apply(lambda x: x[-1])
df["Screen Width"] = df.ScreenResolution.str.split("x").apply(lambda x: x[0])
df["Screen Height"] = df.ScreenResolution.str.split("x").apply(lambda x: x[1])

df = df.drop("ScreenResolution", axis=1)

df["CPU Brand"] = df.Cpu.str.split(" ").apply(lambda x: x[0])
df["CPU Frequency"] = df.Cpu.str.split(" ").apply(lambda x: x[-1])

df = df.drop("Cpu", axis=1)

df["CPU Frequency"] = df["CPU Frequency"].str[:-3]

df["Ram"] = df["Ram"].str[:-2]

df["Ram"] = df["Ram"].astype("int")
df["CPU Frequency"] = df["CPU Frequency"].astype("float")

df["Screen Width"] = df["Screen Width"].astype("int")
df["Screen Height"] = df["Screen Height"].astype("int")

df["Memory Amount"] = df.Memory.str.split(" ").apply(lambda x: x[0])
df["Memory Type"] = df.Memory.str.split(" ").apply(lambda x: x[1])

def turn_memory_into_MB(value):
    if "GB" in value:
        return float(value[:value.find("GB")]) * 1000
    elif "TB" in value:
        return float(value[:value.find("TB")]) * 1000 * 1000

df["Memory Amount"] = df["Memory Amount"].apply(turn_memory_into_MB)

df = df.drop("Memory", axis=1)

df["Weight"] = df["Weight"].str[:-2]

df["Weight"] = df["Weight"].astype("float")

df["GPU Brand"] = df.Gpu.str.split(" ").apply(lambda x: x[0])

df = df.drop("Gpu", axis=1)

df = df.join(pd.get_dummies(df.OpSys))
df = df.drop("OpSys", axis=1)

cpu_categories = pd.get_dummies(df["CPU Brand"])
cpu_categories.columns = [col + "_CPU" for col in cpu_categories.columns]

df = df.join(cpu_categories)
df = df.drop("CPU Brand", axis=1)

gpu_categories = pd.get_dummies(df["GPU Brand"])
gpu_categories.columns = [col + "_GPU" for col in gpu_categories.columns]

df = df.join(gpu_categories)
df = df.drop("GPU Brand", axis=1)

# graph generation part

numeric_df = df.select_dtypes(include=['number'])
# Compute the correlation with the target variable 'Price_euros'
target_correlations = numeric_df.corr()["Price_euros"].apply(abs).sort_values()

selected_features = target_correlations[-21:].index
selected_features = list(selected_features)

limited_df = df[selected_features]

plt.figure(figsize=(18,15))
sns.heatmap(limited_df.corr(), annot=True, cmap="YlGnBu")
plt.savefig('heatmap.png')
plt.show()

# model creation part

X, y = limited_df.drop("Price_euros", axis=1), limited_df["Price_euros"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

forest = RandomForestRegressor()
forest.fit(X_test_scaled, y_test)

y_pred = forest.predict(X_test_scaled)

plt.figure(figsize=(12, 8))
plt.scatter(y_pred, y_test, label='Predicted vs. Actual', color='blue', alpha=0.6)
plt.plot(range(0, 6000), range(0, 6000), c="red", label='Perfect Prediction Line')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Predicted vs. Actual Scatter Plot')
plt.legend()
plt.savefig("scatterplot.png")
plt.show()

# take the first item from x and y
X_new_scaled = scaler.transform([X_test.iloc[0]])
forest.predict(X_new_scaled)

print(forest.predict(X_new_scaled))
print(y_test.iloc[0])
