import os
from functools import reduce
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

COUNTRY_CODE = "JOR"
OUTPUT_FILE = "Jordan.xlsx"

def load_country_series(filename, country_code, value_name):
    import os
    import pandas as pd

    path = os.path.join(filename)

    df_raw = pd.read_excel(path, sheet_name="Data", header=None)
    header_row = df_raw[df_raw.eq("Country Name").any(axis=1)].index[0]
    df = pd.read_excel(path, sheet_name="Data", header=header_row)
    df = df[df["Country Code"].notna()]
    df = df[df["Country Code"] == country_code]

    df = df.drop(columns=["Country Name", "Country Code",
                          "Indicator Name", "Indicator Code"])

    df.columns = pd.to_numeric(df.columns, errors="coerce")
    df = df.loc[:, df.columns.notna()]

    df_long = df.melt(value_name=value_name, var_name="year")
    df_long["year"] = pd.to_numeric(df_long["year"], errors="coerce")
    df_long = df_long.dropna(subset=["year"])
    df_long["year"] = df_long["year"].astype(int)
    df_long = df_long[["year", value_name]]

    return df_long


water = load_country_series(
    "water.xlsx",
    COUNTRY_CODE, "water"
)

air_pol = load_country_series(
    "air_pollution.xlsx",
    COUNTRY_CODE, "air_pol"
)

cancer = load_country_series(
    "disease_mortality.xlsx",
    COUNTRY_CODE, "cancer"
)

gdp = load_country_series(
    "GDP.xlsx",
    COUNTRY_CODE, "gdp"
)

industry = load_country_series(
    "industry.xlsx",
    COUNTRY_CODE, "industry"
)

mortality = load_country_series(
    "mortality.xlsx",
    COUNTRY_CODE, "mortality"
)

tobacco = load_country_series(
    "tobacco.xlsx",
    COUNTRY_CODE, "tobacco"
)

# (RAW DATA)
dfs = [water, air_pol, cancer, gdp, industry, mortality, tobacco]
aut_raw = reduce(
    lambda left, right: pd.merge(left, right, on="year", how="outer"),
    dfs
)
aut_raw = aut_raw.sort_values("year").reset_index(drop=True)
print("RAW DATA:")
print(aut_raw.head())
print(aut_raw.tail())
with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    aut_raw.to_excel(writer, sheet_name="Raw data", index=False)

plt.figure(figsize=(8, 6))
sns.heatmap(aut_raw.isna(), cbar=False, yticklabels=False)
plt.title("Heatmap of Missing Values (Raw Data)")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.tight_layout()
plt.show()

missing_percent_raw = aut_raw.isna().mean() * 100
print("\nPercentage of missing values in RAW DATA:")
print(missing_percent_raw)

row_threshold = int(len(aut_raw.columns) * 0.65)
aut_filtered = aut_raw.dropna(thresh=row_threshold)
print("\nFILTERED DATA (after dropping rows with >35% missing):")
print(aut_filtered.head())
print(aut_filtered.tail())
col_threshold = int(aut_filtered.shape[0] * 0.65)
aut_cleaned = aut_filtered.dropna(axis=1, thresh=col_threshold)
print("\nCLEANED DATA (after dropping columns with many NaNs):")
print(aut_cleaned.head())
print(aut_cleaned.tail())

plt.figure(figsize=(8, 6))
sns.heatmap(aut_cleaned.isna(), cbar=False, yticklabels=False)
plt.title("Heatmap of Missing Values (Cleaned Data)")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.tight_layout()
plt.show()

missing_percent_cleaned = aut_cleaned.isna().mean() * 100
print("\nPercentage of missing values in CLEANED DATA:")
print(missing_percent_cleaned)

with pd.ExcelWriter(OUTPUT_FILE, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
    aut_cleaned.to_excel(writer, sheet_name="Cleaned data", index=False)


cols_with_na = [c for c in aut_cleaned.columns if aut_cleaned[c].isna().sum() > 0]

for col in cols_with_na:
    plt.figure()
    plt.plot(aut_cleaned["year"], aut_cleaned[col], marker="o")
    plt.title(f"Time Series of {col}")
    plt.xlabel("Year")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()

aut_completed = aut_cleaned.copy()
numeric_cols = [c for c in aut_completed.columns if c != "year"]

for col in numeric_cols:
    aut_completed[col] = aut_completed[col].interpolate(
        method="linear"
    )
    aut_completed[col] = aut_completed[col].fillna(
        aut_completed[col].median()
    )

print("\nCOMPLETED DATA:")
print(aut_completed.head())
print(aut_completed.tail())
print("\nAny left NaNs", aut_completed.isna().sum())

with pd.ExcelWriter(OUTPUT_FILE, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
    aut_completed.to_excel(writer, sheet_name="Completed data", index=False)

# LAB 6 – Correlation + Regression
data = aut_completed.copy()
corr = data.drop(columns=["year"]).corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, vmin=-1, vmax=1, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

print("\nCorrelation matrix:")
print(corr)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

reg_data = data.dropna(subset=["gdp", "industry"])

X = reg_data[["industry"]].values
y = reg_data["gdp"].values

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("\nRegression: gdp(y) ~ industry(x)")
print("Intercept:", model.intercept_)
print("Slope (industry coefficient):", model.coef_[0])
print("Mean Squared Error (MSE):", mse)
print("Coefficient of Determination (R^2):", r2)

plt.figure()
plt.scatter(X, y)
plt.plot(X, y_pred)
plt.xlabel("Industry growth (%)")
plt.ylabel("GDP growth (%)")
plt.title("Regression of GDP on Industry")
plt.tight_layout()
plt.show()