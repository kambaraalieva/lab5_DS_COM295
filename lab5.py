import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Load Excel files
# -----------------------------
water_df = pd.read_excel("water.xlsx")
air_df = pd.read_excel("air_pollution.xlsx")
cancer_df = pd.read_excel("mortality_disease.xlsx")
gdp_df = pd.read_excel("gdp.xlsx")
industry_df = pd.read_excel("industry.xlsx")
mortality_df = pd.read_excel("mortality.xlsx")
tobacco_df = pd.read_excel("tobacco.xlsx")

# -----------------------------
# Step 2: Extract data for Kyrgyz Republic
# -----------------------------
def extract_country(df, country_name):
    return df[df['Country Name'] == country_name]

kyr_water = extract_country(water_df, "Kyrgyz Republic")
kyr_air = extract_country(air_df, "Kyrgyz Republic")
kyr_cancer = extract_country(cancer_df, "Kyrgyz Republic")
kyr_gdp = extract_country(gdp_df, "Kyrgyz Republic")
kyr_industry = extract_country(industry_df, "Kyrgyz Republic")
kyr_mortality = extract_country(mortality_df, "Kyrgyz Republic")
kyr_tobacco = extract_country(tobacco_df, "Kyrgyz Republic")

# -----------------------------
# Step 3: Combine into one DataFrame
# -----------------------------
years = kyr_water.columns[4:]  # assuming years start from column 5

kyr = pd.DataFrame({
    'year': years,
    'water': kyr_water.iloc[0, 4:].values,
    'air_pol': kyr_air.iloc[0, 4:].values,
    'cancer': kyr_cancer.iloc[0, 4:].values,
    'gdp': kyr_gdp.iloc[0, 4:].values,
    'industry': kyr_industry.iloc[0, 4:].values,
    'mortality': kyr_mortality.iloc[0, 4:].values,
    'tobacco': kyr_tobacco.iloc[0, 4:].values
})

# Save raw data
kyr.to_excel("Kyrgyz Republic_raw.xlsx", index=False, sheet_name="Raw data")

# -----------------------------
# Step 4: Visualize missing values
# -----------------------------
sns.heatmap(kyr.isnull(), cbar=False, cmap="viridis")
plt.title("Missing values heatmap - Raw data")
plt.show()

missing_percentage = kyr.isnull().mean() * 100
print("Missing values % per column:\n", missing_percentage)

# -----------------------------
# Step 5: Drop rows with too many missing values
# -----------------------------
threshold = len(kyr.columns) * 0.35  # 35%
kyr_filtered = kyr.dropna(thresh=threshold)

# Drop columns with too many missing values
kyr_filtered = kyr_filtered.drop(columns=['air_pol', 'mortality', 'tobacco'])

# Heatmap after cleaning
sns.heatmap(kyr_filtered.isnull(), cbar=False, cmap="viridis")
plt.title("Missing values heatmap - Cleaned data")
plt.show()

missing_percentage_filtered = kyr_filtered.isnull().mean() * 100
print("Missing % after filtering:\n", missing_percentage_filtered)

# Save cleaned data
kyr_filtered.to_excel("Kyrgyz Republic_cleaned.xlsx", index=False, sheet_name="Cleaned data")

# -----------------------------
# Step 6: Handle remaining missing values
# -----------------------------
# Interpolate water and cancer columns (time series)
kyr_filtered['water'] = kyr_filtered['water'].interpolate(method='linear')
kyr_filtered['cancer'] = kyr_filtered['cancer'].interpolate(method='linear')

# If any remaining missing values, fill with mean
kyr_filtered['water'].fillna(kyr_filtered['water'].mean(), inplace=True)
kyr_filtered['cancer'].fillna(kyr_filtered['cancer'].mean(), inplace=True)

# Save completed data
kyr_filtered.to_excel("Kyrgyz Republic_completed.xlsx", index=False, sheet_name="Completed data")

print("Data processing completed! Raw, Cleaned, and Completed Excel files are saved.")
