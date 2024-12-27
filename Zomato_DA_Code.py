import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv("zomato.csv")

# Create a DataFrame from the loaded data
df = pd.DataFrame(data)

# Display the first few rows of the DataFrame
print(df.head())

# Drop the unwanted columns correctly
df = df.drop(columns=["Unnamed: 0.1", "Unnamed: 0"])

print("\n", df.head())
columns = df.columns

df.rename(columns={col: col.title() for col in df.columns}, inplace=True)

print("\n", df.head())

def standardize_value(the_value):
    the_value = the_value.replace("@" , " ").replace("#" , " ").replace("'","").strip()
    the_value = the_value.lower()
    the_value = the_value.title()
    return the_value

df["Restaurant Name"] = df["Restaurant Name"].apply(standardize_value)

print("\n", df.head())

def standardize_values(col_value):
    if "," in col_value:
        return col_value.replace(',', '/').replace(' ', '').strip().title()
    else:
        return col_value

df["Restaurant Type"] = df["Restaurant Type"].apply(standardize_values)

df.rename(columns={
    "Rate (Out Of 5)": "Ratings",
    "Num Of Ratings": "Number of Ratings",
    "Avg Cost (Two People)": "Average cost",
    "Online_Order": "Online Order",
    "Cuisines Type": "Cuisine Types"
}, inplace=True)

print(df.columns)
df.columns = df.columns.str.strip()
print("\n", df.isna().sum())

df.dropna(subset=["Ratings", "Average cost"], inplace=True)
print("\n", df.isna().sum())

df["Ratings"] = df["Ratings"].fillna(df["Ratings"].mean())
df["Average cost"] = df["Average cost"].fillna(df["Average cost"].mean())

print(df)

plt.figure(figsize=(10, 6))
sns.histplot(df['Average cost'], bins=20, kde=True)
plt.title('Distribution of Average Cost')
plt.xlabel('Average Cost')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(12, 8))
df['Cuisine Types'].value_counts().head(10).plot(kind='bar')
plt.title('Top 10 Cuisine Types')
plt.xlabel('Cuisine Types')
plt.ylabel('Number of Restaurants')
plt.show()

# Filter only numeric columns for the heatmap
numeric_df = df.select_dtypes(include=[np.number])

# Correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

df.to_csv("Zomato_dataset.csv")