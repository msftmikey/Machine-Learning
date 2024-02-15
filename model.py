import pandas as pd

                        
# Read the dataset into a DataFrame
df = pd.read_csv('dataset.csv')

# Filter the DataFrame to include rows with relevant indicators
relevant_indicators = ["Deaths from suicide, alcohol, drugs"]
filtered_df = df[df["Indicator"].isin(relevant_indicators)]

more_indicators = ["Homicides"]
more_filtered_df = df[df["Indicator"].isin(more_indicators)]

print(filtered_df[["Indicator", "Value"]].head())
print(more_filtered_df[["Indicator", "Value"]].head())