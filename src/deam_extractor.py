import pandas as pd

# Load the CSV file
csv_path = '/mnt/data/2058.csv'
df = pd.read_csv(csv_path)

# Display basic info
print(f"Columns in {csv_path}:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

# Assuming time-series data, aggregate features per column by mean and std
feature_means = df.mean()
feature_stds = df.std()

# Combine mean and std into one feature vector
aggregated_features = pd.concat([feature_means, feature_stds], axis=0)

print("\nAggregated feature vector (mean + std):")
print(aggregated_features)

# Convert to a DataFrame if you want to store or process further
agg_df = aggregated_features.to_frame().T

# Save aggregated features if needed
agg_df.to_csv('/mnt/data/2058_aggregated_features.csv', index=False)
