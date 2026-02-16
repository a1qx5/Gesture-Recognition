import pandas as pd

# Read the CSV
data = pd.read_csv('../data/gestures_data.csv')

print(f"Original shape: {data.shape}")
print(f"Original unique gesture IDs: {sorted(data['gesture_id'].unique())}")

# Remove rows with gesture_id = 4
data_filtered = data[data['gesture_id'] != 8]

print(f"\nFiltered shape: {data_filtered.shape}")
print(f"Filtered unique gesture IDs: {sorted(data_filtered['gesture_id'].unique())}")
print(f"\nCounts per gesture:")
print(data_filtered['gesture_id'].value_counts().sort_index())

# Save the filtered data
data_filtered.to_csv('../data/gestures_data.csv', index=False)
print("\nSaved filtered data to gestures_data.csv")
