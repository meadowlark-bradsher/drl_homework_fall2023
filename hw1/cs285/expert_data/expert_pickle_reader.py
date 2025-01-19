import pickle
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

with open('expert_data_Ant-v4.pkl', 'rb') as f:
    data = pickle.load(f)

df = pd.DataFrame(data)
print(df.iloc[0])  # Display the first few rows of your data

print(df.columns)

print(df.dtypes)

num_rows = len(df)
print("Number of rows in DataFrame:", num_rows)

#print(df['observation'].iloc[0])  # Shows the first row in the observation column
#print(df['action'].iloc[0])  # Shows the first row in the terminal column

for col in ['observation', 'action', 'reward', 'next_observation', 'terminal']:
    print(f"\nColumn: {col}")
    for i in range(min(5, len(df)-1)):  # Loop only up to the number of rows available
        entry = df[col].iloc[i]
        if isinstance(entry, np.ndarray):
            print(f"Entry {i} in column '{col}' has shape:", entry.shape)
        else:
            print(f"Entry {i} in column '{col}' is scalar with value:", entry)


# for col in ['observation', 'action', 'reward', 'next_observation', 'terminal']:
#     print(f"\nColumn: {col}")
#     for i in range(5):
#         entry = df[col].iloc[i]
#         if isinstance(entry, list):  # Check if entry is a list
#             print(f"Entry {i} length in column '{col}': {len(entry)}")
#         elif hasattr(entry, 'shape'):  # Check if entry has a shape attribute (e.g., numpy array)
#             print(f"Entry {i} shape in column '{col}': {entry.shape}")
#         else:
#             print(f"Entry {i} in column '{col}' is scalar with value:", entry)

df['reward_size'] = df['reward'].apply(lambda x: len(x) if isinstance(x, list) else 1)
df['action_size'] = df['action'].apply(lambda x: len(x) if isinstance(x, list) else 1)
#print(df[['reward_size', 'action_size']].describe())  # Summary statistics for sizes

#print(df.info)

# data = joblib.load('expert_data_Ant-v4.pkl')
#
# # Assuming 'data' is a Pandas DataFrame
# plt.figure(figsize=(10, 6))
# plt.plot(data['your_column'])
# plt.title('Plot of Your Data')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.show()