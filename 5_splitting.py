import pandas as pd
from sklearn.model_selection import train_test_split

# Load the combined dataset
df = pd.read_csv('Sales_data.csv')

# Shuffle the dataset and stratify based on 'Make' to ensure equal representation of Hyundai and Kia
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['Make'], random_state=42)

# Output for checking
print("Training Set Size:", len(train_df))
print("Test Set Size:", len(test_df))

# Save the split datasets (optional)
train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)
