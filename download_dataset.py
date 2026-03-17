from datasets import load_dataset
import pandas as pd

# Load the dataset
dataset = load_dataset("IberaSoft/ecommerce-reviews-sentiment")

# Convert each split to a DataFrame
train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])

# Combine train + test (optional)
full_df = pd.concat([train_df, test_df])

# Save to CSV
full_df.to_csv("data/dataset.csv", index=False)

print("Downloaded and saved dataset.csv!")