from datasets import load_dataset

# Load IMDB dataset
dataset = load_dataset("imdb")

# Print the first training example
print(dataset["train"][0])