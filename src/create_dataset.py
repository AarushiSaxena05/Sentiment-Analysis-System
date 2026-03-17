import pandas as pd

# Define dataset directly in code
data = {
    "text": [
        "I love this product",
        "This is terrible",
        "It was okay, not great",
        "Absolutely fantastic service",
        "I hate this",
        "Very good experience",
        "Not bad",
        "Terrible quality",
        "Amazing movie with great acting",
        "Worst purchase ever",
        "It was fine, nothing special",
        "Loved the customer support",
        "Disappointed by the service",
        "Neutral feelings about this",
        "Superb product quality",
        "The plot was boring",
        "Good value for money",
        "Horrible experience overall",
        "It is acceptable",
        "Excellent, I recommend it"
    ],
    "sentiment": [
        "positive","negative","neutral","positive","negative",
        "positive","neutral","negative","positive","negative",
        "neutral","positive","negative","neutral","positive",
        "negative","positive","negative","neutral","positive"
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save CSV in the data folder
df.to_csv("../data/dataset.csv", index=False)
print("dataset.csv created successfully in the data folder!")