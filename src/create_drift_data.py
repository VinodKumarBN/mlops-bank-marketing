import pandas as pd

reference = pd.read_csv("data/bank-full.csv", sep=";")

current = reference.sample(
    n=5000,
    random_state=42
).copy()

# Create numeric drift
current["age"] = current["age"] + 15

# Create categorical drift
current.loc[:3000, "job"] = "management"

current.to_csv(
    "data/current_data.csv",
    sep=";",
    index=False
)

print("Simulated drift dataset created.")
print("Shape:", current.shape)