from datasets import load_dataset

dataset = load_dataset("GAIR/lima")

print("_" * 100)
print("Dataset structure:")
print(dataset)
print("_" * 100)

print("Available splits:")
print(dataset.keys())
print("_" * 100)

train_data = dataset["train"]
print("Number of training examples:")
print(len(train_data))
print("_" * 100)

print("\nColumn names:")
print(train_data.column_names)
print("_" * 100)

print("\nFirst example:")
print(train_data[0])
print("_" * 100)

print("\nSecond example:")
print(train_data[1])
print("_" * 100)
