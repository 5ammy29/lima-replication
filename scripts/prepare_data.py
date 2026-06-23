import json
from pathlib import Path
from datasets import load_dataset

def convert_to_messages(conversations):
    messages = []
    for i, text in enumerate(conversations):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({
            "role": role,
            "content": text,
        })
    return messages

def clean_example(example):
    conversations = example["conversations"]
    if len(conversations) == 0:
        return None
    return {
        "messages": convert_to_messages(conversations),
        "source": example["source"],
        "num_turns": len(conversations),
    }

def save_jsonl(rows, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

dataset = load_dataset("GAIR/lima")
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

train = []
for example in dataset["train"]:
    cleaned = clean_example(example)
    if cleaned is not None:
        train.append(cleaned)

test = []
for example in dataset["test"]:
    cleaned = clean_example(example)
    if cleaned is not None:
        test.append(cleaned)

save_jsonl(train, data_dir / "train_data.jsonl")
save_jsonl(test, data_dir / "test_data.jsonl")
