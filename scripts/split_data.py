import json
import random
from pathlib import Path


SOURCE_PATH = Path("data/train_data.jsonl")
TRAIN_PATH = Path("data/train_data_900.jsonl")
EVAL_PATH = Path("data/eval_data_130.jsonl")

TRAIN_SIZE = 900
SEED = 42


def load_jsonl(path: Path) -> list[dict]:
    examples = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                examples.append(json.loads(line))

    return examples


def save_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(
                json.dumps(example, ensure_ascii=False) + "\n"
            )


def main() -> None:
    examples = load_jsonl(SOURCE_PATH)

    if len(examples) != 1030:
        raise ValueError(
            f"Expected at least 1000 examples, found {len(examples)}."
        )

    random_generator = random.Random(SEED)
    random_generator.shuffle(examples)

    train_examples = examples[:TRAIN_SIZE]
    eval_examples = examples[TRAIN_SIZE:]

    save_jsonl(train_examples, TRAIN_PATH)
    save_jsonl(eval_examples, EVAL_PATH)

    print(f"Total examples: {len(examples)}")
    print(f"Training examples: {len(train_examples)}")
    print(f"Evaluation examples: {len(eval_examples)}")
    print(f"Split seed: {SEED}")


if __name__ == "__main__":
    main()