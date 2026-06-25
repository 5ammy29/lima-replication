import json
from pathlib import Path
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

INPUT_PATH = Path("data/train_data.jsonl")
OUTPUT_PATH = Path("data/qwen_train_data.jsonl")

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with INPUT_PATH.open("r", encoding="utf-8") as fin, OUTPUT_PATH.open("w", encoding="utf-8") as fout:
        for line in fin:
            example = json.loads(line)

            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )

            fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()