from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate responses from the base and LIMA "
            "LoRA fine-tuned Qwen models."
        )
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
    )

    parser.add_argument(
        "--adapter-path",
        type=str,
        default="outputs/sft_qwen/final_adapter",
    )

    parser.add_argument(
        "--test-file",
        type=str,
        default="data/test_data.jsonl",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default="outputs/evaluation/sft_qwen_metrics.json",
    )

    parser.add_argument(
        "--num-examples",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--max-input-length",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
    test_path = Path(args.test_file)
    adapter_path = Path(args.adapter_path)

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test file does not exist: {test_path}"
        )

    if not adapter_path.exists():
        raise FileNotFoundError(
            f"Adapter directory does not exist: {adapter_path}"
        )

    adapter_config_path = (
        adapter_path / "adapter_config.json"
    )

    adapter_weights_path = (
        adapter_path / "adapter_model.safetensors"
    )

    if not adapter_config_path.exists():
        raise FileNotFoundError(
            f"Missing adapter configuration: "
            f"{adapter_config_path}"
        )

    if not adapter_weights_path.exists():
        raise FileNotFoundError(
            f"Missing adapter weights: "
            f"{adapter_weights_path}"
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: str) -> list[dict]:
    examples = []

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                example = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} "
                    f"of {path}."
                ) from error

            examples.append(example)

    return examples


def validate_test_examples(
    examples: list[dict],
) -> None:
    for index, example in enumerate(examples):
        if "messages" not in example:
            raise KeyError(
                f"Test example {index} does not contain "
                "a 'messages' field."
            )

        messages = example["messages"]

        if not isinstance(messages, list):
            raise TypeError(
                f"The 'messages' field in example {index} "
                "must be a list."
            )

        if len(messages) == 0:
            raise ValueError(
                f"Test example {index} has no messages."
            )

        has_user_message = any(
            message.get("role") == "user"
            for message in messages
        )

        if not has_user_message:
            raise ValueError(
                f"Test example {index} does not contain "
                "a user message."
            )


def select_examples(
    examples: list[dict],
    num_examples: int,
    seed: int,
) -> list[tuple[int, dict]]:
    if num_examples <= 0:
        raise ValueError(
            "The number of examples must be positive."
        )

    if num_examples > len(examples):
        raise ValueError(
            f"Requested {num_examples} examples, but the "
            f"test file contains only {len(examples)}."
        )

    random_generator = random.Random(seed)

    selected_indices = random_generator.sample(
        range(len(examples)),
        k=num_examples,
    )

    selected_examples = [
        (index, examples[index])
        for index in selected_indices
    ]

    return selected_examples


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_model_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Run this script "
            "through Modal or another CUDA machine."
        )

    if torch.cuda.is_bf16_supported():
        return torch.bfloat16

    return torch.float16


def load_base_model(
    model_name: str,
    model_dtype: torch.dtype,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=model_dtype,
        device_map="auto",
    )

    model.eval()
    model.config.use_cache = True

    return model


def extract_prompt(messages: list[dict]) -> str:
    user_messages = [
        message["content"]
        for message in messages
        if message.get("role") == "user"
    ]

    return user_messages[-1]


def generate_response(
    model,
    tokenizer,
    messages: list[dict],
    max_input_length: int,
    max_new_tokens: int,
) -> str:
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
        padding=False,
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    input_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    response_ids = generated_ids[
        0,
        input_length:,
    ]

    response = tokenizer.decode(
        response_ids,
        skip_special_tokens=True,
    )

    return response.strip()


def generate_base_responses(
    model,
    tokenizer,
    selected_examples: list[tuple[int, dict]],
    max_input_length: int,
    max_new_tokens: int,
) -> list[dict]:
    results = []

    print()
    print("Generating base-model responses")
    print("-" * 100)

    for output_id, (
        original_index,
        example,
    ) in enumerate(
        selected_examples,
        start=1,
    ):
        messages = example["messages"]

        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
        )

        result = {
            "id": output_id,
            "original_test_index": original_index,
            "messages": messages,
            "prompt": extract_prompt(messages),
            "base_response": response,
        }

        results.append(result)

        print(
            f"Generated base response "
            f"{output_id}/{len(selected_examples)}"
        )

    return results


def generate_fine_tuned_responses(
    model,
    tokenizer,
    results: list[dict],
    max_input_length: int,
    max_new_tokens: int,
) -> None:
    print()
    print("Generating fine-tuned responses")
    print("-" * 100)

    for index, result in enumerate(
        results,
        start=1,
    ):
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            messages=result["messages"],
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
        )

        result["fine_tuned_response"] = response

        print(
            f"Generated fine-tuned response "
            f"{index}/{len(results)}"
        )


def save_results(
    args: argparse.Namespace,
    results: list[dict],
) -> None:
    output_path = Path(args.output_file)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "evaluation_configuration": {
            "base_model": args.model_name,
            "adapter_path": args.adapter_path,
            "test_file": args.test_file,
            "number_of_examples": args.num_examples,
            "selection_seed": args.seed,
            "max_input_length": args.max_input_length,
            "max_new_tokens": args.max_new_tokens,
            "decoding": {
                "do_sample": False,
                "strategy": "greedy",
            },
        },
        "examples": results,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("Generation complete")
    print("-" * 100)
    print(f"Examples generated: {len(results)}")
    print(f"Output saved to: {output_path}")


def main() -> None:
    args = parse_args()

    set_seed(args.seed)
    validate_paths(args)

    print("Generation configuration")
    print("-" * 100)
    print(f"Base model: {args.model_name}")
    print(f"Adapter path: {args.adapter_path}")
    print(f"Test file: {args.test_file}")
    print(f"Output file: {args.output_file}")
    print(f"Number of examples: {args.num_examples}")
    print(f"Selection seed: {args.seed}")
    print(f"Maximum new tokens: {args.max_new_tokens}")

    examples = load_jsonl(
        args.test_file
    )

    validate_test_examples(
        examples
    )

    selected_examples = select_examples(
        examples=examples,
        num_examples=args.num_examples,
        seed=args.seed,
    )

    print()
    print("Dataset")
    print("-" * 100)
    print(f"Total test examples: {len(examples)}")
    print(
        f"Selected test examples: "
        f"{len(selected_examples)}"
    )

    tokenizer = load_tokenizer(
        args.model_name
    )

    model_dtype = get_model_dtype()

    base_model = load_base_model(
        model_name=args.model_name,
        model_dtype=model_dtype,
    )

    results = generate_base_responses(
        model=base_model,
        tokenizer=tokenizer,
        selected_examples=selected_examples,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
    )

    fine_tuned_model = PeftModel.from_pretrained(
        base_model,
        args.adapter_path,
    )

    fine_tuned_model.eval()
    fine_tuned_model.config.use_cache = True

    generate_fine_tuned_responses(
        model=fine_tuned_model,
        tokenizer=tokenizer,
        results=results,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
    )

    save_results(
        args=args,
        results=results,
    )


if __name__ == "__main__":
    main()
