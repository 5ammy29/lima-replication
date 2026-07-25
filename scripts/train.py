from __future__ import annotations

import subprocess

import modal


APP_NAME = "lima-lora-training"

REMOTE_PROJECT_DIRECTORY = "/root/lima"
REMOTE_OUTPUT_DIRECTORY = "/outputs"

app = modal.App(APP_NAME)


image = (
    modal.Image.debian_slim(
        python_version="3.11"
    )
    .pip_install(
        "torch>=2.6,<3",
        "transformers>=4.51,<5",
        "datasets>=3.5,<5",
        "peft>=0.15,<1",
        "accelerate>=1.6,<2",
        "sentencepiece",
    )
    .add_local_file(
        local_path="scripts/lora_qwen.py",
        remote_path=(
            f"{REMOTE_PROJECT_DIRECTORY}"
            "/scripts/lora_qwen.py"
        ),
    )
    .add_local_dir(
        local_path="data",
        remote_path=(
            f"{REMOTE_PROJECT_DIRECTORY}/data"
        ),
    )
)


output_volume = modal.Volume.from_name(
    "lima-training-outputs",
    create_if_missing=True,
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 4,
    volumes={
        REMOTE_OUTPUT_DIRECTORY: output_volume
    },
)
def train(
    num_train_epochs: float = 3.0,
    learning_rate: float = 2e-4,
    max_length: int = 1024,
    train_batch_size: int = 2,
    eval_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> None:
    train_script_path = (
        f"{REMOTE_PROJECT_DIRECTORY}"
        "/scripts/lora_qwen.py"
    )

    train_file_path = (
        f"{REMOTE_PROJECT_DIRECTORY}"
        "/data/train_data.jsonl"
    )

    test_file_path = (
        f"{REMOTE_PROJECT_DIRECTORY}"
        "/data/test_data.jsonl"
    )

    output_directory = (
        f"{REMOTE_OUTPUT_DIRECTORY}"
        "/sft_qwen"
    )

    command = [
        "python",
        train_script_path,

        "--model-name",
        "Qwen/Qwen2.5-0.5B-Instruct",

        "--train-file",
        train_file_path,

        "--test-file",
        test_file_path,

        "--output-dir",
        output_directory,

        "--num-train-epochs",
        str(num_train_epochs),

        "--learning-rate",
        str(learning_rate),

        "--max-length",
        str(max_length),

        "--train-batch-size",
        str(train_batch_size),

        "--eval-batch-size",
        str(eval_batch_size),

        "--gradient-accumulation-steps",
        str(gradient_accumulation_steps),
    ]

    if max_train_samples is not None:
        command.extend(
            [
                "--max-train-samples",
                str(max_train_samples),
            ]
        )

    if max_eval_samples is not None:
        command.extend(
            [
                "--max-eval-samples",
                str(max_eval_samples),
            ]
        )

    print("Running training command")
    print("-" * 80)
    print(" ".join(command))

    subprocess.run(
        command,
        check=True,
    )

    output_volume.commit()

    print()
    print("Training outputs saved to Modal Volume:")
    print("lima-training-outputs")
    print()
    print("Remote output path:")
    print(output_directory)


@app.local_entrypoint()
def main(
    num_train_epochs: float = 3.0,
    learning_rate: float = 2e-4,
    max_length: int = 1024,
    train_batch_size: int = 2,
    eval_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> None:
    train.remote(
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        max_length=max_length,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=(
            gradient_accumulation_steps
        ),
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
    )
