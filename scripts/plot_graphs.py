import json
import matplotlib.pyplot as plt

with open("outputs/sft_qwen/log_history.json", "r") as f:
    logs = json.load(f)

train_steps = []
train_loss = []
grad_norm = []
learning_rate = []

eval_steps = []
eval_loss = []

for entry in logs:
    if "loss" in entry:
        train_steps.append(entry["step"])
        train_loss.append(entry["loss"])
        grad_norm.append(entry["grad_norm"])
        learning_rate.append(entry["learning_rate"])

    if "eval_loss" in entry:
        eval_steps.append(entry["step"])
        eval_loss.append(entry["eval_loss"])

# Loss (train) vs Steps
plt.figure(figsize=(6,4))
plt.plot(train_steps, train_loss, marker="o")
plt.title("Training Loss vs Steps")
plt.xlabel("Training Step")
plt.ylabel("Training Loss")
plt.grid(True)
plt.tight_layout()
plt.show()

# Loss (eval) vs Steps
plt.figure(figsize=(6,4))
plt.plot(eval_steps, eval_loss, marker="s")
plt.title("Evaluation Loss vs Steps")
plt.xlabel("Training Step")
plt.ylabel("Evaluation Loss")
plt.grid(True)
plt.tight_layout()
plt.show()

# Gradient Norm vs Steps
plt.figure(figsize=(6,4))
plt.plot(train_steps, grad_norm, marker="^")
plt.title("Gradient Norm vs Steps")
plt.xlabel("Training Step")
plt.ylabel("Gradient Norm")
plt.grid(True)
plt.tight_layout()
plt.show()

# Steps vs Learning Rate
plt.figure(figsize=(6,4))
plt.plot(train_steps, learning_rate, marker="o")
plt.title("Learning Rate Schedule")
plt.xlabel("Training Step")
plt.ylabel("Learning Rate")
plt.grid(True)
plt.tight_layout()
plt.show()
