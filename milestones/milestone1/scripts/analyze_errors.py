
from datasets import load_dataset

dataset = load_dataset(
    "3nesdeniz/agentic-prompt-injection-5k"
)

validation = dataset["validation"]

error_ids = {11, 137, 200}

for example in validation:
    if example["id"] not in error_ids:
        continue

    print("\n" + "=" * 60)
    print("ID:", example["id"])
    print("Actual label:", example["label"])
    print("Attack family:", example["attack_family"])
    print("Technique:", example["technique"])
    print("Text:")
    print(example["text"])
