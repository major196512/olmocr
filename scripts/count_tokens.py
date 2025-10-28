import argparse
import random

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal CPU-only token counting repro for Qwen2.5-VL.")
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Hugging Face model identifier for Qwen2.5-VL.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed to reproduce the color.")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    color = tuple(random.randint(0, 255) for _ in range(3))
    image = Image.new("RGB", (1024, 1024), color)
    print(f"Generated image color (RGB): {color}")

    print(f"Loading processor and model from {args.model_id} on CPU ...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
    ).eval()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What color is the image? Reply with the dominant color you see."},
            ],
        }
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    print("---- Rendered Prompt ----")
    print(prompt)
    print("-------------------------")

    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
    )
    tokens = processor.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())

    print("---- Prompt Tokens ----")
    for index, token in enumerate(tokens):
        print(f"{index:04d}: {token}")
    image_pad_count = sum("image_pad" in token for token in tokens)
    print("-----------------------")
    print(f"Prompt length: {len(tokens)} tokens")
    print(f"image_pad tokens: {image_pad_count}")
    print("-----------------------")


if __name__ == "__main__":
    main()
