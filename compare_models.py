import time
import pandas as pd
import torch

from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# Models to compare
MODELS = [
    "google/flan-t5-small",
    "google/flan-t5-base",
    "facebook/blenderbot-400M-distill",
]

# Identical prompts for each model
PROMPTS = [
    "Define artificial intelligence in one simple sentence.",
    "Explain machine learning to a beginner in one sentence.",
    "Answer this geography question: What is the capital city of France?",
]


def compare_models():
    results = []

    for model_name in MODELS:
        print(f"\nLoading model: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.eval()

        for prompt in PROMPTS:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )

            start_time = time.perf_counter()

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=60,
                )

            elapsed = time.perf_counter() - start_time

            response = tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
            ).strip()

            results.append({
                "Model": model_name,
                "Prompt": prompt,
                "Response": response,
                "Response Time (s)": round(elapsed, 3),
                "Generated Tokens": len(outputs[0]),
            })

            print(f"Prompt: {prompt}")
            print(f"Response: {response}")
            print(f"Time: {elapsed:.3f} seconds\n")

        # Release the current model before loading the next
        del model, tokenizer

    df = pd.DataFrame(results)

    # Save results in the results directory
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "comparison_results.csv"
    df.to_csv(output_file, index=False)

    print("\nComparison completed!")
    print(df.to_string(index=False))
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    compare_models()
