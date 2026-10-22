
import os
import json
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Load API credentials from .env
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env")

client = OpenAI()

JUDGE_MODEL = "gpt-4o-mini"

INPUT_FILE = Path("results/comparison_results.csv")
OUTPUT_FILE = Path("results/llm_evaluation.csv")


def evaluate_response(prompt, response):
    """Evaluate a model response using an OpenAI LLM."""

    instructions = """
You are an impartial evaluator of AI-generated responses.

Evaluate the response against the original user prompt.

Score each criterion from 0 to 5:

Accuracy:
0 = Factually incorrect
5 = Factually correct

Relevance:
0 = Completely unrelated
5 = Directly addresses the prompt

Completeness:
0 = Does not fulfil the request
5 = Fully fulfils the request

Important:
- Do not reward fluent but incorrect answers.
- Do not assume the model's response is correct.
- Use the same standards for every response.
- Judge only the provided prompt and response.
- If factual correctness cannot be established, explain the uncertainty.

Return a JSON object with exactly these fields:
{
    "accuracy": 0,
    "relevance": 0,
    "completeness": 0,
    "reasoning": "Brief explanation of the scores"
}
"""

    result = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": instructions},
            {
                "role": "user",
                "content": json.dumps({
                    "prompt": prompt,
                    "response": response,
                }),
            },
        ],
    )

    evaluation = json.loads(result.choices[0].message.content)

    # Validate the returned scores
    for criterion in ["accuracy", "relevance", "completeness"]:
        score = evaluation[criterion]

        if type(score) is not int or not 0 <= score <= 5:
            raise ValueError(f"Invalid {criterion} score: {score}")

    if not isinstance(evaluation.get("reasoning"), str):
        raise ValueError("Missing or invalid evaluation reasoning")

    return evaluation


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing file: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    evaluations = []

    print(f"Evaluating {len(df)} responses using {JUDGE_MODEL}\n")

    for index, row in df.iterrows():
        print(f"[{index + 1}/{len(df)}] {row['Model']}")

        try:
            evaluation = evaluate_response(
                str(row["Prompt"]),
                str(row["Response"]),
            )

        except Exception as error:
            print(f"Evaluation failed: {error}")
            print("Saving completed evaluations before stopping.")
            break

        evaluations.append({
            **row.to_dict(),
            "Accuracy": evaluation["accuracy"],
            "Relevance": evaluation["relevance"],
            "Completeness": evaluation["completeness"],
            "Judge Reasoning": evaluation["reasoning"],
            "Judge Model": JUDGE_MODEL,
        })

        print(
            f"Scores: {evaluation['accuracy']}, "
            f"{evaluation['relevance']}, "
            f"{evaluation['completeness']}"
        )

        time.sleep(0.2)

    if evaluations:
        results = pd.DataFrame(evaluations)

        results["Quality Score"] = results[
            ["Accuracy", "Relevance", "Completeness"]
        ].sum(axis=1)

        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(OUTPUT_FILE, index=False)

        print(f"\nSaved {len(results)} evaluations to {OUTPUT_FILE}")
    else:
        print("\nNo evaluations completed.")


if __name__ == "__main__":
    main()
