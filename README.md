# Transformer Model Comparison with LLM-as-a-Judge

A Python-based NLP project comparing three pretrained Hugging Face transformer models using response quality, inference speed, and automated evaluation with an OpenAI LLM judge.

## Project Overview

This project explores how different pretrained transformer models respond to identical prompts.

Three models are evaluated:

- **FLAN-T5 Small** — lightweight instruction-tuned language model.
- **FLAN-T5 Base** — larger instruction-tuned language model.
- **BlenderBot 400M Distill** — conversational language model.

An OpenAI LLM judge evaluates the generated responses using a consistent scoring rubric.

The project demonstrates local transformer inference, automated evaluation, performance measurement, and exploratory data analysis.

## Project Architecture

```text
Input Prompts
      |
      v
Hugging Face Models
      |
      +-- FLAN-T5 Small
      +-- FLAN-T5 Base
      +-- BlenderBot 400M
      |
      v
Generated Responses
      |
      v
OpenAI LLM-as-a-Judge
      |
      +-- Accuracy (0–5)
      +-- Relevance (0–5)
      +-- Completeness (0–5)
      |
      v
CSV Evaluation Results
      |
      v
Pandas + Matplotlib Analysis
```

## Technologies Used

- Python
- Hugging Face Transformers
- PyTorch (CPU)
- OpenAI API
- Pandas
- Matplotlib
- Jupyter Notebook
- Conda

## Project Structure

```text
Transformer-Chatbot-Comparison/
├── chatbot.py
├── compare_models.py
├── llm_judge.py
├── model_analysis.ipynb
├── requirements.txt
├── .gitignore
├── results/
│   ├── comparison_results.csv
│   └── llm_evaluation.csv
└── README.md
```

## Installation

Clone the repository and navigate to the project directory.

Create and activate a Conda environment:

```bash
conda create -n transformer-chatbot python=3.11 -y
conda activate transformer-chatbot
```

For CPU-only execution, install the compatible PyTorch CPU build using the instructions at [PyTorch](https://pytorch.org/get-started/locally/).

Then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root and add your OpenAI API key:

```dotenv
OPENAI_API_KEY=your_api_key_here
```

The `.env` file must remain private and must not be committed to GitHub.

## Usage

### 1. Run the interactive chatbot

```bash
python chatbot.py
```

This launches an interactive command-line chatbot using the model configured in the script.

### 2. Compare Hugging Face models

```bash
python compare_models.py
```

The script submits identical prompts to the three models and records generated responses, inference times, and output token counts.

The initial experiment results are stored in `results/comparison_results.csv`.

### 3. Evaluate responses using an OpenAI judge

```bash
python llm_judge.py
```

The judge assesses each generated response using three criteria:

| Criterion | Score |
|---|---|
| Accuracy | 0–5 |
| Relevance | 0–5 |
| Completeness | 0–5 |
| **Total** | **0–15** |

Results are saved to `results/llm_evaluation.csv`.

This step requires an OpenAI API key and incurs API usage charges.

### 4. Analyse the results

Open `model_analysis.ipynb` in Jupyter or VS Code.

The notebook provides visualisations of response quality, inference speed, and the relationship between quality and generation time.

## Experimental Results

The initial experiment evaluated three prompts per model.

| Model | Average Quality (/15) | Average Inference Time |
|---|---:|---:|
| FLAN-T5 Small | 4.67 | 0.080 s |
| FLAN-T5 Base | 8.33 | 0.272 s |
| BlenderBot 400M | 2.00 | 2.965 s |

### Key Findings

**FLAN-T5 Base** achieved the highest average LLM-judged quality score in this experiment while maintaining relatively low inference time.

**FLAN-T5 Small** generated responses fastest but struggled with some factual and explanatory questions.

**BlenderBot** generated more conversational responses but frequently failed to address the instruction-based prompts directly.

These findings illustrate that model suitability depends on the task, rather than model size or response speed alone.

## Limitations

- Only three prompts were evaluated per model.
- Response quality was assessed by a single LLM judge.
- The judge's scores are subjective estimates rather than independently verified factual measurements.
- Inference timings were collected from single runs on a local CPU.
- Different response lengths affect generation time.
- The prompts primarily tested instruction following, which may disadvantage conversational models such as BlenderBot.

The results should therefore be interpreted as an exploratory comparison rather than a comprehensive benchmark.

## Future Improvements

Potential extensions include a larger benchmark dataset, reference-based factual evaluation, repeated inference timing measurements, multiple independent judge evaluations, and conversational evaluation prompts.

## Conclusion

This project demonstrates a practical workflow for comparing pretrained transformer models using local inference and automated LLM-based evaluation.

It highlights the importance of evaluating both response quality and computational performance when selecting language models for NLP applications.