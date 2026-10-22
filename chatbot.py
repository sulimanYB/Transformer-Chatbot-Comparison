from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Select the Hugging Face model
# model_name = "google/flan-t5-small"
# model_name = "google/flan-t5-base"
model_name = "facebook/blenderbot-400M-distill"

print(f"Loading model: {model_name}")

# Load the tokenizer and pretrained model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("Model loaded successfully!\n")


def chat_with_bot():
    while True:
        user_input = input("You: ").strip()

        # Exit conditions
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Chatbot: Goodbye!")
            break

        if not user_input:
            continue

        # Convert input text into tokens
        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        # Generate a response
        outputs = model.generate(
            **inputs,
            max_new_tokens=150
        )

        # Convert generated tokens back into text
        response = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        ).strip()

        print(f"Chatbot: {response}\n")


if __name__ == "__main__":
    chat_with_bot()
