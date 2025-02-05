from datasets import load_from_disk
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
# Load the dataset from disk.
dataset = load_from_disk("whatsapp_dataset")

# Choose your model. (Change model_name to e.g. "mistralai/Mistral-7B-v0.1" if desired.)
model_name = "gpt2-large"

tokenizer = AutoTokenizer.from_pretrained(model_name)
# GPT-2 does not have a default pad token so assign it to eos_token.
tokenizer.pad_token = tokenizer.eos_token

# (Optional) Use 8-bit quantization if using a larger model like Mistral.
quantization_config = BitsAndBytesConfig(load_in_8bit=True) if "mistral" in model_name.lower() else None

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config
)

# This tokenization function concatenates the prompt ("input") with the expected answer ("output")
# and then masks out the prompt tokens in the labels so that only the user reply is learned.
def tokenize_function(examples):
    max_length = 128  # Adjust as needed.
    # Create full text by concatenating prompt and answer.
    full_texts = [inp + out for inp, out in zip(examples["input"], examples["output"])]
    tokenized = tokenizer(full_texts, truncation=True, padding="max_length", max_length=max_length)
    
    # Now, create labels that ignore the prompt tokens.
    labels = []
    for i in range(len(examples["input"])):
        # Tokenize the prompt only (without special tokens).
        prompt_tokens = tokenizer(examples["input"][i], truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"]
        full_ids = tokenized["input_ids"][i]
        label_ids = full_ids.copy()
        prompt_length = len(prompt_tokens)
        # Mask out the prompt portion (set to -100 so loss is not computed for these tokens).
        label_ids[:prompt_length] = [-100] * prompt_length
        labels.append(label_ids)
    tokenized["labels"] = labels
    return tokenized

# Apply tokenization.
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets.save_to_disk("tokenized_whatsapp_dataset")

# Define training arguments.
training_args = TrainingArguments(
    output_dir="./chatbot_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,  # Adjust based on your GPU memory.
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    fp16=True,  # Mixed precision training for speed.
    gradient_accumulation_steps=1
)

# Define the Trainer.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer
)

# Define a generation function.
def generate_response(model, tokenizer, prompt, gen_max_length=100):
    """
    Given a friend's message (prompt), generate a predicted user response.
    The prompt should already include "Friend: ..." and "User: " (without the answer).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # You can experiment with sampling parameters for more diverse outputs.
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=gen_max_length,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    # Remove the prompt from the generated text, if it was repeated.
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    return response

# Fine-tune the model and print generation examples after each epoch.
for epoch in range(training_args.num_train_epochs):
    print(f"\nTraining Epoch {epoch + 1}...\n")
    trainer.train()
    
    # Pick a random test example (or use the first example).
    test_sample = tokenized_datasets["test"][0]
    # To get a human-readable prompt, decode the prompt part.
    # (We assume the prompt is the part before the user's answer.)
    prompt = test_sample["input"]
    # You might want to use the original (non-cleaned) prompt from your dataset.
    # In our dataset, "input" is already like "Friend: ...\nUser: "
    
    chatbot_reply = generate_response(model, tokenizer, prompt)
    
    print("\n**Chatbot Test Output**:")
    print(f"**Input (Friend's message + 'User: ' prompt):** {prompt}")
    print(f"**Predicted User Response:** {chatbot_reply}\n")

# Save the fine-tuned model and tokenizer.
model.save_pretrained("./whatsapp_bot")
tokenizer.save_pretrained("./whatsapp_bot")

print("Training Complete! Model saved in './whatsapp_bot'")
