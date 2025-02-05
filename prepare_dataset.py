
import json
from datasets import Dataset
# Load structured chat data.
with open("whatsapp_chat.json", "r", encoding="utf-8") as file:
    chat_data = json.load(file)

# Build conversation examples.
# We assume that whenever a "Friend" message is encountered, the very next message is from "User".
formatted_data = []
for i in range(len(chat_data) - 1):
    if chat_data[i]["name"] == "Friend" and chat_data[i+1]["name"] == "User":
        # The input prompt includes the friend message and a "User:" cue.
        example = {
            "input": f"Friend: {chat_data[i]['message']}\nUser: ",
            "output": chat_data[i+1]["message"]
        }
        formatted_data.append(example)

# Create the Hugging Face Dataset and split into train and test sets.
dataset = Dataset.from_list(formatted_data)
dataset = dataset.train_test_split(test_size=0.1)
dataset.save_to_disk("whatsapp_dataset")

print("Examples from the formatted dataset:")
for i in range(5):
    print(formatted_data[i])