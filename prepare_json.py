
import json
from datetime import datetime, timedelta


# Define regex pattern to detect the start of a new message.
pattern = re.compile(
    r'^\[(?P<datetime>[^\]]+)\]\s*(?P<name>[^:]+):\s*(?P<message>.*)$'
)

messages = []
current_message = None
file_path = 'chat.txt'  # Path to your WhatsApp exported chat file
user_name = "Name"  # Change this to your actual name

# Possible date formats used in WhatsApp exports
date_formats = [
    "%m/%d/%y, %I:%M:%S %p",  # e.g., 03/15/21, 09:15:30 PM
    "%d/%m/%Y, %H:%M",        # e.g., 15/03/2021, 21:15
    "%d-%m-%Y %H:%M",         # e.g., 15-03-2021 21:15
]

def parse_datetime(date_str):
    """Try multiple date formats and return a parsed datetime object."""
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None

with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        match = pattern.match(line)
        if match:
            msg_datetime = parse_datetime(match.group('datetime').strip())
            sender = match.group('name').strip()
            msg_text = match.group('message').strip()
            if msg_datetime is None:
                print(f"Ignoring invalid date format: {match.group('datetime')}")
                continue

            # Rename sender: messages by you become "User", else "Friend"
            sender = "User" if sender == user_name else "Friend"

            # If the new message is from the same sender and within 1 minute, merge it.
            if (current_message is not None and
                current_message['name'] == sender and
                (msg_datetime - current_message['datetime']) <= timedelta(minutes=1)):
                current_message['message'] += "\n" + msg_text
                current_message['datetime'] = msg_datetime  # Update to latest timestamp
            else:
                if current_message is not None:
                    messages.append(current_message)
                current_message = {
                    'datetime': msg_datetime,
                    'name': sender,
                    'message': msg_text
                }
        else:
            if current_message is not None:
                current_message['message'] += "\n" + line  # Append extra lines

# Append last message if exists.
if current_message is not None:
    messages.append(current_message)

# Convert datetime objects to string before saving to JSON.
for msg in messages:
    msg['datetime'] = msg['datetime'].strftime("%m/%d/%y, %I:%M:%S %p")

with open("whatsapp_chat.json", "w", encoding="utf-8") as file:
    json.dump(messages, file, indent=4, ensure_ascii=False)

print(f"Size of parsed messages: {len(messages)}")
print("Chat successfully saved to whatsapp_chat.json")