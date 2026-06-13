"""
CustomJSON task for loading conversations from JSONL files.

Supported line formats:
- [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
- {"prompt": "Hi", "completion": "Hello"}
- {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]}
"""

import os
import json
from tasks.common import Task

class CustomJSON(Task):
    """
    Load conversations from a JSONL file.
    Each line may be a message array, a {"messages": [...]} object, or a
    {"prompt": ..., "completion": ...} object.
    Example line: [{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"}]
    """

    def __init__(self, filepath, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        self.conversations = []

        # Load all conversations from the JSONL file
        if not os.path.exists(filepath):
            # Helpful error message due to recent change. Will be removed in the future.
            print("-" * 80)
            print(f"Warning: File {filepath} does not exist")
            print("HINT (Oct 21 2025)")
            print("If you recently did a git pull and suddenly see this, it might be due to the new addition of identity conversations")
            print("See this discussion for more details: https://github.com/karpathy/nanochat/discussions/139")
            print("Quick fix: simply run the following command to download the file and you're done:")
            print(f"curl -L -o {filepath} https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl")
            print("-" * 80)

        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:  # skip empty lines
                        continue
                    record = json.loads(line)
                    messages = self._normalize_record(record, line_num)
                    self._validate_messages(messages, line_num)

                    self.conversations.append(messages)

        self.length = len(self.conversations)

    @staticmethod
    def _normalize_record(record, line_num):
        if isinstance(record, list):
            return record

        if isinstance(record, dict):
            if isinstance(record.get("messages"), list):
                return record["messages"]

            if "prompt" in record and "completion" in record:
                prompt = record["prompt"]
                completion = record["completion"]
                assert isinstance(prompt, str), f"Line {line_num}: prompt must be a string"
                assert isinstance(completion, str), f"Line {line_num}: completion must be a string"
                return [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ]

        raise AssertionError(f"Line {line_num}: expected messages list or prompt/completion object")

    @staticmethod
    def _validate_messages(messages, line_num):
        assert isinstance(messages, list), f"Line {line_num}: expected list of messages, got {type(messages)}"
        assert len(messages) >= 2, f"Line {line_num}: conversation must have at least 2 messages, got {len(messages)}"

        for i, message in enumerate(messages):
            assert isinstance(message, dict), f"Line {line_num}: message {i} must be an object"
            assert "role" in message, f"Line {line_num}: message {i} missing 'role' field"
            assert "content" in message, f"Line {line_num}: message {i} missing 'content' field"
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == expected_role, (
                f"Line {line_num}: message {i} has role {message['role']} but should be {expected_role}"
            )
            assert isinstance(message["content"], str), f"Line {line_num}: message {i} content must be a string"

    def num_examples(self):
        return self.length

    def get_example(self, index):
        messages = self.conversations[index]
        conversation = {
            "messages": messages,
        }
        return conversation
