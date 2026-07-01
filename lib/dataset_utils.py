"""Validate and describe a user-supplied fine-tuning dataset file."""

import json
import os


class DatasetError(ValueError):
    pass


def load_and_detect(path: str):
    """Return (format, example_count) for an Alpaca- or ShareGPT-format
    JSON/JSONL file, or raise DatasetError with a human-readable reason."""
    if not os.path.isfile(path):
        raise DatasetError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".jsonl":
            records = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        else:
            with open(path, encoding="utf-8") as f:
                records = json.load(f)
    except json.JSONDecodeError as e:
        raise DatasetError(f"Not valid JSON/JSONL: {e}")

    if not isinstance(records, list) or not records:
        raise DatasetError("Expected a non-empty JSON list of examples.")

    first = records[0]
    if not isinstance(first, dict):
        raise DatasetError("Each example must be a JSON object.")

    if "instruction" in first and "output" in first:
        fmt = "alpaca"
    elif "conversations" in first:
        fmt = "sharegpt"
    elif "messages" in first:
        fmt = "sharegpt_messages"
    else:
        raise DatasetError(
            "Unrecognized format. Expected Alpaca-style records with "
            "'instruction'/'output' fields, or ShareGPT-style records "
            "with a 'conversations' or 'messages' field."
        )

    return fmt, len(records)
