import json
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent / "jarvis_behavior.jsonl"
BACKUP_FILE = DATA_FILE.with_suffix(".jsonl.bak")


def main():
    if not DATA_FILE.exists():
        print(f"ERROR: {DATA_FILE} does not exist.")
        return

    original_text = DATA_FILE.read_text()
    BACKUP_FILE.write_text(original_text)

    original_lines = original_text.splitlines()

    cleaned = []
    seen = set()

    blank_lines = 0
    duplicates = 0
    invalid_lines = 0

    for line_number, raw_line in enumerate(original_lines, start=1):
        line = raw_line.strip()

        # Remove blank lines.
        if not line:
            blank_lines += 1
            continue

        # Remove accidental leading/trailing commas.
        line = line.lstrip(",").rstrip(",").strip()

        # Repair the known malformed line.
        if line_number == 22:
            line = (
                '{"messages":['
                '{"role":"user","content":"Where is the calculator tool implemented?"},'
                '{"role":"assistant","tool_calls":['
                '{"function":{'
                '"name":"search_files",'
                '"arguments":{"query":"calculator"}'
                '}}'
                ']}'
                ']}'
            )

        # Verify JSON before accepting the line.
        try:
            json.loads(line)
        except json.JSONDecodeError as error:
            print(f"Removing invalid line {line_number}: {error}")
            invalid_lines += 1
            continue

        # Remove exact duplicates.
        if line in seen:
            duplicates += 1
            continue

        seen.add(line)
        cleaned.append(line)

    DATA_FILE.write_text("\n".join(cleaned) + "\n")

    print("=== JARVIS TRAINING DATA CLEANUP ===")
    print(f"Original lines:      {len(original_lines)}")
    print(f"Blank lines removed: {blank_lines}")
    print(f"Duplicates removed:  {duplicates}")
    print(f"Invalid lines left:  {invalid_lines}")
    print(f"Final lines:         {len(cleaned)}")
    print()
    print(f"Backup: {BACKUP_FILE}")
    print()
    print("Cleanup complete.")


if __name__ == "__main__":
    main()