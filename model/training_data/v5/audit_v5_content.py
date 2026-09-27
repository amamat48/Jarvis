import json
from collections import Counter
from pathlib import Path

DATASET = Path(__file__).parent / "train.jsonl"

with DATASET.open("r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

print("=" * 70)
print("JARVIS V5 CONTENT / STRUCTURE AUDIT")
print("=" * 70)

print(f"\nExamples: {len(data)}")

# ------------------------------------------------------------
# Basic conversation structure
# ------------------------------------------------------------

role_counts = Counter()
final_roles = Counter()
tool_call_counts = Counter()
tool_result_count = 0

examples_with_tools = 0
examples_multi_tool = 0
examples_tool_final = 0

for ex in data:
    messages = ex.get("messages", [])

    roles = [m.get("role") for m in messages]
    role_counts.update(roles)

    if roles:
        final_roles[roles[-1]] += 1

    calls = 0
    results = 0

    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls += len(m["tool_calls"])

        if m.get("role") == "tool":
            results += 1

    if calls:
        examples_with_tools += 1

    if calls > 1:
        examples_multi_tool += 1

    if roles and roles[-1] == "tool":
        examples_tool_final += 1

    tool_call_counts[calls] += 1
    tool_result_count += results

print("\n--- Roles ---")
for role, count in role_counts.items():
    print(f"{role:10}: {count}")

print("\n--- Final roles ---")
for role, count in final_roles.items():
    print(f"{role:10}: {count}")

print("\n--- Tool usage ---")
print(f"Examples with tools : {examples_with_tools}")
print(f"Examples without    : {len(data) - examples_with_tools}")
print(f"Multi-tool examples : {examples_multi_tool}")
print(f"Tool results        : {tool_result_count}")
print(f"Tool-final examples : {examples_tool_final}")

# ------------------------------------------------------------
# Tool distribution
# ------------------------------------------------------------

tool_names = Counter()

for ex in data:
    for m in ex.get("messages", []):
        if m.get("role") == "assistant":
            for call in m.get("tool_calls", []):
                fn = call.get("function", {})
                tool_names[fn.get("name", "UNKNOWN")] += 1

print("\n--- Tool call distribution ---")

for name, count in tool_names.most_common():
    print(f"{name:25}: {count}")

# ------------------------------------------------------------
# Content category detection
# ------------------------------------------------------------

categories = {
    "security": [
        "security", "sandbox", "credential", "secret", "api key",
        "private key", "untrusted", "malicious", "supply-chain",
        "permission", "exfiltrat", "prompt injection", "injection",
        "least privilege", "bypass", "admin"
    ],

    "memory": [
        "memory", "remember", "recall", "forget", "saved preference"
    ],

    "code": [
        "python", "code", "script", "program", "debug", "function",
        "variable", "class", "compile", "exception"
    ],

    "tool": [
        "tool", "search", "file", "read", "run", "calculate",
        "result", "evidence", "verify"
    ],

    "orchestration": [
        "multi-step", "workflow", "sequence", "coordinate",
        "before retry", "inspect", "verify", "dependency"
    ],
}

category_counts = Counter()
category_examples = {k: [] for k in categories}

for i, ex in enumerate(data):
    text = json.dumps(ex, ensure_ascii=False).lower()

    for category, keywords in categories.items():
        if any(keyword in text for keyword in keywords):
            category_counts[category] += 1
            category_examples[category].append(i)

print("\n--- Content categories ---")

for category in categories:
    count = category_counts[category]
    pct = count / len(data) * 100
    print(f"{category:15}: {count:3} ({pct:5.1f}%)")

# ------------------------------------------------------------
# Exact duplicate user prompts
# ------------------------------------------------------------

user_prompt_groups = {}

for i, ex in enumerate(data):
    user_messages = [
        m.get("content", "")
        for m in ex.get("messages", [])
        if m.get("role") == "user"
    ]

    if user_messages:
        prompt = user_messages[0]
        user_prompt_groups.setdefault(prompt, []).append(i)

duplicates = {
    prompt: indexes
    for prompt, indexes in user_prompt_groups.items()
    if len(indexes) > 1
}

print("\n--- Duplicate user prompts ---")
print(f"Duplicate prompt groups: {len(duplicates)}")
print(f"Examples involved       : {sum(len(v) for v in duplicates.values())}")

if duplicates:
    for prompt, indexes in list(duplicates.items())[:15]:
        print(f"\nExamples {indexes}")
        print(f"Prompt: {prompt[:180]!r}")

# ------------------------------------------------------------
# Exact duplicate examples
# ------------------------------------------------------------

serialized = {}
exact_duplicates = {}

for i, ex in enumerate(data):
    key = json.dumps(ex, sort_keys=True, ensure_ascii=False)

    if key in serialized:
        exact_duplicates.setdefault(serialized[key], []).append(i)

    else:
        serialized[key] = i

print("\n--- Exact duplicate examples ---")
print(f"Exact duplicate groups: {len(exact_duplicates)}")

# ------------------------------------------------------------
# Tool sequencing validation
# ------------------------------------------------------------

sequencing_errors = []

for i, ex in enumerate(data):
    messages = ex.get("messages", [])

    pending = []

    for j, m in enumerate(messages):

        if m.get("role") == "assistant":
            for call in m.get("tool_calls", []):
                call_id = call.get("id")
                if call_id:
                    pending.append(call_id)

        elif m.get("role") == "tool":
            result_id = m.get("tool_call_id")

            if not result_id:
                sequencing_errors.append(
                    (i, j, "tool result missing tool_call_id")
                )

            elif result_id not in pending:
                sequencing_errors.append(
                    (i, j, f"tool result references unknown/past call {result_id}")
                )

            else:
                pending.remove(result_id)

print("\n--- Tool sequencing ---")
print(f"Sequencing errors: {len(sequencing_errors)}")

if sequencing_errors:
    for error in sequencing_errors[:20]:
        print(error)

# ------------------------------------------------------------
# Assistant claims after tool calls
# ------------------------------------------------------------

tool_followups = 0
assistant_after_tool = 0

for ex in data:
    messages = ex.get("messages", [])

    for i, m in enumerate(messages[:-1]):
        if m.get("role") == "tool":
            tool_followups += 1

            if messages[i + 1].get("role") == "assistant":
                assistant_after_tool += 1

print("\n--- Tool-result follow-up behavior ---")
print(f"Tool results:                 {tool_followups}")
print(f"Followed by assistant reply:  {assistant_after_tool}")

# ------------------------------------------------------------
# Suspicious structural patterns
# ------------------------------------------------------------

suspicious = []

for i, ex in enumerate(data):
    messages = ex.get("messages", [])

    for j, m in enumerate(messages):
        role = m.get("role")

        if role == "tool":
            # Tool results should normally follow an assistant tool call.
            if j == 0 or messages[j - 1].get("role") != "assistant":
                suspicious.append(
                    (i, j, "tool message not immediately preceded by assistant")
                )

        if role == "assistant" and m.get("tool_calls"):
            if j == len(messages) - 1:
                suspicious.append(
                    (i, j, "assistant tool call is final message")
                )

print("\n--- Suspicious structures ---")
print(f"Suspicious structures: {len(suspicious)}")

if suspicious:
    for item in suspicious[:30]:
        print(item)

# ------------------------------------------------------------
# Length distribution
# ------------------------------------------------------------

lengths = []

for ex in data:
    chars = len(json.dumps(ex, ensure_ascii=False))
    lengths.append(chars)

lengths.sort()

def percentile(values, p):
    if not values:
        return 0

    index = (len(values) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)

    fraction = index - lower

    return values[lower] + (values[upper] - values[lower]) * fraction


print("\n--- Example size ---")
print(f"Minimum characters : {lengths[0]:.0f}")
print(f"Median characters  : {percentile(lengths, 0.50):.0f}")
print(f"P90 characters     : {percentile(lengths, 0.90):.0f}")
print(f"Maximum characters : {lengths[-1]:.0f}")

# ------------------------------------------------------------
# Final verdict
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("AUDIT SUMMARY")
print("=" * 70)

problems = []

if examples_tool_final:
    problems.append(
        f"{examples_tool_final} examples end on a tool result"
    )

if sequencing_errors:
    problems.append(
        f"{len(sequencing_errors)} tool sequencing errors"
    )

if suspicious:
    problems.append(
        f"{len(suspicious)} suspicious structural patterns"
    )

if exact_duplicates:
    problems.append(
        f"{len(exact_duplicates)} exact duplicate examples"
    )

if not problems:
    print("\nPASS")
    print("No structural/content red flags were detected.")
    print("Dataset is ready for final human review and freeze.")
else:
    print("\nREVIEW REQUIRED")
    for problem in problems:
        print(f"- {problem}")

print("\n" + "=" * 70)
