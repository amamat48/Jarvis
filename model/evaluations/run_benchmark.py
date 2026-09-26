import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))

import json

from brain.prompt import SYSTEM_PROMPT
from brain.router import chat
from tools.registry import select_tools
from tools.runner import execute_tool


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_FILE = Path(__file__).resolve().parent / "jarvis_benchmark.json"
RESULTS_FILE = Path(__file__).resolve().parent / "benchmark_results.json"


def run_case(case: dict) -> dict:
    """
    Run one JARVIS benchmark case and record its behavior.
    """

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": case["input"]
        }
    ]

    actual_tools = []
    tool_events = []

    available_tools = select_tools(case["input"])

    response = chat(
        messages,
        tools=list(available_tools.values())
    )

    messages.append(response.message)

    while response.message.tool_calls:
        for tool_call in response.message.tool_calls:
            tool_name = tool_call.function.name
            arguments = tool_call.function.arguments

            actual_tools.append(tool_name)

            result = execute_tool(
                tool_name,
                arguments
            )

            tool_events.append(
                {
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": result
                }
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": str(result)
                }
            )

        response = chat(
            messages,
            tools=list(available_tools.values())
        )

        messages.append(response.message)

    final_response = response.message.content or ""

    expected_tools = case.get("expected_tools", [])
    acceptable_tool_sequences = case.get(
        "acceptable_tool_sequences"
    )
    forbidden_tools = case.get("forbidden_tools", [])

    if acceptable_tool_sequences:
        exact_tool_match = actual_tools in acceptable_tool_sequences
    else:
        exact_tool_match = actual_tools == expected_tools

    forbidden_used = any(
        tool in forbidden_tools
        for tool in actual_tools
    )

    tool_behavior_passed = (
        exact_tool_match
        and not forbidden_used
    )

    return {
        "id": case["id"],
        "category": case["category"],
        "input": case["input"],
        "expected_tools": expected_tools,
        "actual_tools": actual_tools,
        "forbidden_tools": forbidden_tools,
        "forbidden_tool_used": forbidden_used,
        "exact_tool_match": exact_tool_match,
        "tool_behavior_passed": tool_behavior_passed,
        "final_response": final_response,
        "tool_events": tool_events,
        "goal": case.get("goal", "")
    }


def main() -> None:
    print("Loading JARVIS benchmark...")

    with BENCHMARK_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:
        cases = json.load(file)

    results = []

    for number, case in enumerate(cases, start=1):
        print(
            f"\nRunning test {number}/{len(cases)}: "
            f"{case['id']}"
        )

        try:
            result = run_case(case)

        except Exception as error:
            result = {
                "id": case["id"],
                "category": case["category"],
                "input": case["input"],
                "tool_behavior_passed": False,
                "error": str(error)
            }

        if result.get("tool_behavior_passed"):
            print("PASS")
        else:
            print("FAIL")
            print(f"  Expected tools: {result.get('expected_tools')}")
            print(f"  Actual tools:   {result.get('actual_tools')}")
            print(f"  Response:       {result.get('final_response', '')}")

        if result.get("tool_events"):
                print("  Tool events:")

        for event in result["tool_events"]:
            print(f"    Tool: {event['tool']}")
            print(f"    Args: {event['arguments']}")
            print(f"    Result: {event['result']}")

    with RESULTS_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False
        )

    passed = sum(
        1
        for result in results
        if result.get("tool_behavior_passed")
    )

    total = len(results)

    print("\n" + "=" * 50)
    print("JARVIS BENCHMARK COMPLETE")
    print("=" * 50)
    print(f"Tool-behavior score: {passed}/{total}")

    print(f"\nDetailed results saved to:")
    print(RESULTS_FILE)


if __name__ == "__main__":
    main()