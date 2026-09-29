import sys
sys.path.insert(0, r'C:\Users\amana\Documents\Jarvis')
from brain.local import _parse_qwen_tool_calls

# Test with a typical Qwen response
test_response = """{
  "name": "run_python_file",
  "arguments": {"path": "test_run.py"}
}"""

result = _parse_qwen_tool_calls(test_response)
print('Parsed:', result)

# Test with the problematic schema-like response
test_response2 = '''{
  "name": "run_python_file",
  "arguments": {"path": {"description": "Project-relative Python file path.", "type": "string"}}
}'''

result2 = _parse_qwen_tool_calls(test_response2)
print('Parsed2:', result2)