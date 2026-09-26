import inspect

from tools.registry import TOOLS


def execute_tool(tool_name, arguments):
    tool = TOOLS.get(tool_name)

    if tool is None:
        return f"Tool '{tool_name}' was not found."

    try:
        signature = inspect.signature(tool)
        signature.bind(**arguments)

    except TypeError as error:
        return (
            f"Invalid arguments for tool '{tool_name}': {error}"
        )

    try:
        return tool(**arguments)

    except Exception as error:
        return f"Tool '{tool_name}' failed: {error}"