from tools.registry import TOOLS


def execute_tool(tool_name, arguments):
    tool = TOOLS.get(tool_name)

    if tool is None:
        raise ValueError(f"Tool '{tool_name}' was not found.")

    return tool(**arguments)