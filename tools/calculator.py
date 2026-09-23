def calculate(expression: str) -> str:
    """
    Calculate a mathematical expression.

    Args:
        expression: The mathematical expression to calculate.

    Returns:
        The result of the calculation.
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)

    except Exception as error:
        return f"Error: {error}"