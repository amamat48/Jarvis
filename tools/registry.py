from tools.calculator import calculate
from tools.file_reader import read_file
from tools.file_manager import list_files
from tools.code_runner import run_python_file
from tools.code_search import search_files
from tools.memory_tools import remember_memory, recall_memory
from tools.debugger import debug_python_file



TOOLS = {
    "calculate": calculate,
    "read_file": read_file,
    "list_files": list_files,
    "run_python_file": run_python_file,
    "search_files": search_files,
    "remember_memory": remember_memory,
    "recall_memory": recall_memory,
    "debug_python_file": debug_python_file
}