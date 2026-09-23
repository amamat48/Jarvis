from tools.calculator import calculate
from tools.file_reader import read_file
from tools.file_manager import list_files
from tools.code_runner import run_python_file
from tools.code_search import search_files



TOOLS = {
    "calculate": calculate,
    "read_file": read_file,
    "list_files": list_files,
    "run_python_file": run_python_file,
    "search_files": search_files
}