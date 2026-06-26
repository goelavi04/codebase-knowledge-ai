# repo_parser.py
# Clones a GitHub repo and parses every Python file into structured data
# using Python's built-in ast (Abstract Syntax Tree) module

import os
import ast
import shutil
from git import Repo


def clone_repo(repo_url: str, repo_name: str) -> str:
    """
    Clones a GitHub repo to a local folder under cloned_repos/{repo_name}.
    If it already exists locally, deletes it first for a clean clone.
    """
    clone_path = os.path.join("cloned_repos", repo_name)

    if os.path.exists(clone_path):
        shutil.rmtree(clone_path)

    Repo.clone_from(repo_url, clone_path)
    return clone_path


def find_python_files(repo_path: str) -> list:
    """
    Walks the cloned repo and returns a list of all .py file paths,
    skipping common folders we don't care about (venv, node_modules, etc).
    """
    skip_dirs = {"venv", "env", "__pycache__", "node_modules", ".git", "tests"}
    python_files = []

    for root, dirs, files in os.walk(repo_path):
        # Modify dirs in-place to skip unwanted folders during the walk
        dirs[:] = [d for d in dirs if d not in skip_dirs]

        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                python_files.append(full_path)

    return python_files


def parse_python_file(file_path: str) -> dict:
    """
    Parses a single Python file using ast and extracts:
    - imports (what this file depends on)
    - functions (name + the code that calls inside it)
    - classes (name + its methods)
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        source_code = f.read()

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        # Some files might have syntax errors or be incompatible - skip them
        return None

    imports = []
    functions = []
    classes = []

    for node in ast.walk(tree):
        # Capture "import x" and "from x import y"
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)

        # Capture function definitions
        elif isinstance(node, ast.FunctionDef):
            calls = extract_function_calls(node)
            functions.append({
                "name": node.name,
                "calls": calls,
                "docstring": ast.get_docstring(node) or ""
            })

        # Capture class definitions
        elif isinstance(node, ast.ClassDef):
            method_names = [
                n.name for n in node.body if isinstance(n, ast.FunctionDef)
            ]
            classes.append({
                "name": node.name,
                "methods": method_names,
                "docstring": ast.get_docstring(node) or ""
            })

    return {
        "file_path": file_path,
        "imports": imports,
        "functions": functions,
        "classes": classes,
        "source_code": source_code
    }


def extract_function_calls(function_node: ast.FunctionDef) -> list:
    """
    Given a function's AST node, finds every function call made inside it.
    E.g. inside def login(): validate_password() -> returns ["validate_password"]
    """
    calls = []

    for node in ast.walk(function_node):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)

    return calls


def parse_repo(repo_url: str, repo_name: str) -> list:
    """
    Main entry point: clones the repo, finds all Python files,
    parses each one, and returns a list of parsed file data.
    """
    repo_path = clone_repo(repo_url, repo_name)
    python_files = find_python_files(repo_path)

    parsed_files = []
    for file_path in python_files:
        parsed = parse_python_file(file_path)
        if parsed:
            parsed_files.append(parsed)

    return parsed_files