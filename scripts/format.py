# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "typer>=0.12.5",
#     "black>=24.1.1",
#     "pathspec>=0.12.1",
# ]
# ///
import typer
import pathlib
from typing import Optional
import black
import sys
import pathspec
import os

app = typer.Typer()


def load_gitignore(root_dir: pathlib.Path) -> pathspec.PathSpec:
    """Load .gitignore patterns and create a PathSpec matcher"""
    gitignore_file = root_dir / ".gitignore"
    patterns = []

    if gitignore_file.exists():
        with open(gitignore_file) as f:
            patterns = f.readlines()

    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def format_python_files(path: pathlib.Path = pathlib.Path(".")) -> bool:
    """
    Format all Python files in the given directory and its subdirectories,
    respecting .gitignore patterns.
    Returns True if formatting was successful, False if there were any errors.
    """
    # Load gitignore patterns
    gitignore_spec = load_gitignore(path)

    # Find all Python files recursively
    python_files = []
    for root, _, files in os.walk(str(path)):
        rel_root = os.path.relpath(root, str(path))
        for file in files:
            if not file.endswith(".py"):
                continue

            rel_path = os.path.join(rel_root, file)
            # Skip files that match gitignore patterns
            if gitignore_spec.match_file(rel_path):
                continue

            python_files.append(pathlib.Path(root) / file)

    if not python_files:
        print("No Python files found")
        return True

    error = False
    # Format each file
    for file in python_files:
        try:
            # Read the file content
            content = file.read_text()
            # Format the content using black
            formatted_content = black.format_str(content, mode=black.FileMode())
            # Write back to the file
            file.write_text(formatted_content)
            print(f"Formatted {file}")
        except Exception as e:
            print(f"Error formatting {file}: {e}")
            error = True

    if not error:
        print("Successfully formatted all files")
        return True
    else:
        print("Errors occurred during formatting")
        return False


@app.command()
def format(path: Optional[str] = None):
    """Format all Python files in the specified directory (or current directory if not specified)"""
    target_path = pathlib.Path(path) if path else pathlib.Path(".")
    success = format_python_files(target_path)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    app()
