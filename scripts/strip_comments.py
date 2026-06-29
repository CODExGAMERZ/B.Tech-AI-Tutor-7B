import os
import json
import argparse

def strip_py_comments(filepath):
    print(f"Stripping comments from Python file: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    new_lines = []
    for line in lines:
        # Check if the line, when stripped of whitespace, starts with '#'
        if line.strip().startswith('#'):
            # Keep shebang if present
            if line.startswith('#!'):
                new_lines.append(line)
            continue
        new_lines.append(line)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

def strip_ipynb_comments(filepath):
    print(f"Stripping comments from Jupyter Notebook: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            notebook = json.load(f)
        except Exception as e:
            print(f"Error reading notebook {filepath}: {e}")
            return
            
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            new_source = []
            for line in source:
                # Check if the line starts with '#' after optional whitespace
                if line.strip().startswith('#'):
                    continue
                new_source.append(line)
            cell['source'] = new_source
            
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)

def main():
    parser = argparse.ArgumentParser(description="Strip all comment lines from Python files and Jupyter Notebooks")
    parser.add_argument("--dir", type=str, default=".", help="Directory to process")
    args = parser.parse_args()
    
    for root, dirs, files in os.walk(args.dir):
        # Skip hidden directories like .git
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            path = os.path.join(root, file)
            if file.endswith('.py'):
                # Avoid self-processing
                if file == 'strip_comments.py':
                    continue
                strip_py_comments(path)
            elif file.endswith('.ipynb'):
                strip_ipynb_comments(path)

if __name__ == "__main__":
    main()
