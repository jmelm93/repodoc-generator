# RepoDoc Generator

Efficiently compile your code repository into a single, structured document optimized for AI analysis.

## Features

- **.gitignore Support**: Automatically excludes files and directories based on your `.gitignore`.
- **Customizable File Types**: Specify which file types to include in the documentation.
- **Token Counting**: Provides the total number of tokens in the generated document.
- **Comprehensive Metrics**: Generates repository metrics and highlights top files by token count.
- **Robust Error Handling**: Logs errors and handles exceptions gracefully.
- **Relative Paths**: Displays all file paths relative to the repository root.

## Installation

1. **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/repo-doc-generator.git
    cd repo-doc-generator
    ```

2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

    *Alternatively, install manually:*
    ```bash
    pip install tiktoken gitignore_parser
    ```

## Usage

Run the script using Python:

```bash
python repo_doc_generator.py
```

## Configuration
You can customize the script by modifying the parameters in the RepositoryProcessor instance within repo_doc_generator.py:

- `directory`: Root directory to process (default: ./)
- `output_file`: Name of the output document (default: combined_docs.txt)
- `gitignore_file`: Path to your .gitignore file (default: ./.gitignore)
- `directories_to_skip`: List of directories to exclude (default: ['venv', '.git', 'notes', 'archive'])
- `file_types_to_capture`: List of file patterns to include (default: Python, JSON, Dockerfile)

## Example

```python
processor = RepositoryProcessor(
    directory='./',
    output_file='combined_docs.txt',
    gitignore_file='./.gitignore',
    directories_to_skip=['venv', '.git', 'notes', 'archive'],
    file_types_to_capture=[
        {'match': '.py', 'match_type': 'endswith'},
        {'match': '.json', 'match_type': 'endswith'},
        {'match': 'Dockerfile', 'match_type': 'equals'},
    ]
)
processor.write_combined_docs()
```

## Logs

All logs are saved to `repodoc.log` for debugging and auditing purposes.
