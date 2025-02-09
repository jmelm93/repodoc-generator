import os
import sys
import logging
from collections import defaultdict
from gitignore_parser import parse_gitignore
import tiktoken


class RepositoryProcessor:
    def __init__(self, directory='./', output_file='combined_docs.txt', gitignore_file='./.gitignore',
                 directories_to_skip=None, file_types_to_capture=None):
        """
        Initializes the RepositoryProcessor with the given configuration.
        """
        self.directory = os.path.abspath(directory)
        self.output_file = output_file
        self.gitignore_file = gitignore_file
        self.directories_to_skip = directories_to_skip or ['venv', '.git', 'notes', 'archive']
        self.file_types_to_capture = file_types_to_capture or []

        # Initialize tokenizer encoding once
        self.tokenizer_encoding = tiktoken.get_encoding("cl100k_base")

        # Configure logging
        logging.basicConfig(
            filename='script.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        # Parse .gitignore for exclusions
        try:
            self.is_ignored = parse_gitignore(self.gitignore_file)
        except Exception as e:
            logging.error(f"Error parsing .gitignore file: {e}")
            self.is_ignored = lambda x: False  # No files are ignored if parsing fails

        # Calculate absolute paths of directories to skip
        self.skip_dirs = [os.path.realpath(os.path.join(self.directory, d)) for d in self.directories_to_skip]

    def get_relative_path(self, path):
        """
        Returns the path relative to the root directory.
        """
        return os.path.relpath(path, self.directory)

    def count_tokens(self, file_path):
        """
        Counts tokens in a text file using the specified encoding.

        Args:
            file_path (str): The path to the file.

        Returns:
            int: The number of tokens in the file.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                tokens = self.tokenizer_encoding.encode(text)
                return len(tokens)
        except UnicodeDecodeError as e:
            logging.error(f"Encoding error in file {file_path}: {e}")
            return 0
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")
            return 0

    def should_capture_file(self, file):
        """
        Check if a file matches the capture rules.

        Args:
            file (str): The filename.

        Returns:
            bool: True if the file should be captured, False otherwise.
        """
        for file_type in self.file_types_to_capture:
            match = file_type['match']
            match_type = file_type['match_type']
            if match_type == 'endswith' and file.endswith(match):
                return True
            elif match_type == 'equals' and file == match:
                return True
        return False

    def generate_repo_metrics(self):
        """
        Generates repository metrics for the output.

        Returns:
            tuple: Total files, total tokens, file metrics, top files by token count.
        """
        file_metrics = defaultdict(lambda: {"count": 0, "tokens": 0})
        total_files = 0
        total_tokens = 0
        token_counts = {}

        try:
            for root, dirs, files in os.walk(self.directory):
                # Normalize root path
                root_realpath = os.path.realpath(root)

                # Modify dirs in-place to skip directories based on their full paths
                dirs[:] = [d for d in dirs if self.should_include_dir(os.path.join(root, d))]

                for file in files:
                    file_path = os.path.join(root, file)
                    if self.is_ignored(file_path):
                        continue
                    if self.should_capture_file(file):
                        token_count = self.count_tokens(file_path)
                        total_files += 1
                        total_tokens += token_count
                        file_extension = os.path.splitext(file)[-1] or file
                        file_metrics[file_extension]["count"] += 1
                        file_metrics[file_extension]["tokens"] += token_count
                        token_counts[file_path] = token_count
        except Exception as e:
            logging.error(f"Error generating repository metrics: {e}")

        # Adjust token_counts keys to be relative paths
        token_counts_rel = {self.get_relative_path(k): v for k, v in token_counts.items()}

        top_files = sorted(token_counts_rel.items(), key=lambda item: item[1], reverse=True)[:5]
        return total_files, total_tokens, file_metrics, top_files

    def generate_repo_structure(self):
        """
        Generates a tree-like textual representation of the repository structure.

        Returns:
            str: The repository structure as a string.
        """
        from pathlib import Path

        structure_lines = []

        def tree(dir_path: Path, prefix: str = ''):
            """A recursive generator that yields a visual tree structure line by line."""
            try:
                contents = sorted([p for p in dir_path.iterdir()
                                   if not self.is_ignored(str(p))
                                   and self.should_include_dir(str(p))])  # full-path check
            except PermissionError as e:
                logging.error(f"Permission error accessing {dir_path}: {e}")
                return
            pointers = ['├── '] * (len(contents) - 1) + (['└── '] if contents else [])
            for pointer, path in zip(pointers, contents):
                yield prefix + pointer + path.name
                if path.is_dir():
                    extension = '│   ' if pointer == '├── ' else '    '
                    yield from tree(path, prefix=prefix + extension)

        try:
            root_path = Path(self.directory)
            structure_lines = [line for line in tree(root_path)]
            structure = '\n'.join(structure_lines)
        except Exception as e:
            logging.error(f"Error generating repository structure: {e}")
            structure = ''

        return structure

    def get_all_files(self):
        """
        Retrieves a list of all files to be processed.

        Returns:
            list: A list of file paths.
        """
        all_files = []

        for root, dirs, files in os.walk(self.directory):
            # Normalize root path
            root_realpath = os.path.realpath(root)

            # Modify dirs in-place to skip directories based on their full paths
            dirs[:] = [d for d in dirs if self.should_include_dir(os.path.join(root, d))]

            for file in files:
                file_path = os.path.join(root, file)
                if not self.is_ignored(file_path) and self.should_capture_file(file):
                    all_files.append(file_path)
        return all_files

    def should_include_dir(self, dir_path, check_full_path=True):
        """
        Determines if a directory should be included based on skip directories and ignore patterns.

        Args:
            dir_path (str): The directory path.
            check_full_path (bool): Whether to compare the full path or just the directory name.

        Returns:
            bool: True if the directory should be included, False otherwise.
        """
        if self.is_ignored(dir_path):
            return False

        dir_realpath = os.path.realpath(dir_path)

        if check_full_path:
            # If the directory is exactly one of the skip directories or is within one, ignore it.
            for skip_dir in self.skip_dirs:
                # This check works even if the directory is nested inside a skip directory.
                if dir_realpath == skip_dir or dir_realpath.startswith(skip_dir + os.sep):
                    return False
        else:
            # When not checking the full path, just compare the directory name.
            if os.path.basename(dir_realpath) in self.directories_to_skip:
                return False

        return True

    def count_output_tokens(self):
        """
        Counts tokens in the combined output file using the specified encoding.

        Returns:
            int: The number of tokens in the output file.
        """
        try:
            with open(self.output_file, "r", encoding="utf-8") as f:
                text = f.read()
                tokens = self.tokenizer_encoding.encode(text)
                return len(tokens)
        except UnicodeDecodeError as e:
            logging.error(f"Encoding error in output file {self.output_file}: {e}")
            return 0
        except Exception as e:
            logging.error(f"Error reading output file {self.output_file}: {e}")
            return 0

    def write_combined_docs(self):
        """
        Generates the combined document of the repository files.
        """
        try:
            with open(self.output_file, 'w', encoding='utf-8') as outfile:
                # Generate repository metrics
                total_files, total_tokens, file_metrics, top_files = self.generate_repo_metrics()

                # Write the document introduction
                outfile.write("****** DOCUMENT INTRODUCTION ******\n\n")
                outfile.write(
                    "This file is a merged representation of the entire codebase, combining all repository files into a single document.\n"
                )
                outfile.write("\n================================================================\n")
                outfile.write("File Summary\n")
                outfile.write("================================================================\n\n")
                outfile.write("Purpose:\n--------\n")
                outfile.write(
                    "This file contains a packed representation of the entire repository's contents.\n"
                    "It is designed to be easily consumable by AI systems for analysis, code review,\n"
                    "or other automated processes.\n\n"
                )
                outfile.write("File Format:\n------------\n")
                outfile.write(
                    "The content is organized as follows:\n"
                    "1. This summary section\n"
                    "2. Repository metrics\n"
                    "3. Repository structure\n"
                    "4. Multiple file entries, each consisting of:\n"
                    "  a. A separator line (================)\n"
                    "  b. The file path (File: path/to/file)\n"
                    "  c. Another separator line\n"
                    "  d. The full contents of the file\n"
                    "  e. A blank line\n\n"
                )

                # Write repository metrics
                outfile.write("\n================================================================\n")
                outfile.write("Repository Metrics\n")
                outfile.write("================================================================\n\n")
                outfile.write(f"Total Files (Non-Ignored): {total_files}\n")
                outfile.write(f"Total Tokens: {total_tokens}\n")
                outfile.write("Total Files by Type:\n")
                for file_type, metrics in file_metrics.items():
                    outfile.write(f"    - {file_type}: {metrics['count']} ({metrics['tokens']} tokens)\n")
                outfile.write("\nTop 5 Files by Tokens:\n")
                for file_path, token_count in top_files:
                    outfile.write(f"    - {file_path}: {token_count} tokens\n")

                # Write the repository structure
                outfile.write("\n================================================================\n")
                outfile.write("Repository Structure\n")
                outfile.write("================================================================\n\n")
                repo_structure = self.generate_repo_structure()
                outfile.write(repo_structure)
                outfile.write("\n")

                # Write the repository files
                outfile.write("\n================================================================\n")
                outfile.write("Repository Files\n")
                outfile.write("================================================================\n\n")

                all_files = self.get_all_files()

                # Process files
                for file_path in all_files:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            content = infile.read()
                            relative_file_path = self.get_relative_path(file_path)
                            outfile.write(f"================\nFile: {relative_file_path}\n================\n")
                            outfile.write(content)
                            outfile.write("\n\n")
                    except UnicodeDecodeError as e:
                        logging.error(f"Encoding error in file {file_path}: {e}")
                    except Exception as e:
                        logging.error(f"Error reading file {file_path}: {e}")

            logging.info(f"Combined document generated successfully at {self.output_file}")

            # Count tokens in the output file
            total_output_tokens = self.count_output_tokens()
            print(f"Total tokens in the output file '{self.output_file}': {total_output_tokens}")
            logging.info(f"Total tokens in the output file '{self.output_file}': {total_output_tokens}")

        except Exception as e:
            logging.critical(f"An unexpected error occurred while generating the combined document: {e}")
            sys.exit(1)


if __name__ == "__main__":
    try:
        processor = RepositoryProcessor(
            directory='./client',
            output_file='combined_docs.txt',
            gitignore_file='./.gitignore',
            # You can now include folder paths; for example, to ignore the entire UI components directory:
            directories_to_skip=['venv', '.git', 'notes', 'archive', './client/src/components/ui'],
            file_types_to_capture=[
                {'match': '.py', 'match_type': 'endswith'},
                {'match': '.json', 'match_type': 'endswith'},
                {'match': 'Dockerfile', 'match_type': 'equals'},
            ]
        )
        processor.write_combined_docs()
    except Exception as e:
        logging.critical(f"Script terminated due to an unexpected error: {e}")
        sys.exit(1)
