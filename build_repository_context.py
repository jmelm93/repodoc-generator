import os
import sys
import logging
from collections import defaultdict
from gitignore_parser import parse_gitignore
import tiktoken
from pathlib import Path


class RepositoryProcessor:
    def __init__(self, directory='./', output_file='combined_docs.txt', gitignore_file='./.gitignore',
                 directories_to_skip=None, file_types_to_capture=None,
                 special_exclude_dir_root=None, filenames_to_skip=None):
        """
        Initializes the RepositoryProcessor with the given configuration.

        Args:
            directory (str): The root directory of the repository to process.
            output_file (str): The path to the output file.
            gitignore_file (str): The path to the .gitignore file.
            directories_to_skip (list): A list of directory names or paths (relative to 'directory') to skip entirely.
            file_types_to_capture (list): A list of dictionaries specifying file types to capture.
                                          If empty or None, captures all files not otherwise excluded.
            special_exclude_dir_root (str, optional): A specific directory path (relative to 'directory')
                                                      where files directly within it should be excluded,
                                                      but files in its subdirectories should be included.
                                                      Example: 'client/src/components'. Defaults to None.
            filenames_to_skip (list, optional): A list of exact filenames (e.g., ['config.py', 'NOTES.md'])
                                                to skip regardless of their directory. Defaults to None.
        """
        self.directory = os.path.abspath(directory)
        self.output_file = output_file
        self.gitignore_file = gitignore_file
        self.directories_to_skip = directories_to_skip or ['venv', '.git', 'notes', 'archive']
        self.file_types_to_capture = file_types_to_capture or []
        self.special_exclude_dir_root = special_exclude_dir_root
        self.skip_filenames_set = set(filenames_to_skip or [])

        # --- New: Cache for token counts ---
        self.file_token_cache = {}
        # ---

        # Calculate the absolute path for the special exclusion directory if provided
        self.special_exclude_dir_abs = None
        if self.special_exclude_dir_root:
            # Handle cases where special_exclude_dir_root might be '.' or empty
            if self.special_exclude_dir_root and self.special_exclude_dir_root != '.':
                 normalized_special_path = os.path.join(*self.special_exclude_dir_root.split('/'))
                 self.special_exclude_dir_abs = os.path.realpath(os.path.join(self.directory, normalized_special_path))
                 logging.info(f"Special exclusion rule active for files directly within: {self.special_exclude_dir_abs}")
            else:
                 # If special root is '.' or empty, it refers to the main directory being processed
                 self.special_exclude_dir_abs = self.directory
                 logging.info(f"Special exclusion rule active for files directly within the root directory: {self.directory}")


        # Initialize tokenizer encoding once
        self.tokenizer_encoding = tiktoken.get_encoding("cl100k_base")

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        # Parse .gitignore for exclusions
        try:
            abs_gitignore_path = os.path.abspath(self.gitignore_file)
            if not os.path.exists(abs_gitignore_path):
                 # Try relative to the target directory as well
                 abs_gitignore_path_rel = os.path.join(self.directory, self.gitignore_file)
                 if os.path.exists(abs_gitignore_path_rel):
                     abs_gitignore_path = abs_gitignore_path_rel
                 else:
                     abs_gitignore_path = None # Indicate not found

            if abs_gitignore_path and os.path.exists(abs_gitignore_path):
                self.is_ignored = parse_gitignore(abs_gitignore_path)
                logging.info(f"Loaded .gitignore rules from: {abs_gitignore_path}")
            else:
                logging.warning(f".gitignore file not found at specified/relative paths. No gitignore rules applied.")
                self.is_ignored = lambda x: False
        except Exception as e:
            logging.error(f"Error parsing .gitignore file '{self.gitignore_file}': {e}", exc_info=True)
            self.is_ignored = lambda x: False

        # Calculate absolute paths of directories to skip
        self.skip_dirs_abs = set()
        for d in self.directories_to_skip:
            # Normalize path separators for comparison
            d_normalized = os.path.join(*d.split('/'))
            skip_path_abs = os.path.realpath(os.path.join(self.directory, d_normalized))
            self.skip_dirs_abs.add(skip_path_abs)

        logging.info(f"Original directories_to_skip: {self.directories_to_skip}")
        logging.info(f"Absolute paths calculated for skip_dirs: {self.skip_dirs_abs}")
        if self.skip_filenames_set:
            logging.info(f"Skipping files with exact names: {self.skip_filenames_set}")


    def get_relative_path(self, path):
        """
        Returns the path relative to the root directory.
        """
        try:
            return os.path.relpath(path, self.directory)
        except ValueError:
             # Handle cases where path might not be under self.directory (e.g., symlinks outside)
             logging.warning(f"Could not get relative path for {path} against {self.directory}")
             return path # Return absolute path as fallback


    def count_tokens(self, file_path):
        """
        Counts tokens in a text file using the specified encoding.
        Returns 0 if the file cannot be read or decoded.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            try:
                # Fallback encoding
                with open(file_path, "r", encoding="latin-1") as f:
                    text = f.read()
                logging.warning(f"Encoding error (UTF-8) in file {self.get_relative_path(file_path)}, read with latin-1.")
            except Exception as e_fallback:
                logging.error(f"Could not read file {self.get_relative_path(file_path)} with UTF-8 or latin-1: {e_fallback}")
                return 0
        except FileNotFoundError:
             logging.error(f"File not found during token counting: {self.get_relative_path(file_path)}")
             return 0
        except Exception as e:
            logging.error(f"Error reading file {self.get_relative_path(file_path)} for token counting: {e}")
            return 0

        try:
            tokens = self.tokenizer_encoding.encode(text)
            return len(tokens)
        except Exception as e:
             logging.error(f"Error encoding tokens for file {self.get_relative_path(file_path)}: {e}")
             return 0


    def should_capture_file_type(self, file_name):
        """
        Check if a file matches the capture rules based on file type/name.
        """
        if not self.file_types_to_capture:
            return True
        for file_type in self.file_types_to_capture:
            match = file_type['match']
            match_type = file_type['match_type']
            if match_type == 'endswith' and file_name.endswith(match):
                return True
            elif match_type == 'equals' and file_name == match:
                return True
        return False

    def should_exclude_special_case(self, file_path):
        """
        Checks if the file is directly within the special_exclude_dir_root.
        """
        if not self.special_exclude_dir_abs:
            return False
        try:
            # Use resolved paths to handle symlinks consistently
            parent_dir_abs = str(Path(file_path).resolve().parent)
            special_exclude_resolved = str(Path(self.special_exclude_dir_abs).resolve())
            if parent_dir_abs == special_exclude_resolved:
                return True
        except Exception as e:
             logging.warning(f"Error resolving path for special case check {file_path}: {e}")
        return False


    def should_process_file(self, file_path, file_name):
        """
        Determines if a file should be processed based on all rules.
        """
        file_path_abs = os.path.realpath(file_path) # Use real path for checks

        # 1. Check .gitignore
        if self.is_ignored(file_path_abs):
            return False

        # 2. Check specific filename skip list
        if file_name in self.skip_filenames_set:
            return False

        # 3. Check special exclusion rule
        if self.should_exclude_special_case(file_path_abs):
            return False

        # 4. Check if file type should be captured
        if not self.should_capture_file_type(file_name):
            return False

        # 5. Basic file system check (ensure it's actually a file)
        if not os.path.isfile(file_path_abs):
             # This might happen with broken symlinks etc.
             # logging.warning(f"Path is not a file: {self.get_relative_path(file_path_abs)}")
             return False

        return True

    # --- New Method: Pre-calculate token counts ---
    def _calculate_and_cache_token_counts(self):
        """
        Walks the directory, identifies processable files, counts their tokens,
        and stores the counts in self.file_token_cache.
        """
        logging.info("Calculating token counts for processable files...")
        self.file_token_cache = {} # Reset cache
        processed_files_count = 0

        for root, dirs, files in os.walk(self.directory, topdown=True):
            root_realpath = os.path.realpath(root)
            # Prune directories based on skip rules BEFORE iterating files within them
            dirs[:] = [d for d in dirs if self.should_include_dir(os.path.join(root_realpath, d))]

            for file_name in files:
                file_path = os.path.join(root_realpath, file_name)
                # Use the consistent should_process_file check
                if self.should_process_file(file_path, file_name):
                    # Use realpath as the key for consistency
                    file_path_abs = os.path.realpath(file_path)
                    token_count = self.count_tokens(file_path_abs)
                    self.file_token_cache[file_path_abs] = token_count
                    processed_files_count += 1

        logging.info(f"Token counts calculated and cached for {processed_files_count} files.")
    # --- End New Method ---


    def generate_repo_metrics(self):
        """
        Generates repository metrics using the pre-calculated token counts.
        """
        file_metrics = defaultdict(lambda: {"count": 0, "tokens": 0})
        total_files = 0
        total_tokens = 0
        token_counts_rel = {} # Store relative paths here for top files output

        # Iterate through the cached token counts
        for file_path_abs, token_count in self.file_token_cache.items():
            # No need to call should_process_file again, cache only contains processable files
            try:
                file_name = os.path.basename(file_path_abs)
                relative_path = self.get_relative_path(file_path_abs)

                total_files += 1
                total_tokens += token_count
                file_extension = os.path.splitext(file_name)[-1] or file_name # Handle no extension
                file_metrics[file_extension]["count"] += 1
                file_metrics[file_extension]["tokens"] += token_count
                token_counts_rel[relative_path] = token_count
            except Exception as e:
                 logging.error(f"Error processing cached file {file_path_abs} for metrics: {e}")


        top_files = sorted(token_counts_rel.items(), key=lambda item: item[1], reverse=True)[:5]
        return total_files, total_tokens, file_metrics, top_files


    def generate_repo_structure(self):
        """
        Generates a tree-like textual representation of the repository structure,
        including token counts for files, respecting all skip rules.
        """
        structure_lines = []
        root_path = Path(self.directory)

        # --- Inner recursive function ---
        def tree(dir_path: Path, prefix: str = ''):
            try:
                contents = []
                # Iterate and filter items in the current directory
                for p in dir_path.iterdir():
                    try:
                        p_abs = str(p.resolve()) # Use resolved absolute path for checks
                        p_name = p.name

                        # --- Filtering Logic ---
                        # 1. Gitignore check
                        if self.is_ignored(p_abs): continue

                        # 2. Directory specific checks
                        if p.is_dir():
                            if not self.should_include_dir(p_abs): continue
                        # 3. File specific checks
                        elif p.is_file():
                            if p_name in self.skip_filenames_set: continue
                            if self.should_exclude_special_case(p_abs): continue
                            # Check if file type is captured (important for structure consistency)
                            if not self.should_capture_file_type(p_name): continue
                            # Final check: ensure it's in our cache (meaning it passed all checks)
                            if p_abs not in self.file_token_cache: continue
                        # 4. Handle other types like symlinks (optional: decide if they should be listed)
                        #    Currently, only files and included directories are added.
                        else:
                             continue # Skip symlinks, sockets, etc.

                        # If all checks pass, add to contents for this level
                        contents.append(p)

                    except OSError as oe:
                         # Handle errors like broken symlinks during iteration/resolve
                         logging.warning(f"OS error processing path {p} in {dir_path}: {oe}")
                         continue
                    except Exception as ie:
                         logging.error(f"Unexpected error processing path {p} in {dir_path}: {ie}", exc_info=True)
                         continue


                # Sort directories first, then files, alphabetically within each group
                contents.sort(key=lambda p: (not p.is_dir(), p.name.lower()))

            except PermissionError as e:
                logging.warning(f"Permission error accessing {dir_path}: {e}")
                return # Stop recursion for this branch
            except FileNotFoundError as e:
                 logging.warning(f"Directory not found during tree generation {dir_path}: {e}")
                 return # Stop recursion for this branch
            except Exception as e:
                 logging.error(f"Error listing directory contents for {dir_path}: {e}", exc_info=True)
                 return

            # --- Generate output lines for this level ---
            pointers = ['├── '] * (len(contents) - 1) + ['└── '] if contents else []
            for pointer, path in zip(pointers, contents):
                entry_name = path.name
                # --- Append token count for files ---
                if path.is_file():
                    p_abs = str(path.resolve())
                    token_count = self.file_token_cache.get(p_abs, None) # Get from cache
                    if token_count is not None:
                        entry_name += f" ({token_count} tokens)"
                    else:
                        # This case should ideally not happen if caching is done correctly
                        entry_name += " (tokens N/A)"
                        logging.warning(f"Token count missing in cache for file in structure: {p_abs}")
                # --- Yield the line ---
                yield prefix + pointer + entry_name

                # --- Recurse into subdirectories ---
                if path.is_dir():
                    extension = '│   ' if pointer == '├── ' else '    '
                    yield from tree(path, prefix=prefix + extension)
        # --- End of inner function ---

        try:
            # Start the tree generation
            structure_lines.append(f"{root_path.name}/") # Add trailing slash to indicate root dir
            structure_lines.extend(list(tree(root_path)))
            structure = '\n'.join(structure_lines)
        except Exception as e:
            logging.error(f"Error generating repository structure: {e}", exc_info=True)
            structure = 'Error generating structure.'

        return structure


    def get_all_files(self):
        """
        Retrieves a list of absolute file paths that should be processed,
        using the pre-calculated token cache keys.
        """
        # The keys of the token cache are exactly the files that passed all checks
        return list(self.file_token_cache.keys())


    def should_include_dir(self, dir_path):
        """
        Determines if a directory should be included based on skip directories
        (absolute paths) and ignore patterns.

        Args:
            dir_path (str): The absolute directory path.

        Returns:
            bool: True if the directory should be included, False otherwise.
        """
        try:
            dir_realpath = os.path.realpath(dir_path) # Resolve symlinks for consistent checks

            # 1. Check if the exact absolute path is in the calculated skip set
            if dir_realpath in self.skip_dirs_abs:
                # logging.debug(f"Skipping directory via absolute path match: {self.get_relative_path(dir_realpath)}")
                return False

            # 2. Check if the directory is ignored by .gitignore
            #    Need to check with and without trailing slash for gitignore compatibility
            if self.is_ignored(dir_realpath) or self.is_ignored(dir_realpath + os.sep):
                # logging.debug(f"Skipping directory via .gitignore: {self.get_relative_path(dir_realpath)}")
                return False

            # 3. Check if any parent directory's absolute path is in skip_dirs_abs
            #    This prevents walking into subdirs of explicitly skipped paths.
            parent = Path(dir_realpath).parent
            root_dir_resolved = Path(self.directory).resolve()
            # Stop when we go above the root processing directory or hit the filesystem root
            while parent != parent.parent and parent.resolve() != root_dir_resolved.parent:
                if str(parent.resolve()) in self.skip_dirs_abs:
                    # logging.debug(f"Skipping directory because parent '{self.get_relative_path(str(parent.resolve()))}' is skipped: {self.get_relative_path(dir_realpath)}")
                    return False
                parent = parent.parent

            # If none of the above apply, include the directory
            return True

        except OSError as e:
             # Handle cases like broken symlinks when checking paths
             logging.warning(f"OS error checking directory inclusion for {dir_path}: {e}")
             return False
        except Exception as e:
             logging.error(f"Unexpected error checking directory inclusion for {dir_path}: {e}", exc_info=True)
             return False


    def count_output_tokens(self):
        """
        Counts tokens in the combined output file.
        """
        try:
            with open(self.output_file, "r", encoding="utf-8") as f:
                text = f.read()
                tokens = self.tokenizer_encoding.encode(text)
                return len(tokens)
        except FileNotFoundError:
            logging.error(f"Output file {self.output_file} not found for token counting.")
            return 0
        except Exception as e:
            logging.error(f"Error reading output file {self.output_file}: {e}")
            return 0

    def write_combined_docs(self):
        """
        Generates the combined document of the repository files.
        """
        logging.info(f"Starting processing for directory: {self.directory}")
        logging.info(f"Output file: {self.output_file}")

        try:
            # --- Step 1: Pre-calculate token counts for all processable files ---
            self._calculate_and_cache_token_counts()

            # --- Step 2: Generate metrics using the cached counts ---
            logging.info("Generating repository metrics from cached data...")
            total_files, total_tokens, file_metrics, top_files = self.generate_repo_metrics()
            logging.info(f"Metrics generated: {total_files} files, {total_tokens} tokens.")

            # --- Step 3: Generate structure using cached counts ---
            logging.info("Generating repository structure with token counts...")
            repo_structure = self.generate_repo_structure()
            logging.info("Repository structure generated.")

            # --- Step 4: Get the final list of files to include (from cache keys) ---
            logging.info("Collecting files to include from cache...")
            all_files_to_include = self.get_all_files() # Uses cache keys
            logging.info(f"Found {len(all_files_to_include)} files to include in the output.")

            # --- Step 5: Write the output file ---
            logging.info(f"Writing combined document to {self.output_file}...")
            with open(self.output_file, 'w', encoding='utf-8') as outfile:
                # Write header sections (Introduction, Summary, Metrics, Structure)
                outfile.write("****** DOCUMENT INTRODUCTION ******\n\n")
                outfile.write(
                    "This file is a merged representation of the codebase, combining selected repository files into a single document.\n"
                    "Files ignored by .gitignore, specific skip rules (directories, filenames, special cases), or file type filters are excluded.\n"
                )
                outfile.write("\n================================================================\n")
                outfile.write("File Summary\n")
                outfile.write("================================================================\n\n")
                outfile.write("Purpose:\n--------\n")
                outfile.write(
                    "This file contains a packed representation of the repository's relevant contents.\n"
                    "It is designed to be easily consumable by AI systems for analysis, code review,\n"
                    "or other automated processes.\n\n"
                )
                outfile.write("File Format:\n------------\n")
                outfile.write(
                    "The content is organized as follows:\n"
                    "1. This summary section\n"
                    "2. Repository metrics (based on included files)\n"
                    "3. Repository structure (visual tree with token counts, respecting ignores/skips)\n" # Updated description
                    "4. Multiple file entries, each consisting of:\n"
                    "  a. A separator line (================)\n"
                    "  b. The file path relative to the root (File: path/to/file)\n"
                    "  c. Another separator line\n"
                    "  d. The full contents of the file\n"
                    "  e. A blank line\n\n"
                )

                outfile.write("\n================================================================\n")
                outfile.write("Repository Metrics (Included Files)\n")
                outfile.write("================================================================\n\n")
                outfile.write(f"Total Files Included: {total_files}\n")
                outfile.write(f"Total Tokens (Included Files): {total_tokens}\n")
                if file_metrics:
                    outfile.write("Included Files by Type:\n")
                    for file_type, metrics in sorted(file_metrics.items()):
                        outfile.write(f"    - {file_type}: {metrics['count']} ({metrics['tokens']} tokens)\n")
                else:
                     outfile.write("No files included based on current filters.\n")
                if top_files:
                    outfile.write("\nTop 5 Included Files by Tokens:\n")
                    for file_path, token_count in top_files:
                        # Ensure consistent path separators in output
                        outfile.write(f"    - {file_path.replace(os.sep, '/')}: {token_count} tokens\n")
                outfile.write("\n")

                outfile.write("\n================================================================\n")
                outfile.write("Repository Structure (Filtered, with Token Counts)\n") # Updated title
                outfile.write("================================================================\n\n")
                outfile.write(repo_structure)
                outfile.write("\n")

                # Write the repository files content
                outfile.write("\n================================================================\n")
                outfile.write("Repository Files Content\n")
                outfile.write("================================================================\n\n")

                # Sort files by path for consistent output order
                all_files_to_include.sort()

                for file_path_abs in all_files_to_include:
                    try:
                        with open(file_path_abs, 'r', encoding='utf-8', errors='replace') as infile:
                            content = infile.read()
                            relative_file_path = self.get_relative_path(file_path_abs)
                            relative_file_path_unix = relative_file_path.replace(os.sep, '/')
                            outfile.write(f"================\nFile: {relative_file_path_unix}\n================\n")
                            outfile.write(content)
                            outfile.write("\n\n") # Ensure separation between files
                    except FileNotFoundError:
                         # Should not happen if cache is correct, but handle defensively
                         logging.warning(f"File from cache not found during writing: {file_path_abs}")
                    except Exception as e:
                        logging.error(f"Error reading/writing file content for {self.get_relative_path(file_path_abs)}: {e}")

            logging.info(f"Combined document generated successfully at {self.output_file}")

            total_output_tokens = self.count_output_tokens()
            if total_output_tokens > 0:
                print(f"Total tokens in the final output file '{self.output_file}': {total_output_tokens}")
                logging.info(f"Total tokens in the final output file '{self.output_file}': {total_output_tokens}")

        except Exception as e:
            logging.critical(f"An unexpected error occurred during write_combined_docs: {e}", exc_info=True)


if __name__ == "__main__":
    try:
        # --- Configuration ---
        target_directory = './client/src'
        output_filename = 'repodoc.txt'
        gitignore_path = './.gitignore' # Assumes .gitignore is in the directory where the script is run

        # Directories to skip - paths should be relative to target_directory
        # Or just names like 'node_modules' if they can appear anywhere
        dirs_to_skip = [
            # Simple names (will be skipped anywhere inside target_directory)
            'node_modules',
            'venv',
            '.git',
            'notes',
            'archive',
            'build',
            'dist',
            # Paths relative to target_directory ('./client/src')
            'components/ui',
            'sections/results',
            'sections/profile',
            'contexts',
            'components/layouts', # Corrected missing comma from user input
            'components/skeletons', # Corrected missing comma from user input
            'lib/test-utils', # Example: skip test utilities
            '__tests__', # Example: skip test directories by name
            'mocks',
        ]

        # File types to include
        types_to_capture = [
            {'match': '.tsx', 'match_type': 'endswith'},
            {'match': '.ts', 'match_type': 'endswith'},
            # Add other types if needed
            # {'match': '.css', 'match_type': 'endswith'},
            # {'match': 'package.json', 'match_type': 'equals'},
        ]

        # # Special rule: exclude files directly in 'components' relative to target_directory
        # special_exclude = 'components'

        # Specific filenames to skip globally within target_directory
        files_to_skip = [
            'RequestTool.tsx',
            'setupTests.ts', # Example
            '.DS_Store',
            'reportWebVitals.ts', # Example
        ]
        # --- End Configuration ---


        processor = RepositoryProcessor(
            directory=target_directory,
            output_file=output_filename,
            gitignore_file=gitignore_path,
            directories_to_skip=dirs_to_skip,
            file_types_to_capture=types_to_capture,
            # special_exclude_dir_root=special_exclude,
            filenames_to_skip=files_to_skip
        )
        processor.write_combined_docs()
        sys.exit(0) # Indicate success

    except Exception as e:
        logging.critical(f"Script terminated due to an unexpected error: {e}", exc_info=True)
        sys.exit(1) # Indicate failure
