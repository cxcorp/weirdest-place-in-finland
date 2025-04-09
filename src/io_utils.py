import os
from os import path


def find_first_file(path_to_dir: str, file_ext: str) -> str:
    for dirpath, _, filenames in os.walk(path_to_dir):
        for filename in filenames:
            if filename.endswith(file_ext):
                return path.abspath(path.join(dirpath, filename))
    return None


def find_all_files_with_extension_recursively(path_to_dir: str, file_ext: str):
    for dirpath, _, filenames in os.walk(path_to_dir):
        for filename in filenames:
            if filename.endswith(file_ext):
                yield path.abspath(path.join(dirpath, filename))
