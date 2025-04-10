from collections.abc import Iterable


def write_filelist_to_disk(files: Iterable[str], filelist_path: str):
    # set label to be same as index on this list
    files_with_label = [[file, str(i)] for i, file in enumerate(files)]
    contents = "\n".join([" ".join(pair) for pair in files_with_label])

    with open(filelist_path, "w", encoding="utf-8") as fp:
        fp.write(contents)
