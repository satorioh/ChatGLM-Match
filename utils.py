import os


def get_abs_path(path: str) -> str:
    current_path = os.path.abspath(__file__)
    dir_name, file_name = os.path.split(current_path)
    return os.path.join(dir_name, path)


def get_file_list(path, suffix='.pdf') -> []:
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(suffix):
                file_list.append(os.path.join(root, file))
    return file_list
