import os

def rename_val2():
    path = "dataset/ILSVRC/val2/"
    dir_list = os.listdir(path)

    change_file_dict = {}
    with open("dataset/mapping.txt") as file:
        change_file_list = file.readlines()
        for change_file in change_file_list:
            after_name, before_name = change_file.rstrip("\n").split(", ")
            change_file_dict[before_name] = after_name

    for dir_value in dir_list:
        dir_path = path + dir_value + '/'
        file_list = os.listdir(dir_path)
        for file in file_list:
            os.rename(dir_path + file, path + change_file_dict[dir_value + '/' + file])


if __name__ == "__main__":
    rename_val2()
