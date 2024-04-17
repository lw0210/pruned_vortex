import os
import json


def get_average_acc_value(parent_folder):
    acc_values = []
    for folder_name in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder_name)
        if os.path.isdir(folder_path):
            json_file_path = os.path.join(folder_path, 'metrics.json')
            if os.path.exists(json_file_path):
                with open(json_file_path) as json_file:
                    data = json.load(json_file)
                    acc_value = data.get('fscore')
                    if acc_value is not None:
                        acc_values.append(acc_value)

    if len(acc_values) > 0:
        average = sum(acc_values) / len(acc_values)
        return average
    else:
        return None


# 调用函数并传入父文件夹的路径
parent_folder = 'output_mine'
average_acc_value = get_average_acc_value(parent_folder)
print(f"The average acc value is: {average_acc_value}")
