import os
import json

json_save_path = "D:\\GarmentGen\\Pattern2Vector\\SewingLDM\\data_info\\test_info.json"

root_dir = 'D:\\GarmentGen\\GarmentCodeData\\GarmentCodeData_v2'
folder_names = ['skirt_levels']


info_dict = {}
for folder_name in folder_names:
    folder_dict = {}
    folder_path = os.path.join(root_dir, folder_name)
    sub_folder_names = os.listdir(folder_path)
    for sub_folder_name in sub_folder_names:
        sub_folder_dict = {"body_name": "00000"}
        folder_dict[sub_folder_name] = sub_folder_dict
    info_dict[folder_name] = folder_dict

with open(json_save_path, 'w') as f:
    json.dump(info_dict, f, indent=2)
