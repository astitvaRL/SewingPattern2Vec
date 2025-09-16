import os
import json
import natsort

json_save_path = "data_info\\test_info.json"

# root_dir = 'D:\\GarmentGen\\GarmentCodeData\\GarmentCodeData_v2'
root_dir = 'D:\\GarmentGen\\GarmentCode\\SAMPLED_garments\\'
folder_names = ['shirt']


info_dict = {}
for folder_name in folder_names:
    folder_dict = {}
    folder_path = os.path.join(root_dir, folder_name)
    sub_folder_names = natsort.natsorted(os.listdir(folder_path))
    for sub_folder_name in sub_folder_names:
        sub_folder_dict = {"body_name": "00000"}
        folder_dict[sub_folder_name] = sub_folder_dict
    info_dict[folder_name] = folder_dict

with open(json_save_path, 'w') as f:
    json.dump(info_dict, f, indent=2)
