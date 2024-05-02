########################################## Printing files encoding ############################################
import os
import chardet

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        rawdata = f.read()
    encoding = chardet.detect(rawdata)['encoding']
    return encoding

def print_file_encodings(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            encoding = detect_encoding(file_path)
            print(f"{file_name}: {encoding}")

folder_path = './Dataset/RE/RE-24'
print_file_encodings(folder_path)
