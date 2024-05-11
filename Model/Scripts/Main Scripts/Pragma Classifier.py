####################################  Move contracts with certain RegEx (i.e, version) to a different folder  ###########################################
import re
import os
import shutil

def move_contracts_with_pragma(source_folder, destination_folder):
    #contracts_with_pragma = []
    #pragma_pattern = re.compile(r'^\s*pragma solidity\s+(\^\d+\.\d+\.\d+);')
    pattern = re.compile(r'0.4.24')
    #pattern = re.compile(r'^(?!.*\d+\.\d+\.\d+)')   # Match any line that does not contain a pragma directive
    #pattern = re.compile(r'import')
    
    # Iterate through all .sol files in the specified folder
    for filename in os.listdir(source_folder):
        if filename.endswith('.sol'):
            file_path = os.path.join(source_folder, filename)
            with open(file_path, 'r', encoding='ISO-8859-1') as f:
                    # Read the content of the file
                content = f.read()
                    # Search for pragma directives
                if re.search(pattern, content):
                    source_path = os.path.join(source_folder, filename)
                    destination_path = os.path.join(destination_folder, filename)
                    shutil.move(source_path, destination_path)
                        

    

# Example usage
source_folder = './Dataset/OF/OF-24'
destination_folder = './Dataset/OF/OF-24/x'
move_contracts_with_pragma(source_folder, destination_folder)
    