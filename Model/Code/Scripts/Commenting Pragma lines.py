import os

def comment_pragma(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.sol'):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r') as file:
                lines = file.readlines()
            with open(filepath, 'w') as file:
                for line in lines:
                    if line.strip().startswith('pragma solidity'):
                        file.write('// ' + line)
                    else:
                        file.write(line)

def main():
    folder_path = './Dataset/BN'  # Specify the path to your contracts folder
    comment_pragma(folder_path)
    print("Pragma lines commented out in all Solidity files.")

if __name__ == "__main__":
    main()
