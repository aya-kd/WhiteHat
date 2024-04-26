import os
import json
from solcx import compile_standard, install_solc

# Install solc compiler
install_solc("0.4.24")
        

def compile_contract(contract_path, filename):
    with open(contract_path, 'r') as file:
        contract_content = file.read()
    
    # Compilation standard
    compiled_sol = compile_standard(
        {
            "language": "Solidity",
            "sources": {filename: {"content": contract_content}},
            "settings": {
                "outputSelection": {
                    "*": {
                        "*": ['abi', 'metadata', 'evm.bytecode', 'evm.opcodes', 'evm.sourceMap']
                    }
                }
            }
        },
        solc_version="0.4.24"
    )

    # Get file name without extension
    file = os.path.splitext(filename)[0]
    with open(f"compiled_{file}.json", 'w') as file:
        json.dump(compiled_sol, file)

    contract_name = next(iter(compiled_sol["contracts"][filename]))
    bytecode = compiled_sol["contracts"][filename][contract_name]["evm"]["bytecode"]["object"]
    opcodes = compiled_sol["contracts"][filename][contract_name]["evm"]["bytecode"]["opcodes"]
    abi = compiled_sol["contracts"][filename][contract_name]["abi"]

    print(f"Bytecode for contract '{contract_name}':")
    print(bytecode)
    print(opcodes)
    print("==================================================")
    

    return bytecode, opcodes



# Function to iterate through contracts folder and compile each contract
def compile_all_contracts(folder_path, vulnerability):
    with open(f'{vulnerability}_bytecode.csv', 'a', newline='') as csvfile:
        fieldnames = ['bytecode', 'opcodes', 'vulnerability']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for filename in os.listdir(folder_path):
            if filename.endswith('.sol'):
                contract_path = os.path.join(folder_path, filename)
                bytecode, opcodes = compile_contract(contract_path, filename)
                # write the bytecode and opcodes to a csv file
                writer.writerow({'bytecode': bytecode, 'opcodes': opcodes, 'vulnerability': vulnerability})
            
            


def main():
    # Change folder_path according to your needs
    folder_path = './Dataset/BN'
    folder_name = os.path.basename(folder_path)
    # Folder_name is the vulnerability name -> make sure to name the folder correctly!
    compile_all_contracts(folder_path, folder_name)

if __name__ == "__main__":
    main()
