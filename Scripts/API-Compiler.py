import os
import json
import re

from solcx import compile_standard, install_solc


####################################### Get the epxlicit version using RegEx #############################################
def extract_version(expression):
    # Define a regular expression pattern to match version numbers
    pattern = r'(\d+\.\d+\.\d+)'
    
    # Search for the version number in the expression
    match = re.search(pattern, expression)
    
    if match:
        # If a match is found, return the version number
        return match.group(1)
    else:
        # If no match is found, return None
        return None

    
    

################################################# Compile contract ####################################################        

def compile_contract(code, filename, version):
    install_solc(version)
    
    # Compilation standard
    compiled_sol = compile_standard(
        {
            "language": "Solidity",
            "sources": {filename: {"content": code}},
            "settings": {
                "outputSelection": {
                    "*": {
                        "*": ['abi', 'metadata', 'evm.bytecode', 'evm.opcodes', 'evm.sourceMap']
                    }
                }
            }
        },
        solc_version=version
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
            
            


def main():
    # Get values:
    code = '''// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;


contract Test {

    uint256 number;

    function setNum (uint256 _number) public {
        number = _number;

        if(number == 3) revert();

    }

 
}'''
    filename = "Hello.sol"
    version = extract_version("compiler version v0.8.25+commit.b61c2a91")
    
    # Compile code
    compile_contract(code, filename, version)

if __name__ == "__main__":
    main()