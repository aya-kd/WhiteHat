from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import os
import json
import re

from solcx import compile_standard, install_solc

app = Flask(__name__)

# Load the trained model 
# model = tf.keras.models.load_model('path/to/your/model')


# Get the epxlicit version using RegEx 
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



# Compile contract

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
    opcodes = compiled_sol["contracts"][filename][contract_name]["evm"]["bytecode"]["opcodes"]
    

    print(f"Opcodes for contract '{contract_name}':")
    print(opcodes)
    print("==================================================")
    return opcodes
            


# Define a route for the prediction endpoint
# @app.route('/predict', methods=['POST'])
# def predict():
@app.route("/")
def home():
    
    # Get input data from the request 
    # data = request.json
    
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
    
    # Preprocess the input data (if necessary)
    opcodes = compile_contract(code, filename, version)
    
    # Perform prediction using the loaded model
    # prediction = model.predict(input_data)
    
    # Prepare the response
    # response = {'prediction': prediction.tolist()}
    
    # Return the response as JSON
    # return jsonify(response)
    
    return opcodes

if __name__ == '__main__':
    app.run(debug=True)