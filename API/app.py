from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS class from flask_cors
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import json
import re
import binascii
from pyevmasm import disassemble_hex
from code_analysis import count_stats 
from custom_metrics import F1Score


app = Flask(__name__)
CORS(app, resources={r"/scan": {"origins": "*"}})  # Wrap your Flask app with CORS

# Load the trained models 
# Binary classification model
model1_path = os.path.join(os.path.dirname(__file__), 'binary1.keras')
model1 = tf.keras.models.load_model(model1_path, custom_objects={'F1Score': F1Score})
# Multi-class classification model
model2_path = os.path.join(os.path.dirname(__file__), 'multi1.keras')
model2 = tf.keras.models.load_model(model2_path, custom_objects={'F1Score': F1Score})

tokenizer = Tokenizer()

replacements = {
    r'\b(ADD|MUL|SUB|DIV|SDIV|MOD|SMOD|ADDMOD|MULMOD|EXP)\b': 'AOP', # Arithmetic Operations
    r'\b(LT|GT|SLT|SGT)\b': 'COM', # Comparison Operations
    r'\b(AND|OR|XOR|NOT)\b': 'LOP', # Logic Operations
    r'\bDUP\d+\b': 'DUP',
    r'\bSWAP\d+\b': 'SWAP',
    r'\bPUSH\d+\b': 'PUSH',
    r'\bLOG\d+\b': 'LOG'
}



###################################### Version #####################################

# Get the explicit version using RegEx
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


####################################### Code analysis ##########################################
def code_analysis(code):
    stats = count_stats(code)
    print("Total number of lines:", stats['total_lines'])
    print("Empty lines:", stats['empty_lines'])
    print("Code lines:", stats['code_lines'])
    print("Comment lines:", stats['comment_lines'])
    print("Public functions:", stats['public_functions'])
    print("Private functions:", stats['private_functions'])
    print("Internal functions:", stats['internal_functions'])
    print("External functions:", stats['external_functions'])
    print("View functions:", stats['view_functions'])
    print("Pure functions:", stats['pure_functions'])
    print("Payable functions:", stats['payable_functions'])
    print("Nonpayable functions:", stats['nonpayable_functions'])
    print("Receives funds:", stats['receives_funds'])
    print("Transfers ETH:", stats['transfers_eth'])
    print("Uses assembly:", stats['uses_assembly'])
    print("Destroyables:", stats['destroyables'])
    print("Delegatecall:", stats['delegatecall'])
    print("Signature or hashing:", stats['signature_or_hashing'])
    print("New/Create/Create2:", stats['new_create_create2'])
    print("Unchecked:", stats['unchecked'])
    return stats

###################################### Preprocessing ##################################

# Bytecode -> Opcodes conversion
def disassemble_bytecode(bytecode):
    # Disassemble the hex bytecode into instructions
    instructions = disassemble_hex(bytecode)
    # Join the instructions into a single string, correctly formatted
    opcodes = ''.join(str(instr).replace('\n', ' ') for instr in instructions)
    print(opcodes)
    
    return opcodes
    

def simplify_opcodes(opcodes):
    # Perform replacements using regular expressions
    for pattern, replacement in replacements.items():
        opcodes = re.sub(pattern, replacement, opcodes)
        # Remove hexadecimal values
        opcodes = re.sub(r'0x[0-9a-fA-F]+', '', opcodes)
    return opcodes



def preprocess(bytecode):
    opcodes = simplify_opcodes(disassemble_bytecode(bytecode))
    return opcodes
    
    

####################################### Classification ################################

# Tokenization and padding function
def tokanize_opcodes(opcode):
    # Tokenization
    tokenizer.fit_on_texts([opcode])
    sequence = tokenizer.texts_to_sequences([opcode])[0]

    # Sequence padding
    max_length = max([len(sequence)])
    sequence = pad_sequences([sequence], maxlen=max_length)[0]
    return sequence


# Binary classification (vulnerable or not vulnerable)       
def classify_vulnerable(opcodes):
    # Tokanize opcode
    sequence = tokanize_opcodes(opcodes)
    
    # Perform prediction
    classification = model1.predict(np.array([sequence]))
    return classification


# Multi-class classification (vulnerability type)  
def classify_vulnerability(opcodes):
    # Tokanize opcode
    sequence = tokanize_opcodes(opcodes)
    
    # Perform prediction
    classification = model2.predict(np.array([sequence]))
    return classification


def print_likelihoods(classification):
    total = np.sum(classification)
    likelihoods = (classification / total) * 100
    for i, likelihood in enumerate(likelihoods[0]):
        print(f'Class {i}: {likelihood:.2f}%')
        
        
 
    

####################################### API ##########################################
@app.route('/scan', methods=['POST'])
def scan():
    try:
        # Get input data from the request 
        data = request.json

        # Get values
        code = data.get('code', '')
        version = extract_version(data.get('version', ''))
        bytecode = data.get('bytecode', '')
        stats = count_stats(code)

        # Preprocess the input data (if necessary)
        opcodes = preprocess(bytecode)
        print("opcodes:", opcodes)

        # Get the predicted class
        vulnerable = classify_vulnerable(opcodes)
        vulnerabilities = classify_vulnerability(opcodes)

        # Print the predicted class
        print('\nVulnerable:', np.argmax(vulnerable))
        
        # Print the predicted vulnerabilities
        print('\nVulnerabilities:', np.argmax(vulnerabilities))

        # Print the likelihoods
        print('\nVulnenrable likelihoods:')
        print_likelihoods(vulnerable)
        print('\nVulnerabilities likelihoods:')
        print_likelihoods(vulnerabilities)

        # Return the prediction result as JSON
        response = {
            "vulnerable": int(np.argmax(vulnerable)),
            "likelihoods1": vulnerable[0].tolist(),
            'vulnerabilities': int(np.argmax(vulnerabilities)),
            'likelihoods2': vulnerabilities[0].tolist(),
            'version': version,
            'stats': stats
        }

        return jsonify(response)

    except Exception as e:
        # Log the error and return a 500 error response
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)