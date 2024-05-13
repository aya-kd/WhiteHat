from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS class from flask_cors
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import json
import re
from solcx import compile_standard, install_solc

app = Flask(__name__)
CORS(app)  # Wrap your Flask app with CORS

# Load the trained model 
model_path = os.path.join(os.path.dirname(__file__), 'model1.h5')
model = tf.keras.models.load_model(model_path)

tokenizer = Tokenizer()


###################################### Compilation ##################################

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

####################################### Classification ################################

# Tokenization and padding function
def preprocess_opcode(opcode):
    # Tokenization
    tokenizer.fit_on_texts([opcode])
    sequence = tokenizer.texts_to_sequences([opcode])[0]

    # Sequence padding
    max_length = max([len(sequence)])
    sequence = pad_sequences([sequence], maxlen=max_length)[0]
    return sequence

def predict_opcode_class(opcode):
    # Preprocess opcode
    sequence = preprocess_opcode(opcode)
    
    # Perform prediction
    classification = model.predict(np.array([sequence]))
    return classification

def print_likelihoods(classification):
    total = np.sum(classification)
    likelihoods = (classification / total) * 100
    for i, likelihood in enumerate(likelihoods[0]):
        print(f'Class {i}: {likelihood:.2f}%')


# Define a route for the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request 
    data = request.json
    
    # Get values
    code = data.get('code', '')
    filename = data.get('file', '')
    version = extract_version(data.get('version', ''))
    
    # Preprocess the input data (if necessary)
    opcodes = compile_contract(code, filename, version)
    print(opcodes)
    
    # Get the predicted class
    predicted_class = predict_opcode_class(opcodes)
    
    # Print the predicted class
    print('\nPredicted class:', np.argmax(predicted_class))
    
    # Print the likelihoods
    print('\nLikelihoods:')
    print_likelihoods(predicted_class)


if __name__ == '__main__':
    app.run(debug=True)