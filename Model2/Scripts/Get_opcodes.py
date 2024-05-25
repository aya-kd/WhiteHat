##################################################### Get opcodes #######################################################

import csv
import binascii
from pyevmasm import disassemble_hex

def bytecode_to_opcodes(bytecode):
    # Disassemble the hex bytecode into instructions
    instructions = disassemble_hex(bytecode)
    # Join the instructions into a single string, correctly formatted
    opcodes = ''.join(str(instr).replace('\n', ' ') for instr in instructions)
    return opcodes

def process_csv(input_file, output_file):
    with open(input_file, mode='r', newline='') as infile, open(output_file, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ['opcodes']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in reader:
            bytecode = row['bytecode']
            row['opcodes'] = bytecode_to_opcodes(bytecode)
            writer.writerow(row)

# Replace 'address_bytecode.csv' with your input CSV file and 'address_bytecode_opcodes.csv' with the desired output file
input_csv_file = 'address_bytecode.csv'
output_csv_file = 'address_bytecode_opcodes.csv'
process_csv(input_csv_file, output_csv_file)
