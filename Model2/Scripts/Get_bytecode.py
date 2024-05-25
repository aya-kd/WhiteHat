import requests
import pandas as pd

# Replace with your Etherscan API key
api_key = 'VU7YAQVMHXX926MUCU7AE6C25RCFBPBIZF'

def get_contract_bytecode(contract_address):
    # Etherscan API URL
    url = f'https://api.etherscan.io/api?module=proxy&action=eth_getCode&address={contract_address}&tag=latest&apikey={api_key}'
    try:
        # Send a GET request to the Etherscan API
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()
            return data['result']
        else:
            print(f"Error: Failed to fetch bytecode for {contract_address}, status code {response.status_code}")
    except Exception as e:
        print(f'An error occurred: {str(e)}')
    return None

def main():
    # Read CSV file containing contract addresses
    df = pd.read_csv('contracts.csv')

    # List to hold bytecode for each address
    bytecode_list = []

    # Fetch bytecode for each contract address
    for index, row in df.iterrows():
        contract_address = row['address']
        print(f'Fetching bytecode for contract: {contract_address}')
        
        # Fetch bytecode for the contract address
        bytecode = get_contract_bytecode(contract_address)
        print(bytecode)
        bytecode_list.append(bytecode)

    # Add bytecode list as a new column in the dataframe
    df['bytecode'] = bytecode_list

    # Save the updated dataframe to a new CSV file
    df.to_csv('address_bytecode.csv', index=False)

    print('Done')

if __name__ == "__main__":
    main()
