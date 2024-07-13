def count_stats(file_path):
    total_lines = 0
    empty_lines = 0
    code_lines = 0
    comment_lines = 0
    public_functions = 0
    private_functions = 0
    internal_functions = 0
    external_functions = 0
    receives_funds = False
    transfers_eth = False
    uses_assembly = False
    destroyables = False
    delegatecall = False
    signature_or_hashing = False
    new_create_create2 = False
    unchecked = False
    view_functions = 0
    pure_functions = 0
    payable_functions = 0
    nonpayable_functions = 0

    with open(file_path, 'r') as f:
        in_multiline_comment = False
        for line in f:
            total_lines += 1
            line = line.strip()

            # Empty lines
            if not line:
                empty_lines += 1
                continue

            # Comments
            if line.startswith('//'):
                comment_lines += 1
                continue
            if line.startswith('/*'):
                comment_lines += 1
                in_multiline_comment = True
                continue
            if in_multiline_comment:
                comment_lines += 1
                if line.endswith('*/'):
                    in_multiline_comment = False
                continue

            # Code lines
            code_lines += 1

            # Functions
            if 'function' in line:
                if 'public' in line:
                    public_functions += 1
                elif 'private' in line:
                    private_functions += 1
                elif 'internal' in line:
                    internal_functions += 1
                elif 'external' in line:
                    external_functions += 1

                if 'view' in line:
                    view_functions += 1
                if 'pure' in line:
                    pure_functions += 1
                if 'payable' in line:
                    payable_functions += 1
                if 'nonpayable' in line:
                    nonpayable_functions += 1

            # Contract capabilities
            if 'receive' in line or 'payable' in line:
                receives_funds = True
            if 'transfer(' in line or 'send(' in line or '.call{value:' in line:
                transfers_eth = True
            if 'assembly' in line:
                uses_assembly = True
            if 'selfdestruct' in line:
                destroyables = True
            if 'delegatecall' in line:
                delegatecall = True
            if 'ecrecover' in line or 'keccak256' in line:
                signature_or_hashing = True
            if 'new ' in line or 'Create(' in line or 'Create2(' in line:
                new_create_create2 = True
            if 'unchecked' in line:
                unchecked = True

    return {
        'total_lines': total_lines,
        'empty_lines': empty_lines,
        'code_lines': code_lines,
        'comment_lines': comment_lines,
        'public_functions': public_functions,
        'private_functions': private_functions,
        'internal_functions': internal_functions,
        'external_functions': external_functions,
        'view_functions': view_functions,
        'pure_functions': pure_functions,
        'payable_functions': payable_functions,
        'nonpayable_functions': nonpayable_functions,
        'receives_funds': receives_funds,
        'transfers_eth': transfers_eth,
        'uses_assembly': uses_assembly,
        'destroyables': destroyables,
        'delegatecall': delegatecall,
        'signature_or_hashing': signature_or_hashing,
        'new_create_create2': new_create_create2,
        'unchecked': unchecked
    }

def main():
    file_path = 'contracts/Test.sol'  # Update with your Solidity file path
    stats = count_stats(file_path)
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


if __name__ == "__main__":
    main()
