{/*const Web3 = require('web3');*/}
const { toChecksumAddress } = require('ethereum-checksum-address');
const fetch = require('node-fetch');
const solc = require('solc');
const prettier = require('prettier');
// Initialize Web3 instance with an Ethereum node URL
{/*const web3 = new Web3('https://sepolia.infura.io/v3/b13028e884ff4f6fae748d8e9f7bbc4f');*/}
// Contract address and bytecode
const contractAddress = '0xe34e25a674fd600a4C31bf152116cf6f0e925D42';
const bytecode = '0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890';
// Function to fetch contract metadata
async function fetchContractMetadata(contractAddress) {
    const url = `https://api-sepolia.etherscan.io/api?module=contract&action=getsourcecode&address=${contractAddress}&apikey=VU7YAQVMHXX926MUCU7AE6C25RCFBPBIZF `;
    const response = await fetch(url);
    const data = await response.json();
    return data;
}
// Fetch contract metadata
fetchContractMetadata(toChecksumAddress(contractAddress))
    .then(metadata => {
        if (metadata.result && metadata.result.length > 0) {
            let sourceCodeString = metadata.result[0].SourceCode;
            sourceCodeString=  sourceCodeString.slice(1, -1);
            let sourceCodeObject= JSON.parse(sourceCodeString);
            if (sourceCodeObject.sources && typeof sourceCodeObject.sources === 'object') {
                const sourceFiles = Object.keys(sourceCodeObject.sources);
                sourceFiles.forEach(file => {
                    let code = sourceCodeObject.sources[file].content;
                    let formattedCode = prettier.format(code, { parser: "babel" });
                    console.log(formattedCode);
                    console.log('Source code for', file, ':');
                    console.log(code);
                    const input ={
                        language: 'Solidity',
                        sources: {
                            file : {
                                content: code,
                            },
                        },
                        settings: {
                            outputSelection: {
                                '*': {
                                    '*': ['*'],
                                },
                            },
                        },
                    };
                    const compiledCode = JSON.parse(solc.compile(JSON.stringify(input)));
                    console.log(solc.compile(JSON.stringify(input)));
                    console.log('compiledCode', compiledCode);
                    let contractName = Object.keys(compiledCode.contracts[file])[0];
                    // Extract the bytecode and opcodes
                    const bytecode = compiledCode.contracts[file][contractName].evm.bytecode.object;
                    const opcodes = compiledCode.contracts[file][contractName].evm.bytecode.opcodes;
                    // Print the opcodes
                    console.log('Opcodes:', opcodes);
                });
            } else {
                console.log('Sources not found in the source code object.');
            }
        } else {
            console.log('Contract metadata not found.');
        }
    })
    .catch(error => {
        console.error('Error fetching contract metadata:', error);
    });
    