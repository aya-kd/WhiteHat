import React, { useEffect, useState, useContext } from 'react';
import { toChecksumAddress } from 'ethereum-checksum-address';
import fetch from 'node-fetch';
import { ImportContext } from '../code/ImportContext';
import './import.css';

const Import = ({ onCodeFetch }) => {
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const { setCodeOrFiles } = useContext(ImportContext);
  //const [worker, setWorker] = useState(null);

  {/*useEffect(() => {
    if (window.Worker) {
      const myWorker = new Worker('./solcWorker.js');
      setWorker(myWorker);
    }
  }, []);*/}

  const handleInputChange = (event) => {
    setInputValue(event.target.value);
  };
  
  async function fetchContractMetadata(contractAddress) {
    const url = `https://api-sepolia.etherscan.io/api?module=contract&action=getsourcecode&address=${contractAddress}&apikey=VU7YAQVMHXX926MUCU7AE6C25RCFBPBIZF`;
    const response = await fetch(url);
    const data = await response.json();
    return data;
  }

  const handleImport = async () => {
    setIsLoading(true);
    setErrorMessage('');
    try{
      const metadata = await fetchContractMetadata(inputValue);
      if (metadata.result && metadata.result.length > 0){
        let sourceCodeString = metadata.result[0].SourceCode;
        sourceCodeString=  sourceCodeString.slice(1, -1);
        let sourceCodeObject= JSON.parse(sourceCodeString);
        if (sourceCodeObject.sources && typeof sourceCodeObject.sources === 'object') {
          const sourceFiles = Object.keys(sourceCodeObject.sources);
          sourceFiles.forEach(file => {
              let code = sourceCodeObject.sources[file].content;
              console.log('Source code for', file, ':');
              console.log(code);
              onCodeFetch(code);
          });
        }
      }
    } catch (error){
      setErrorMessage('Error importing:', error.message);
    }finally{
      setIsLoading(false);
    }
  }
    {/*
    try {
      
      if (metadata.result && metadata.result.length > 0) {
        console.log('Contract Metadata:', metadata.result[0]);
      } else {
        console.log('Contract metadata not found.');
      }
    } catch (error) {
      
    } finally {
      
    }
  };*/}
  
       {/* let sourceCodeString = metadata.result[0].SourceCode;
        sourceCodeString = sourceCodeString.slice(1, -1);
        let sourceCodeObject = JSON.parse(sourceCodeString);
        if (sourceCodeObject.sources && typeof sourceCodeObject.sources === 'object') {
          const sourceFiles = Object.keys(sourceCodeObject.sources);
          sourceFiles.forEach(async (file) => {
            let code = sourceCodeObject.sources[file].content;
            const input = {
              language: 'Solidity',
              sources: {
                file: {
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
            worker.postMessage(input);
          });
          worker.onmessage = function (e) {
            const compiledCode = JSON.parse(e.data);
            const { bytecode } = compiledCode.contracts['contract.sol']['<stdin>'];
            setCodeOrFiles({ code: bytecode, address: inputValue });
          };
        } else {
          console.log('Sources not found in the source code object.');
        }
      } else {
        console.log('Contract metadata not found.');
      }
    } catch (error) {
      setErrorMessage('Error importing:', error.message);
    } finally {
      setIsLoading(false);
    }*/}
  

  return (
    <div className='import'>
      <input className='import_input' type="text" placeholder="Enter GitHub repository or smart contract address" value={inputValue} onChange={handleInputChange} />
      <button className='import_button' onClick={handleImport} disabled={isLoading}>
        {isLoading ? '' : 'Import'}
      </button>
    </div>
  );
};

export default Import;
