import React, {useState} from 'react'
import {Chat, Code, Files, Navbar, Import} from '../../components';
import {ImportProvider} from '../../components/code/ImportContext';
import './scan.css';
const Scan = () => {
  const [fetchedCode, setFetchedCode] = useState('');

  const handleCodeFetch = (code) => {
    setFetchedCode(code);
  };

  return (
    <div className="vertical-layout">
      <Navbar />
      <div className="content-wrapper">
        <div className="left-section" >
          <ImportProvider>
            <Import onCodeFetch={handleCodeFetch}/>
            <Files />
          </ImportProvider>
        </div>
        <div className="code-section" >
            <Code code={fetchedCode} />
        </div>
        <div className="right-section" >
          <Chat />
        </div>
      </div>
    </div>
  )
}
export default Scan