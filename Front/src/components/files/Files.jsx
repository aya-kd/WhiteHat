import React, { useContext } from 'react';
import { ImportContext } from '../code/ImportContext';
import './files.css';
const Files = () => {
  const { importedFiles } = useContext(ImportContext);

  return (
    <div className='files'>
      files
    </div>
  );
};

export default Files;