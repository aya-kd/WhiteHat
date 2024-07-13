import React, { createContext, useState } from 'react';

export const ImportContext = createContext();

export const ImportProvider = ({ children }) => {
  const [importedFiles, setImportedFiles] = useState([]);

  const setCodeOrFiles = (data) => {
    setImportedFiles(data);
  };

  const contextValue = { importedFiles, setCodeOrFiles };

  return (
    <ImportContext.Provider value={contextValue}>
      {children}
    </ImportContext.Provider>
  );
};
