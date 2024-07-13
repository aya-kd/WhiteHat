# Scripts:  

This folder contains the different Python scripts used for data collection and preprocessing. It is divided into two sub-folders:

## Addirional Scripts:  
1. `Commenting Pragma lines.py`: comments the "solidity pragma x.y.z;" line (removing the version was intented to compile all files at once using a single version but it didn't work, it only complicated things).  
2. `Count Lines.py`: counts the rows in a CSV file to check if it contains the right number of opcodes.
3. `Files encoding.py`: used to detect the solidity files encoding (used to solve an error caused by the compiler's inability to read certain files)  


## Main Scripts:  
The scripts were used in the following order: 
1. `Pragma Classifier`: used to seperate Solidity files with different Solidty versions into separate folders accordingly
2. `Compiler.py`: used to compile all of the Solidty files in the dataset and write the result (**opcodes** and **vulnerability**) in a CSV file.
3. `Merge and Clean.py`: used to merge several CSV files into a single  CSV file and remove empty lines from it (cleaning the data)

## Others
1. `API-Compiler.py`: a preliminary version of the compiler used later in the API (used to compile the code retrieved from the front). 