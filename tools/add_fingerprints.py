def add_fingerprints(df, column='RDKit_Mol', fp_type='RDKit', nBits = 512, radius = 2):
    '''Read SMILES of molecules, generates corresponding fingerprints (RDKit - default -, MACCS, Avalon, Atom-Pairs, Topological-Torsions or Morgan-Circular)
    and add them into the data frame (inplace = True) or to a new data frame (inplace = False)'''
    
    import pandas as pd
    import numpy as np
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    
    fp_list = []
    # looping over the molecules to generate the fingerprints and append to a list of fingerprints for each molecule
    if fp_type.lower() == 'atom-pairs':
        fpgen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=nBits)
            
    elif fp_type.lower() == 'topological-torsions':
        fpgen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=nBits)
            
    elif fp_type.lower() == 'morgan':
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius,fpSize=nBits)
            
    else:
        if fp_type.lower() != 'rdkit':
            print('WARNING! Fingerprint type incorrect or undefined! Using defaults RDKit fingerprints...')
        fpgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=nBits)
    
    for mol in df[column]:
            fp = np.array(fpgen.GetFingerprint(mol))
            fp_list.append(fp)
    
    # Generating a list of column names
    fp_str = [f'Fingerprint_{i}' for i in range(len(fp_list[0]))]
    
    # Generate fingerprint data frame
    fp_df = pd.DataFrame(fp_list,columns=fp_str)
    new_df = pd.concat([df,fp_df],axis=1)
    return new_df
    