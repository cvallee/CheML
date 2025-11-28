def add_descriptors(df, column='RDKit_Mol', add2D=True, add3D=False):
    '''Read SMILES of molecules, generates corresponding 2D Descriptors (add2D = True) and/or 3D Descriptors,
    and add them into the data frame (inplace = True) or to a new data frame (inplace = False)'''
    
    if bool(add2D) is False and bool(add3D) is False:
        return print('ERROR! Need either add2D=True or add3D=True')
    
    import pandas as pd
    import rdkit
    from rdkit import Chem
    
    descriptors = {}
    # looping over the molecules to generate the fingerprints and append to a list of fingerprints for each molecule
    for mol in df[column]:
        if add2D:
            from rdkit.Chem import Descriptors
            d = Descriptors.CalcMolDescriptors(mol)
            for key, value in d.items():
                if key in descriptors:
                    descriptors[key].append(value)
                else:
                    descriptors[key] = [value]
        if add3D:
            from rdkit.Chem import AllChem
            from rdkit.Chem import Descriptors3D
            mol3D = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol3D,randomSeed=0xf00d)
            d = Descriptors3D.CalcMolDescriptors3D(mol3D)
            for key, value in d.items():
                if key in descriptors:
                    descriptors[key].append(value)
                else:
                    descriptors[key] = [value]

    # Generate descriptors data frame
    descriptors_df = pd.DataFrame.from_dict(descriptors)
    new_df = pd.concat([df,descriptors_df],axis=1)
    return new_df
