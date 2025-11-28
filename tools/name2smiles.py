def name2smiles(mol_id, isomeric=False):
    '''Convert molecule names to canonical or isomeric SMILES by serching into the PubChem database''' 
    import pubchempy as pcp
    import numpy as np
    
    try:
        compound = pcp.get_compounds(mol_id, 'name')
        if isomeric:
            # Get the isomeric SMILES
            smiles = compound[0].smiles
        else:
            # Get the canonical SMILES
            smiles = compound[0].connectivity_smiles
        return smiles
    except:
        if mol_id[-3:] == 'ate':
            try:
                compound = pcp.get_compounds(mol_id[:-3]+'ic acid', 'name')
                if isomeric:
                    # Get the isomeric SMILES
                    smiles = compound[0].smiles
                else:
                    # Get the canonical SMILES
                    smiles = compound[0].connectivity_smiles
                return smiles
            except:
                return np.nan
        else:
            return np.nan
