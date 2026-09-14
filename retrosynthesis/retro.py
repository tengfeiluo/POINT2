import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from rdkit import Chem
import matplotlib.pyplot as plt
import os
import re
from rdkit.Chem import AllChem, Draw, rdMolDescriptors, DataStructs, rdmolops
import cv2
import numpy as np
import argparse
import json
from SA_Score.sascorer import calculateScore
from scipy.stats import hmean
from tqdm import tqdm

font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
plt.rcParams['figure.dpi'] = 300

def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Retro-synthesis planning for polymers')
    parser.add_argument('--dataset', type=str, default="train", help='which dataset to use')
    args = parser.parse_args()
    return args



def calculate_monomer_syn_score(monomer_smiles):#pubchem_fps, pubchem_smiles
    """
    Calculate synthesizability scores for a monomer.

    Parameters:
    - monomer_smiles (str): SMILES string of the monomer.
    # - pubchem_fps (list): Precomputed fingerprints of PubChem molecules.
    # - pubchem_smiles (list): List of PubChem SMILES corresponding to the fingerprints.

    Returns:
    - dict: A dictionary containing SAscore and max Tanimoto similarity.
    """
    # Convert monomer SMILES to a molecule
    monomer_mol = Chem.MolFromSmiles(monomer_smiles)
    if not monomer_mol:
        raise ValueError("Invalid SMILES string")

    # # Calculate SAscore
    sa_score = calculateScore(monomer_mol)

    # # Compute the fingerprint for the monomer
    # monomer_fp = AllChem.GetMorganFingerprintAsBitVect(monomer_mol, radius=2, nBits=2048)

    # # Calculate Tanimoto similarity with PubChem molecules
    # tanimoto_scores = [DataStructs.TanimotoSimilarity(monomer_fp, fp) for fp in pubchem_fps]
    # max_tanimoto_similarity = max(tanimoto_scores) if tanimoto_scores else 0

    return sa_score
    # return {
    #     "SMILES": monomer_smiles,
    #     "SAscore": sa_score,
    #     "MaxTanimotoSimilarity": max_tanimoto_similarity,
    #     "ClosestPubChemMatch": pubchem_smiles[np.argmax(tanimoto_scores)] if tanimoto_scores else None
    # }


def concat_psmiles(smiles_1, smiles_2):
    mol_1 = Chem.MolFromSmiles(smiles_1)
    mol_2 = Chem.MolFromSmiles(smiles_2)
    mol_combo = Chem.CombineMols(mol_1, mol_2)
    smiles_combo = Chem.MolToSmiles(mol_combo)
    partA = re.search(r'(.*\.)\*(\/*)([a-zA-Z][0-9]*)(.*)',smiles_combo).group(1)
    partB = re.search(r'(.*\.)\*(\/*)([a-zA-Z][0-9]*)(.*)',smiles_combo).group(2)
    partC = re.search(r'(.*\.)\*(\/*)([a-zA-Z][0-9]*)(.*)',smiles_combo).group(3)
    partD = re.search(r'(.*\.)\*(\/*)([a-zA-Z][0-9]*)(.*)',smiles_combo).group(4)
    #For linear polymers
    if smiles_1.count('*') == 2:
        combE = partA+partC+partB+'%90'+partD
        smiles_final = combE.replace('*','%90',2).replace('%90',"*",1).replace('(%90)','%90').replace('(/%90)','%90')
    #For ladder polymers
    if smiles_1.count('*') == 4:
        combE = partA+partC+partB+'%80'+partD
        smiles_final = combE.replace('*','%90',4).replace('*','%90',1).replace('%90','%80',3).replace('%80','*',2).replace('(%90)','%90').replace('(/%90)','%90').replace('(%80)','%80').replace('(/%80)','%80')

    mol_final = Chem.MolFromSmiles(smiles_final)
    smiles_final = Chem.MolToSmiles(mol_final)
    return smiles_final

def concat_n_psmiles(psmiles,n):
    psmiles_1 = psmiles
    psmiles_2 = psmiles
    for i in range(n-1):
        psmiles_2 = concat_psmiles(psmiles_1,psmiles_2)
    return psmiles_2



def is_meaningful(mol, min_heavy_atoms=2):
    """
    Check if a molecule is meaningful (exclude small fragments like [OH]).
    
    Parameters:
    - mol: RDKit Mol object
    - min_heavy_atoms: Minimum number of heavy (non-hydrogen) atoms to consider the molecule meaningful.

    Returns:
    - True if the molecule is meaningful, False otherwise.
    """
    if mol is None:
        return False

    # Count heavy atoms (non-H atoms)
    heavy_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
    
    # Exclude if fewer heavy atoms than threshold
    return heavy_atoms >= min_heavy_atoms

def modify_terminal_atoms(smiles, termination_rule = "condensation-Ester"):
    """
    Modify the `*` atoms in a SMILES string:
    Rules:
    - "condensation-Ester_OH":
        - Replace `*` with `OH` if connected to a `C`.
        - Replace `*` with `H` if connected to an `O`.
    - "condensation-Ester_Cl":
        - Replace `*` with `OH` if connected to a `C`.
        - Replace `*` with `H` if connected to an `O`.
    - "condensation-Imide":
        - Replace `*` with `NH2` if connected to a `C`.
        - Replace `N-*` with `O` (replace the nitrogen atom with a oxygen).
    - No change otherwise.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    
    if termination_rule == "condensation-Ester_OH":
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 1:  # Ensure the `*` has exactly one neighbor
                    neighbor = neighbors[0]
                    atom_index = atom.GetIdx()

                    if neighbor.GetSymbol() == "C":
                        # Replace `*` with `OH` for this specific atom
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("O"))  # Replace `*` with Oxygen
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
                        
                    elif neighbor.GetSymbol() == "O":
                        # Replace `*` with `H` for this specific atom
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("H"))  # Replace `*` with Hydrogen
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
    
    elif termination_rule == "condensation-Ester_O_CH3":
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 1:  # Ensure the `*` has exactly one neighbor
                    neighbor = neighbors[0]
                    atom_index = atom.GetIdx()

                    if neighbor.GetSymbol() == "C":
                        # Replace `*` with `OH` for this specific atom
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("O"))  # Replace `*` with Oxygen
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
                        
                    elif neighbor.GetSymbol() == "O":
                        # Replace `*` with `H` for this specific atom
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("C"))  # Replace `*` with CH3
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol

    elif termination_rule == "condensation-Ester_O_CCH3":
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 1:  # Ensure the `*` has exactly one neighbor
                    neighbor = neighbors[0]
                    atom_index = atom.GetIdx()

                    if neighbor.GetSymbol() == "C":
                        # Replace `*` with `OH` for this specific atom
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("O"))  # Replace `*` with Oxygen
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
                        
                    elif neighbor.GetSymbol() == "O":
                        # Replace `*` with `H` for this specific atom
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("Be"))  # Replace `*` with Oxygen
                        smarts_pattern = "[Be]" 
                        replacement_smiles = "CC" 
                        replace_mol = Chem.MolFromSmiles(replacement_smiles)
                        # Perform substructure replacement
                        edit_mol = Chem.ReplaceSubstructs(edit_mol, Chem.MolFromSmarts(smarts_pattern), replace_mol, replaceAll=False)[0]
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
    elif termination_rule == "condensation-Ester_O_CCO":
        # for atom in mol.GetAtoms():
        # Iterate over atoms in descending order of indices
        for atom in sorted(mol.GetAtoms(), key=lambda a: a.GetIdx(), reverse=True):
            if atom.GetSymbol() == "*":
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 1:  # Ensure the `*` has exactly one neighbor
                    neighbor = neighbors[0]
                    atom_index = atom.GetIdx()

                    if neighbor.GetSymbol() == "C":
                        # Replace `*` with `OH` for this specific atom
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("O"))  # Replace `*` with Oxygen
                        print("XXXX",Chem.MolToSmiles(edit_mol))
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
                        
                    elif neighbor.GetSymbol() == "O":
                        # Replace `*` with `H` for this specific atom
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("Be"))  # Replace `*` with Be
                        smarts_pattern = "[#8][Be]" 
                        replacement_smiles = "[#8][#6](=[#8])[CH3]" 
                        replace_mol = Chem.MolFromSmiles(replacement_smiles)
                        # Perform substructure replacement
                        edit_mol = Chem.ReplaceSubstructs(edit_mol, Chem.MolFromSmarts(smarts_pattern), replace_mol, replaceAll=False)[0]
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
                        print("XXXXXXXX",Chem.MolToSmiles(mol))

    elif termination_rule == "condensation-Ester_Cl":
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 1:  # Ensure the `*` has exactly one neighbor
                    neighbor = neighbors[0]
                    atom_index = atom.GetIdx()

                    if neighbor.GetSymbol() == "C":
                        # Replace `C-*` with `Cl` for this specific atom
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("Cl"))  # Replace `*` with Oxygen
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
                        
                    elif neighbor.GetSymbol() == "O":
                        # Replace `O*` with `OH` for this specific atom
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("H")) 
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
    elif termination_rule == "condensation-Ester_Cl_2":
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 1:  # Ensure the `*` has exactly one neighbor
                    neighbor = neighbors[0]
                    atom_index = atom.GetIdx()

                    if neighbor.GetSymbol() == "C":
                        # Replace `C-*` with `OH` for this specific atom
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("O"))  # Replace `*` with Oxygen
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
                        
                    elif neighbor.GetSymbol() == "O":
                        # Replace `O*` with `Cl` for this specific atom
                        edit_mol = Chem.RWMol(mol)
                        # edit_mol.ReplaceAtom(atom_index, Chem.Atom("H")) 
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("Be"))  # Replace `*` with Be (any rare atom)
                        smarts_pattern = "[#8]([Be])" 
                        replacement_smiles = "Cl" 
                        replace_mol = Chem.MolFromSmiles(replacement_smiles)
                        # # Perform substructure replacement
                        edit_mol = Chem.ReplaceSubstructs(edit_mol, Chem.MolFromSmarts(smarts_pattern), replace_mol, replaceAll=False)[0]
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
    elif termination_rule == "condensation-Ester_Cl_3":
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 1:  # Ensure the `*` has exactly one neighbor
                    neighbor = neighbors[0]
                    atom_index = atom.GetIdx()

                    if neighbor.GetSymbol() == "C":
                        # Replace `C-*` with `Cl` for this specific atom
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("O"))  # Replace `*` with Oxygen
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
                        
                    elif neighbor.GetSymbol() == "O":
                        # Replace `O*` with `OH` for this specific atom
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("H")) 
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol

    elif termination_rule == "condensation-Imide":
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 1:  # Ensure the `*` has exactly one neighbor
                    neighbor = neighbors[0]
                    atom_index = atom.GetIdx()
                    
                    if neighbor.GetSymbol() == "C":
                        # Replace `*` with NH2
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("N"))
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
                        
                    elif neighbor.GetSymbol() == "N":
                        # Replace N-* with =O
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("Be"))  # Replace `*` with Be (any rare atom)

                        smarts_pattern = "[#7]([Be])" 
                        replacement_smiles = "[#8]" 
                        replace_mol = Chem.MolFromSmiles(replacement_smiles)
                        # Perform substructure replacement
                        edit_mol = Chem.ReplaceSubstructs(edit_mol, Chem.MolFromSmarts(smarts_pattern), replace_mol, replaceAll=False)[0]
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol

    elif termination_rule == "condensation-Amide_Cl":
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 1:  # Ensure the `*` has exactly one neighbor
                    neighbor = neighbors[0]
                    atom_index = atom.GetIdx()
                    
                    if neighbor.GetSymbol() == "C":
                        ## Replace `*` with NH2
                        # Replace `*` with Cl
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("Cl"))
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
                        
                    elif neighbor.GetSymbol() == "N":
                        ## Replace N-* with Cl
                        # Replace N-* with -NH
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("H"))
                        # edit_mol.ReplaceAtom(atom_index, Chem.Atom("Be"))  # Replace `*` with Be (any rare atom)
                        # smarts_pattern = "[#7]([Be])" 
                        # replacement_smiles = "Cl" 
                        # replace_mol = Chem.MolFromSmiles(replacement_smiles)
                        # # Perform substructure replacement
                        # edit_mol = Chem.ReplaceSubstructs(edit_mol, Chem.MolFromSmarts(smarts_pattern), replace_mol, replaceAll=False)[0]
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
    elif termination_rule == "condensation-Amide_S_Cl":
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 1:  # Ensure the `*` has exactly one neighbor
                    neighbor = neighbors[0]
                    atom_index = atom.GetIdx()
                    
                    if neighbor.GetSymbol() == "S":
                        # Replace `*` with Cl
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("Cl"))
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
                        
                    elif neighbor.GetSymbol() == "N":
                        ## Replace N-* with Cl
                        # Replace N-* with -NH
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("H"))
                        # edit_mol.ReplaceAtom(atom_index, Chem.Atom("Be"))  # Replace `*` with Be (any rare atom)
                        # smarts_pattern = "[#7]([Be])" 
                        # replacement_smiles = "Cl" 
                        # replace_mol = Chem.MolFromSmiles(replacement_smiles)
                        # # Perform substructure replacement
                        # edit_mol = Chem.ReplaceSubstructs(edit_mol, Chem.MolFromSmarts(smarts_pattern), replace_mol, replaceAll=False)[0]
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
    

    elif termination_rule == "condensation-Amide_Cl_2":
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 1:  # Ensure the `*` has exactly one neighbor
                    neighbor = neighbors[0]
                    atom_index = atom.GetIdx()
                    
                    if neighbor.GetSymbol() == "C":
                        # Replace `*` with NH2
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("N"))
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
                        
                    elif neighbor.GetSymbol() == "N":
                        # Replace N-* with Cl
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("Be"))  # Replace `*` with Be (any rare atom)
                        smarts_pattern = "[#7]([Be])" 
                        replacement_smiles = "Cl" 
                        replace_mol = Chem.MolFromSmiles(replacement_smiles)
                        # Perform substructure replacement
                        edit_mol = Chem.ReplaceSubstructs(edit_mol, Chem.MolFromSmarts(smarts_pattern), replace_mol, replaceAll=False)[0]
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol

    elif termination_rule == "condensation-Amide_Cl_O_Benz":
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 1:  # Ensure the `*` has exactly one neighbor
                    neighbor = neighbors[0]
                    atom_index = atom.GetIdx()
                    
                    if neighbor.GetSymbol() == "C":
                        # Replace `*` with NH2
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("N"))
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
                        
                    elif neighbor.GetSymbol() == "N":
                        # Replace N-* with -O-BenzRing
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("Be"))  # Replace `*` with Be (any rare atom)
                        smarts_pattern = "[#7]([Be])" 
                        replacement_smiles = "Oc1ccccc1" 
                        replace_mol = Chem.MolFromSmiles(replacement_smiles)
                        # Perform substructure replacement
                        edit_mol = Chem.ReplaceSubstructs(edit_mol, Chem.MolFromSmarts(smarts_pattern), replace_mol, replaceAll=False)[0]
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
    elif termination_rule == "condensation-S_Cl":
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 1:  # Ensure the `*` has exactly one neighbor
                    neighbor = neighbors[0]
                    atom_index = atom.GetIdx()
                    
                    if neighbor.GetSymbol() == "C":
                        # Replace `*` with H
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("H"))
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
                        
                    elif neighbor.GetSymbol() == "S":
                        # Replace S-* with S-Cl
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("Cl")) 
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
    elif termination_rule == "condensation-S_C=O":
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 1:  # Ensure the `*` has exactly one neighbor
                    neighbor = neighbors[0]
                    atom_index = atom.GetIdx()
                    
                    if neighbor.GetSymbol() == "C":
                        # Replace `*` with H
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("S"))
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
                        
                    elif neighbor.GetSymbol() == "S":
                        # Replace S-* with S-Cl
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("Be")) 
                        smarts_pattern = "[S][Be]" 
                        replacement_smiles = "[Cl]" 
                        replace_mol = Chem.MolFromSmiles(replacement_smiles)
                        # Perform substructure replacement
                        edit_mol = Chem.ReplaceSubstructs(edit_mol, Chem.MolFromSmarts(smarts_pattern), replace_mol, replaceAll=False)[0]
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
    elif termination_rule == "condensation-S_C=O_2":
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 1:  # Ensure the `*` has exactly one neighbor
                    neighbor = neighbors[0]
                    atom_index = atom.GetIdx()
                    
                    if neighbor.GetSymbol() == "C":
                        # Replace `*` with H
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("S"))
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
                        
                    elif neighbor.GetSymbol() == "S":
                        # Replace S-* with S-Cl
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("Be")) 
                        smarts_pattern = "[S][Be]" 
                        replacement_smiles = "O" 
                        replace_mol = Chem.MolFromSmiles(replacement_smiles)
                        # Perform substructure replacement
                        edit_mol = Chem.ReplaceSubstructs(edit_mol, Chem.MolFromSmarts(smarts_pattern), replace_mol, replaceAll=False)[0]
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
    elif termination_rule == "condensation-c_O_Br":
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 1:  # Ensure the `*` has exactly one neighbor
                    neighbor = neighbors[0]
                    atom_index = atom.GetIdx()
                    
                    if neighbor.GetSymbol() == "C":
                        # Replace `*` with Br
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("Br"))
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
                        
                    elif neighbor.GetSymbol() == "O":
                        # Replace O-* with -OH
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("H")) 
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol

    elif termination_rule == "condensation-P_O":
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 1:  # Ensure the `*` has exactly one neighbor
                    neighbor = neighbors[0]
                    atom_index = atom.GetIdx()
                    
                    if neighbor.GetSymbol() == "C":
                        # Replace `*` with OH
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("O"))
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
                        print("## ",Chem.MolToSmiles(mol))
                        
                    elif neighbor.GetSymbol() == "O":
                        # Replace O-* with -Cl
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("Be"))  # Replace `*` with Be (any rare atom)
                        smarts_pattern = "[O][Be]" 
                        replacement_smiles = "[Cl]" 
                        replace_mol = Chem.MolFromSmiles(replacement_smiles)
                        # Perform substructure replacement
                        edit_mol = Chem.ReplaceSubstructs(edit_mol, Chem.MolFromSmarts(smarts_pattern), replace_mol, replaceAll=False)[0]
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
                        print("## ",Chem.MolToSmiles(mol))
    elif termination_rule == "condensation-P_O_2":
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 1:  # Ensure the `*` has exactly one neighbor
                    neighbor = neighbors[0]
                    atom_index = atom.GetIdx()
                    
                    if neighbor.GetSymbol() == "P":
                        # Replace `*` with OH
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("Cl"))
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
                        
                    elif neighbor.GetSymbol() == "O":
                        # Replace O-* with -Cl
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("H"))  # Replace `*` with Be (any rare atom)
                        # smarts_pattern = "[O][Be]" 
                        # replacement_smiles = "[Cl]" 
                        # replace_mol = Chem.MolFromSmiles(replacement_smiles)
                        # # Perform substructure replacement
                        # edit_mol = Chem.ReplaceSubstructs(edit_mol, Chem.MolFromSmarts(smarts_pattern), replace_mol, replaceAll=False)[0]
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol

    elif termination_rule == "condensation-c_Cl_O":
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 1:  # Ensure the `*` has exactly one neighbor
                    neighbor = neighbors[0]
                    atom_index = atom.GetIdx()
                    
                    if neighbor.GetSymbol() == "C":
                        # Replace `*` with Cl
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("Cl"))
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
                        
                    elif neighbor.GetSymbol() == "O":
                        # Replace O-* with -OH
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("H")) 
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol

    elif termination_rule == "condensation-N_Cl_c":
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 1:  # Ensure the `*` has exactly one neighbor
                    neighbor = neighbors[0]
                    atom_index = atom.GetIdx()
                    
                    if neighbor.GetSymbol() == "N":
                        # Replace `*` with Cl
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("Cl"))
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
                        
                    elif neighbor.GetSymbol() == "C":
                        # Replace c-* with c-H
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("H")) 
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol

    elif termination_rule == "condensation-COOH_OH":
        # SMARTS pattern to check if * is connected to -C=O
        pattern_c_equals_o = Chem.MolFromSmarts("[#0][C](=O)")
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 1:  # Ensure the `*` has exactly one neighbor
                    neighbor = neighbors[0]
                    atom_index = atom.GetIdx()
                    
                    if neighbor.GetSymbol() == "O":
                        # Replace `*` with H
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("H"))
                        Chem.SanitizeMol(edit_mol)
                        # mol = Chem.RemoveHs(edit_mol)  # Remove explicit hydrogens
                        mol = edit_mol
                        
                    elif neighbor.GetSymbol() == "C" and mol.HasSubstructMatch(pattern_c_equals_o):
                        # Replace *-C=O with -COOH
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("O")) 
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol

    elif termination_rule == "condensation-Amide_OH":
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 1:  # Ensure the `*` has exactly one neighbor
                    neighbor = neighbors[0]
                    atom_index = atom.GetIdx()
                    
                    if neighbor.GetSymbol() == "C":
                        ## Replace `*` with NH2
                        # Replace `*` with OH
                        edit_mol = Chem.RWMol(mol)
                        # edit_mol.ReplaceAtom(atom_index, Chem.Atom("N"))
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("O"))
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
                        
                    elif neighbor.GetSymbol() == "N":
                        ## Replace N-* with Cl
                        # Replace N-* with N-H
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("H"))  # Replace `*` with Be (any rare atom)
                        # smarts_pattern = "[#7]([Be])" 
                        # replacement_smiles = "O" 
                        # replace_mol = Chem.MolFromSmiles(replacement_smiles)
                        # # Perform substructure replacement
                        # edit_mol = Chem.ReplaceSubstructs(edit_mol, Chem.MolFromSmarts(smarts_pattern), replace_mol, replaceAll=False)[0]
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol

    elif termination_rule == "condensation-Amide_OH_2":
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 1:  # Ensure the `*` has exactly one neighbor
                    neighbor = neighbors[0]
                    atom_index = atom.GetIdx()
                    
                    if neighbor.GetSymbol() == "C":
                        ## Replace `*` with NH2
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("N"))
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
                        
                    elif neighbor.GetSymbol() == "N":
                        ## Replace N-* with -OH
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("Be"))  # Replace `*` with Be (any rare atom)
                        smarts_pattern = "[#7]([Be])" 
                        replacement_smiles = "O" 
                        replace_mol = Chem.MolFromSmiles(replacement_smiles)
                        # Perform substructure replacement
                        edit_mol = Chem.ReplaceSubstructs(edit_mol, Chem.MolFromSmarts(smarts_pattern), replace_mol, replaceAll=False)[0]
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
    
    elif termination_rule == "condensation-c_O_c":
        for atom in sorted(mol.GetAtoms(), key=lambda a: a.GetIdx(), reverse=True):
        # for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 1:  # Ensure the `*` has exactly one neighbor
                    neighbor = neighbors[0]
                    atom_index = atom.GetIdx()
                    
                    if neighbor.GetSymbol() == "C":
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("F"))
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
                        
                    elif neighbor.GetSymbol() == "O":
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("Be"))  # Replace `*` with Be (any rare atom)
                        smarts_pattern = "[#8]([Be])" 
                        replacement_smiles = "O" 
                        replace_mol = Chem.MolFromSmiles(replacement_smiles)
                        # Perform substructure replacement
                        edit_mol = Chem.ReplaceSubstructs(edit_mol, Chem.MolFromSmarts(smarts_pattern), replace_mol, replaceAll=False)[0]
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
    elif termination_rule == "condensation-c_O_c_2":
        for atom in sorted(mol.GetAtoms(), key=lambda a: a.GetIdx(), reverse=True):
        # for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 1:  # Ensure the `*` has exactly one neighbor
                    neighbor = neighbors[0]
                    atom_index = atom.GetIdx()
                    
                    if neighbor.GetSymbol() == "C":
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("O"))
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
                        
                    elif neighbor.GetSymbol() == "O":
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("Be"))  # Replace `*` with Be (any rare atom)
                        smarts_pattern = "[#8]([Be])" 
                        replacement_smiles = "F" 
                        replace_mol = Chem.MolFromSmiles(replacement_smiles)
                        # Perform substructure replacement
                        edit_mol = Chem.ReplaceSubstructs(edit_mol, Chem.MolFromSmarts(smarts_pattern), replace_mol, replaceAll=False)[0]
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
    elif termination_rule == "condensation-c_O_c_Cl":
        for atom in sorted(mol.GetAtoms(), key=lambda a: a.GetIdx(), reverse=True):
        # for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 1:  # Ensure the `*` has exactly one neighbor
                    neighbor = neighbors[0]
                    atom_index = atom.GetIdx()
                    
                    if neighbor.GetSymbol() == "C":
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("Cl"))
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
                        
                    elif neighbor.GetSymbol() == "O":
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("Be"))  # Replace `*` with Be (any rare atom)
                        smarts_pattern = "[#8]([Be])" 
                        replacement_smiles = "O" 
                        replace_mol = Chem.MolFromSmiles(replacement_smiles)
                        # Perform substructure replacement
                        edit_mol = Chem.ReplaceSubstructs(edit_mol, Chem.MolFromSmarts(smarts_pattern), replace_mol, replaceAll=False)[0]
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
    elif termination_rule == "condensation-c_O_c_Cl_2":
        for atom in sorted(mol.GetAtoms(), key=lambda a: a.GetIdx(), reverse=True):
        # for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 1:  # Ensure the `*` has exactly one neighbor
                    neighbor = neighbors[0]
                    atom_index = atom.GetIdx()
                    
                    if neighbor.GetSymbol() == "C":
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("O"))
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
                        
                    elif neighbor.GetSymbol() == "O":
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("Be"))  # Replace `*` with Be (any rare atom)
                        smarts_pattern = "[#8]([Be])" 
                        replacement_smiles = "Cl" 
                        replace_mol = Chem.MolFromSmiles(replacement_smiles)
                        # Perform substructure replacement
                        edit_mol = Chem.ReplaceSubstructs(edit_mol, Chem.MolFromSmarts(smarts_pattern), replace_mol, replaceAll=False)[0]
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
    
    elif termination_rule == "condensation-c_s_c":
        smarts_pattern = "[c][s][c]"  # Define the SMARTS pattern for [c][s][c]; only do while there is a subs match
        substructure = Chem.MolFromSmarts(smarts_pattern)
        if mol.HasSubstructMatch(substructure):
            smiles = smiles.replace("*","[Be]")
            mol = Chem.MolFromSmiles(smiles)
            edit_mol = Chem.RWMol(mol)
            smarts_pattern = "[Be]" 
            replacement_smiles = "[H]" 
            replace_mol = Chem.MolFromSmiles(replacement_smiles)
            # Perform substructure replacement
            edit_mol = Chem.ReplaceSubstructs(edit_mol, Chem.MolFromSmarts(smarts_pattern), replace_mol, replaceAll=True)[0]
            Chem.SanitizeMol(edit_mol)
            mol = edit_mol
    # elif termination_rule == "condensation-c_n_c":
    #     smarts_pattern = "[c][n][c]"  # Define the SMARTS pattern for [c][n][c]; only do while there is a subs match
    #     substructure = Chem.MolFromSmarts(smarts_pattern)
    #     if mol.HasSubstructMatch(substructure):
    #         smiles = smiles.replace("*","[Be]")
    #         mol = Chem.MolFromSmiles(smiles)
    #         edit_mol = Chem.RWMol(mol)
    #         smarts_pattern = "[Be]" 
    #         replacement_smiles = "[H]" 
    #         replace_mol = Chem.MolFromSmiles(replacement_smiles)
    #         # Perform substructure replacement
    #         edit_mol = Chem.ReplaceSubstructs(edit_mol, Chem.MolFromSmarts(smarts_pattern), replace_mol, replaceAll=True)[0]
    #         Chem.SanitizeMol(edit_mol)
    #         mol = edit_mol
    elif termination_rule == "condensation-c_n_c":
        smarts_pattern = "[c][n][c]"  # Define the SMARTS pattern for [c][n][c]; only do while there is a subs match
        substructure = Chem.MolFromSmarts(smarts_pattern)
        if mol.HasSubstructMatch(substructure):
            smiles = smiles.replace("*","[H]")
            mol = Chem.MolFromSmiles(smiles)
            # edit_mol = Chem.RWMol(mol)
            # smarts_pattern = "[Be]" 
            # replacement_smiles = "[H]" 
            # replace_mol = Chem.MolFromSmiles(replacement_smiles)
            # # Perform substructure replacement
            # edit_mol = Chem.ReplaceSubstructs(edit_mol, Chem.MolFromSmarts(smarts_pattern), replace_mol, replaceAll=True)[0]
            # Chem.SanitizeMol(edit_mol)
            # mol = edit_mol


    elif termination_rule == "condensation-_Si_":
        smiles = smiles.replace("*","[Be]")
        mol = Chem.MolFromSmiles(smiles)
        smarts_pattern = "[Be][Si][Be]"  # Define the SMARTS pattern for [c][n][c]; only do while there is a subs match
        substructure = Chem.MolFromSmarts(smarts_pattern)
        if mol.HasSubstructMatch(substructure):
            smiles = smiles.replace("[Be]","[Cl]")
            mol = Chem.MolFromSmiles(smiles)
            # edit_mol = Chem.RWMol(mol)
            # smarts_pattern = "[Be]" 
            # replacement_smiles = "[H]" 
            # replace_mol = Chem.MolFromSmiles(replacement_smiles)
            # # Perform substructure replacement
            # edit_mol = Chem.ReplaceSubstructs(edit_mol, Chem.MolFromSmarts(smarts_pattern), replace_mol, replaceAll=True)[0]
            # Chem.SanitizeMol(edit_mol)
            # mol = edit_mol
    elif termination_rule == "condensation-c_OH":
        smiles = smiles.replace("*","[Be]")
        mol = Chem.MolFromSmiles(smiles)
        smarts_pattern = "[Be][c]1[c][c][c]([c][c]1)[O][Be]"  # Define the SMARTS pattern; only do while there is a subs match
        substructure = Chem.MolFromSmarts(smarts_pattern)
        if mol.HasSubstructMatch(substructure):
            edit_mol = Chem.RWMol(mol)
            smarts_pattern = "[Be]" 
            replacement_smiles = "[H]" 
            replace_mol = Chem.MolFromSmiles(replacement_smiles)
            # Perform substructure replacement
            edit_mol = Chem.ReplaceSubstructs(edit_mol, Chem.MolFromSmarts(smarts_pattern), replace_mol, replaceAll=True)[0]
            Chem.SanitizeMol(edit_mol)
            mol = edit_mol
    elif termination_rule == "condensation-c_CCl":
        smiles = smiles.replace("*","[Be]")
        mol = Chem.MolFromSmiles(smiles)
        smarts_pattern = "[Be][C][c]1[c][c][c]([c][c]1)[Be]"  # Define the SMARTS pattern; only do while there is a subs match
        substructure = Chem.MolFromSmarts(smarts_pattern)
        if mol.HasSubstructMatch(substructure):
            edit_mol = Chem.RWMol(mol)
            
            smarts_pattern = "[C][Be]" 
            replacement_smiles = "C[Cl]" 
            replace_mol = Chem.MolFromSmiles(replacement_smiles)
            # Perform substructure replacement
            edit_mol = Chem.ReplaceSubstructs(edit_mol, Chem.MolFromSmarts(smarts_pattern), replace_mol, replaceAll=True)[0]

            smarts_pattern = "[Be]" 
            replacement_smiles = "[H]" 
            replace_mol = Chem.MolFromSmiles(replacement_smiles)
            # Perform substructure replacement
            edit_mol = Chem.ReplaceSubstructs(edit_mol, Chem.MolFromSmarts(smarts_pattern), replace_mol, replaceAll=True)[0]
            
            Chem.SanitizeMol(edit_mol)
            mol = edit_mol
            print(Chem.MolToSmiles(mol))

    elif termination_rule == "addition-CN":
        return concat_n_psmiles(smiles,n=2).replace("*","[Be]")

    elif termination_rule == "addition-Vinyl":
        # print(concat_n_psmiles(smiles,n=2))
        return concat_n_psmiles(smiles,n=2).replace("*","[Be]")
    elif termination_rule == "addition-Vinyl_3":
        return concat_n_psmiles(smiles,n=4).replace("*","[Be]")
    elif termination_rule == "addition-Vinyl_2":
        return smiles.replace("*","[Be]")
    
    # elif termination_rule == "addition-Vinyl_double":
    #     smiles_2 = concat_n_psmiles(smiles,n=2).replace("*","[Be]")
    #     print(smiles_2)
    #     mol_2 = Chem.MolFromSmiles(smiles_2)
    #     substructure = Chem.MolFromSmarts("CC=CC")
    #     matches = mol.GetSubstructMatches(substructure)
    #     print(matches)
    #     match = matches[0]
    #     atom_indices = list(match)
    #     matched_mol = Chem.PathToSubmol(mol_2, atom_indices)
    #     matched_smiles = Chem.MolToSmiles(matched_mol)
    #     print(matched_smiles)
    #     return matched_smiles

    elif termination_rule == "addition-Special_ONCC":
        return smiles.replace("*","[Be]")
    elif termination_rule == "addition-Urethane":
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 1:  # Ensure the `*` has exactly one neighbor
                    neighbor = neighbors[0]
                    atom_index = atom.GetIdx()
                    
                    if neighbor.GetSymbol() == "C":
                        # Replace `*` with NH2
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("O"))
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
                        
                    elif neighbor.GetSymbol() == "O":
                        # Replace N-* with Cl
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("Be"))  # Replace `*` with Be (any rare atom)

                        smarts_pattern = "[#7][#6](=[#8])[#8]([Be])" 
                        replacement_smiles = "[#7]=[#6]=[#8]" 
                        replace_mol = Chem.MolFromSmiles(replacement_smiles)
                        # Perform substructure replacement
                        edit_mol = Chem.ReplaceSubstructs(edit_mol, Chem.MolFromSmarts(smarts_pattern), replace_mol, replaceAll=False)[0]
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
    elif termination_rule == "addition-Urea":
        # print(concat_n_psmiles(smiles,n=2))
        return concat_n_psmiles(smiles,n=2).replace("*","[Be]")
    elif termination_rule == "addition-C=O":
        # print(concat_n_psmiles(smiles,n=2))
        return concat_n_psmiles(smiles,n=2).replace("*","[Be]")
    
    elif termination_rule == "addition-C=C_C=O":
        return smiles.replace("*","[Be]")
    
    elif termination_rule == "addition-CC_CO":
        return smiles.replace("*","[Be]")
    
    elif termination_rule == "addition-CCO_C":
        return smiles.replace("*","[Be]")
    elif termination_rule == "addition-Diene":
        smiles = concat_n_psmiles(smiles,n=2).replace("*","[Be]")
        # print(smiles)
        mol = Chem.MolFromSmiles(smiles)
        # Collect all indices of "Be" atoms
        be_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "Be"]
        # Remove Be atoms in reverse order
        edit_mol = Chem.RWMol(mol)
        for atom_index in sorted(be_indices, reverse=True):
            edit_mol.RemoveAtom(atom_index)  # Remove the atom
        Chem.SanitizeMol(edit_mol)
        mol = edit_mol  # Update the molecule

    elif termination_rule == "addition-Special_o1cc1":
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 1:  # Ensure the `*` has exactly one neighbor
                    neighbor = neighbors[0]
                    atom_index = atom.GetIdx()
                    if neighbor.GetSymbol() == "C":
                        # Replace *-CC(O)C- with O1CC1C-
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("Be"))  # Replace `*` with Be (any rare atom)
                        smarts_pattern = "[Be][C][C]([O])" 
                        replacement_smiles = "[C]1[C]([O]1)" 
                        replace_mol = Chem.MolFromSmiles(replacement_smiles)
                        # Perform substructure replacement
                        edit_mol = Chem.ReplaceSubstructs(edit_mol, Chem.MolFromSmarts(smarts_pattern), replace_mol, replaceAll=False)[0]
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
                        # print("## ",Chem.MolToSmiles(mol))
                        
                    elif neighbor.GetSymbol() == "O":
                        # Replace O-* with -OH
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("H"))  # Replace `*` with H
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
                        # print("#### ",Chem.MolToSmiles(mol))

    elif termination_rule == "addition-C=C_N":
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 1:  # Ensure the `*` has exactly one neighbor
                    neighbor = neighbors[0]
                    atom_index = atom.GetIdx()
                    if neighbor.GetSymbol() == "C":
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("Be"))  # Replace `*` with Be (any rare atom)
                        smarts_pattern = "[C][C]([Be])" 
                        replacement_smiles = "[C]=[C]" 
                        replace_mol = Chem.MolFromSmiles(replacement_smiles)
                        # Perform substructure replacement
                        edit_mol = Chem.ReplaceSubstructs(edit_mol, Chem.MolFromSmarts(smarts_pattern), replace_mol, replaceAll=False)[0]
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
                        print("@@ ",Chem.MolToSmiles(mol))
                    elif neighbor.GetSymbol() == "N":
                        # Replace N-* with N-H
                        edit_mol = Chem.RWMol(mol)
                        edit_mol.ReplaceAtom(atom_index, Chem.Atom("Be"))  # Replace `*` with Be (any rare atom)
                        # edit_mol.ReplaceAtom(atom_index, Chem.Atom("H"))  # Replace `*` with H
                        smarts_pattern = "[#7](-[Be])" 
                        replacement_smiles = "[#7]" 
                        replace_mol = Chem.MolFromSmiles(replacement_smiles)
                        edit_mol = Chem.ReplaceSubstructs(edit_mol, Chem.MolFromSmarts(smarts_pattern), replace_mol, replaceAll=False)[0]
                        Chem.SanitizeMol(edit_mol)
                        mol = edit_mol
                        print("@@@ ",Chem.MolToSmiles(mol))

    elif termination_rule.startswith("ring_opening-"):
        if termination_rule == "ring_opening-N_C=O":
            substructure_smarts = "[N][C](=[O])"
            substructure = Chem.MolFromSmarts(substructure_smarts)
            if mol.HasSubstructMatch(substructure):
                final_smiles = connect_neighbors_and_remove_attachments(smiles)
                return final_smiles
        # replace all * with [Be]
        smiles = smiles.replace("*", "[Be]")
        # print(smiles)
        mol = Chem.MolFromSmiles(smiles)
    else:
        pass
        # print("Unknowing termination type")
    
    return Chem.MolToSmiles(mol)
def find_attachment_and_neighbor(mol):
    """
    Find the attachment point (*) and its neighbor.
    """
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == '*':
            neighbors = atom.GetNeighbors()
            if len(neighbors) == 1:
                return atom.GetIdx(), neighbors[0].GetIdx()
    raise ValueError("Attachment point '*' not found or has multiple neighbors.")

def concat_molecules_with_attachment(smiles1, smiles2):
    """
    Concatenate two molecules by their attachment points (*).
    """
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    attach1, neighbor1 = find_attachment_and_neighbor(mol1)
    attach2, neighbor2 = find_attachment_and_neighbor(mol2)
    
    combined = rdmolops.CombineMols(mol1, mol2)
    
    offset = mol1.GetNumAtoms()
    neighbor2 += offset
    attach2 += offset
    
    editable_combined = Chem.EditableMol(combined)
    editable_combined.AddBond(neighbor1, neighbor2, order=Chem.BondType.SINGLE)
    editable_combined.RemoveAtom(max(attach1, attach2))
    editable_combined.RemoveAtom(min(attach1, attach2))
    
    final_molecule = editable_combined.GetMol()
    Chem.SanitizeMol(final_molecule)
    final_smiles = Chem.MolToSmiles(final_molecule)
    return final_smiles

def find_all_attachment_and_neighbors(mol):
    """
    Find all attachment points (*) and their neighbors.

    Returns:
        attachment_indices: List of indices of * atoms.
        neighbor_indices: List of indices of the neighbors of * atoms.
    """
    attachment_indices = []
    neighbor_indices = []
    
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == '*':
            neighbors = atom.GetNeighbors()
            if len(neighbors) == 1:  # Ensure * has only one neighbor
                attachment_indices.append(atom.GetIdx())
                neighbor_indices.append(neighbors[0].GetIdx())
    if len(attachment_indices) != 2:
        raise ValueError("The molecule does not contain exactly two attachment points (*)")
    
    return attachment_indices, neighbor_indices

def connect_neighbors_and_remove_attachments(smiles):
    """
    Connect the two neighbors of attachment points (*) and remove the * atoms.

    Args:
        smiles (str): Input SMILES string with two * attachment points.

    Returns:
        str: Final concatenated SMILES after connecting neighbors.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Invalid SMILES input")

    attachment_indices, neighbor_indices = find_all_attachment_and_neighbors(mol)

    # Combine and edit the molecule
    editable_mol = Chem.EditableMol(mol)
    editable_mol.AddBond(neighbor_indices[0], neighbor_indices[1], order=Chem.BondType.SINGLE)
    
    # Remove * atoms (in descending order to keep indices consistent)
    for idx in sorted(attachment_indices, reverse=True):
        editable_mol.RemoveAtom(idx)
    
    # Finalize and sanitize the molecule
    final_molecule = editable_mol.GetMol()
    Chem.SanitizeMol(final_molecule)
    final_smiles = Chem.MolToSmiles(final_molecule)
    return final_smiles

def run_reaction_monomer_split(smiles_polymer, reaction_template, termination_rule):
    """
    Execute the reaction template on the polymer with monomer split issue and handle subreactions.
    
    Parameters:
        smiles (str): The SMILES of the polymer.
        template (str): Reaction template in SMARTS format.
        
    Returns:
        list: Final products after handling subreactions.
    """
    polymer = Chem.MolFromSmiles(smiles_polymer)
    reaction = AllChem.ReactionFromSmarts(reaction_template)
    
    if termination_rule in ["condensation-Ester_OH_split", 
                            ]: 
        # Run the initial reaction
        initial_product = reaction.RunReactants((polymer,))[0] #only get the first is enough
        final_products = []
        sub_products = []
        for prod in initial_product:
            # Run further subreaction on each product
            subreaction_products = reaction.RunReactants((prod,))
            for subproduct_set in subreaction_products:
                sub_products.extend([Chem.MolToSmiles(sp) for sp in subproduct_set])
            
        # Process final products
        # this only works for "condensation-Ester_OH_split"
        oh_products = [p for p in sub_products if p == "[OH]"]  # Identifying [OH] product
        star_products = [p for p in sub_products if '*' in p and p.count('*') == 1]
        o_star_o = [
                p for p in sub_products 
                if p not in oh_products and p not in star_products
            ]  # Remaining is O-*-O
            
        if len(star_products) == 2:
            concatenated = concat_molecules_with_attachment(star_products[0], star_products[1])
            final_products.append(concatenated)
            
        final_products.extend(o_star_o)

    elif termination_rule in ["condensation-Amide_OH_split", 
                              "condensation-Amide_Cl_split", 
                              "condensation-Amide_Cl_split_2", 
                              "condensation-Amide_OH_split_2",
                              "condensation-s_c_n_OH",
                              "condensation-s_c_n_Cl",
                              "condensation-Ester_Cl_split", 
                              "addition-Urethane_splt",
                              "condensation-o_c_c_n_n",
                              "condensation-Amide_S_Cl_split",
                            ]: 
        # Run the initial reaction
        initial_product = reaction.RunReactants((polymer,))[0] #only get the first is enough
        final_products = []
        sub_products = []
        for prod in initial_product:
            # Run further subreaction on each product
            subreaction_products = reaction.RunReactants((prod,))
            if len(subreaction_products)==0:
                sub_products.extend([Chem.MolToSmiles(prod)])
            for subproduct_set in subreaction_products:
                sub_products.extend([Chem.MolToSmiles(sp) for sp in subproduct_set])
        # print(f"DEBUG: {sub_products}")
        # Process final products
        # this only works for "condensation-Amide_OH_split"
        star_products = [p for p in sub_products if '*' in p and p.count('*') == 1]
        n_star_n = [
                p for p in sub_products 
                if p not in star_products
            ]  # Remaining is N-*-N
            
        if len(star_products) == 2:
            concatenated = concat_molecules_with_attachment(star_products[0], star_products[1])
            final_products.append(concatenated)
            
        final_products.extend(n_star_n)

    elif termination_rule in ["condensation-COOH_OH_split", 
                              "condensation-Amide_OCC",
                            ]: 
        # Run the initial reaction
        initial_product = reaction.RunReactants((polymer,))[0]
        final_products = []
        sub_products = []
        for prod in initial_product:
            sub_products.extend([Chem.MolToSmiles(prod)])
        star_products = [p for p in sub_products if '*' in p and p.count('*') == 1]
        if len(star_products) == 2:
            concatenated = concat_molecules_with_attachment(star_products[0], star_products[1])
            final_products.append(concatenated)
            final_products.append(concatenated)
    
    elif termination_rule in ["condensation-Imide_split",
                              "condensation-Imide_split_2",
                              "condensation-Imide_split_3",
                              "condensation-N_=O",
                              "condensation-O_C=N_ring",
                              "condensation-O_C=N_ring_Cl",
                              "condensation-C=N_c",
                              ]:
        # Run the initial reaction
        initial_product = reaction.RunReactants((polymer,))[0]
        final_products = []
        sub_products = []
        for prod in initial_product:
            # Run further subreaction on each product
            subreaction_products = reaction.RunReactants((prod,))
            if len(subreaction_products)==0:
                print(Chem.MolToSmiles(prod))
                sub_products.extend([Chem.MolToSmiles(prod)])
            for j, subproduct_set in enumerate(subreaction_products):
                if j==0:
                    sub_products.extend([Chem.MolToSmiles(sp) for sp in subproduct_set])
        star_products = [p for p in sub_products if '*' in p and p.count('*') == 1]
        n_star_n = [
                p for p in sub_products 
                if p not in star_products
            ]  # Remaining is 2N-*-2N
            
        if len(star_products) == 2:
            concatenated = concat_molecules_with_attachment(star_products[0], star_products[1])
            final_products.append(concatenated)
        final_products.extend(n_star_n)

    elif termination_rule in ["condensation-c_O_c_split",
                              "condensation-c_O_c_split_Cl",
                              "addition-Urea_split",
                              "condensation-Special_NN_split",
                              "condensation-c_N_c_split",
                              ]:# this is for very symmertical linkage -> get all possible reactions so that at least one of them is the GT
        # Run the initial reaction
        initial_products = reaction.RunReactants((polymer,))
        final_products_ls = []
        for initial_product in initial_products: #e.g., reaction 0, 1, 2
            # final_products = []
            # simple_sub_products = []

            # first get the simple prod (level-1)
            for prod in initial_product:  #e.g., reaction 0 - prod0+prod1
                # Run further subreaction on each product
                subreaction_products = reaction.RunReactants((prod,))
                if len(subreaction_products)==0:
                    # sub_products.extend([Chem.MolToSmiles(prod)])
                    simple_sub_product = Chem.MolToSmiles(prod)

            # then get the complex one (level-2)
            for prod in initial_product:  #e.g., reaction 0 - prod0+prod1
                # Run further subreaction on each product
                subreaction_products = reaction.RunReactants((prod,))

                for subproduct_set in subreaction_products:
                    final_products = []
                    sub_products = [simple_sub_product]
                    sub_products.extend([Chem.MolToSmiles(sp) for sp in subproduct_set])
                    # Process final products
                    star_products = [p for p in sub_products if '*' in p and p.count('*') == 1]
                    n_star_n = [
                            p for p in sub_products 
                            if p not in star_products
                        ]  
                    if len(star_products) == 2:
                        concatenated = concat_molecules_with_attachment(star_products[0], star_products[1])
                        final_products.append(concatenated)
                        final_products.extend(n_star_n)
                    if len(final_products)==2:
                        final_products_ls.append(final_products)
        return final_products_ls
        
    else:
        raise ValueError("Unsupoorted termination_rule type for monomer split reactions!")
    
    return [final_products] if len(final_products)==2 else None

def depolymerize(smiles_polymer, reaction_template, termination_rule):
    """
    Depolymerize a polymer into its monomers using a reaction template.
    Only keeps products that are fully bonded.
    """
    smiles_products_ls = []
    unique_products = set()  # To store unique SMILES products

    polymer_smiles_modified = modify_terminal_atoms(smiles_polymer, termination_rule)
    # Convert SMILES notation to RDKit molecule object
    polymer_ori = Chem.MolFromSmiles(smiles_polymer)

    polymer = Chem.MolFromSmiles(polymer_smiles_modified)

    if termination_rule in ["condensation-COOH_OH", 
                            "condensation-c_s_c",
                            "condensation-c_n_c",
                            "condensation-_Si_",
                            "addition-Vinyl_double",
                            "condensation-c_OH",
                            "ring_opening-N_C=O",
                            "condensation-c_CCl",
                            ]: # no reaction template
        return [[polymer_smiles_modified]]
    
    if termination_rule in ["condensation-Ester_OH_split", 
                            "condensation-Amide_OH_split",
                            "condensation-Amide_Cl_split",
                            "condensation-Amide_Cl_split_2", 
                            "condensation-Amide_OH_split_2",
                            "condensation-s_c_n_OH",
                            "condensation-s_c_n_Cl",
                            "condensation-COOH_OH_split",
                            "condensation-Amide_OCC",
                            "condensation-Ester_Cl_split", 
                            "condensation-Imide_split",
                            "condensation-Imide_split_2",
                            "condensation-Imide_split_3",
                            "condensation-N_=O",
                            "condensation-O_C=N_ring",
                            "condensation-O_C=N_ring_Cl",
                            "condensation-C=N_c",
                            "addition-Urethane_splt",
                            "condensation-o_c_c_n_n",
                            "addition-Urea_split",
                            "condensation-c_O_c_split",
                            "condensation-c_O_c_split_Cl",
                            "condensation-Amide_S_Cl_split",
                            "condensation-Special_NN_split",
                            "condensation-c_N_c_split",
                            ]: 
        return run_reaction_monomer_split(smiles_polymer, reaction_template, termination_rule)
    
    # Create a reaction template
    reaction = AllChem.ReactionFromSmarts(reaction_template)

    # Apply the reaction to the polymer
    products = reaction.RunReactants((polymer,))
    # print("Num of reactions:",len(products))
    
    
    duplicate_count = 0  # Counter for duplicates
    for i,product in enumerate(products):
        # Filter products to keep only fully bonded molecules
        if all(is_meaningful(m, 2) for m in product):
            # Convert the product molecules to SMILES notation
            smiles_products = tuple(sorted(Chem.MolToSmiles(product_mono) for product_mono in product)) # [Chem.MolToSmiles(product_mono) for product_mono in product]
            if smiles_products not in unique_products:
                unique_products.add(smiles_products)  # Add to unique products
                # # Plot the molecules
                # if len(smiles_products) == 1:
                #     mols_to_plot = [polymer_ori, Chem.MolFromSmiles(smiles_products[0])]
                #     legend = ["Polymer", "Monomer"]
                #     img = Draw.MolsToGridImage(mols_to_plot, molsPerRow=2, legends=legend, subImgSize=(300, 300),returnPNG=False)
                # else:
                #     mols_to_plot = [polymer_ori, Chem.MolFromSmiles(smiles_products[0]),Chem.MolFromSmiles(smiles_products[1])]#products[0][0], products[0][1]
                #     legend = ["Polymer", "Monomer 1", "Monomer 2"]
                #     img = Draw.MolsToGridImage(mols_to_plot, molsPerRow=3, legends=legend, subImgSize=(300, 300),returnPNG=False)

                # img.save(f'./reaction_{smiles_polymer.replace("/","_")}_{i}.png')
                smiles_products_ls.append(smiles_products)
            else:
                duplicate_count += 1
                # print(f"Duplicate found! Skipping...")
        else:
            continue
            # print(f"Reaction {i} skipped: Contains not meaningful products")
    # if duplicate_count > 0:
    #     print(f"Total duplicates found and skipped: {duplicate_count}")
    return smiles_products_ls


def standardize_smiles(smiles):
    """
    Standardize a SMILES string using RDKit.
    Ensures consistent representation of the same molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Chem.MolToSmiles(mol, canonical=True)
    return None

def check_accuracy(group, successful_results):
    """
    Check if there is a match between the ground truth (group) and the successful results.
    Standardizes all SMILES, and doubles single monomers for comparison.

    Parameters:
    - group: DataFrame group for a specific PID.
    - successful_results: List of successful reaction results for that PID.

    Returns:
    - 1 if there is at least one complete match between ground truth and successful results, otherwise 0.
    """
    # Extract and standardize ground truth reactions
    ground_truth_reactions = []
    for _, row in group.iterrows():
        monomer_smiles = []
        if pd.notna(row["SMILES_CID1"]):
            monomer_smiles.append(standardize_smiles(row["SMILES_CID1"]))
        if pd.notna(row["SMILES_CID2"]):
            monomer_smiles.append(standardize_smiles(row["SMILES_CID2"]))
        
        # Double single monomer for comparison
        if len(monomer_smiles) == 1:
            monomer_smiles.append(monomer_smiles[0])
        
        # Only add valid and standardized monomers
        if all(monomer_smiles):
            ground_truth_reactions.append(sorted(monomer_smiles))

    # Standardize successful results and compare
    for result in successful_results:
        result_products = [standardize_smiles(prod) for prod in result["products"]]
        
        # Double single monomer for comparison
        if len(result_products) == 1:
            result_products.append(result_products[0])
        
        # Ensure all products are valid and standardized
        if None not in result_products and sorted(result_products) in ground_truth_reactions:
            return 1  # Found a complete match

    return 0  # No match found



def summarize_gt_and_results_from_df(df, templates, savepath):
    """
    Summarize the successful depolymerization results into an image.
    Only support linear polymers (filtering/terminating conditions do not consider ladder polymers and more)

    Returns:
    - Saves a high-resolution summary image of the reactions with the polymer, template, and resulting reactants.
    """
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    accuracy_ls = []
    success_coverage_ls = []  # List to track polymers with successful results
    gt_polymerization_ls = []  # Store ground truth polymerization info
    predicted_polymerization_ls = []  # Store predicted polymerization info
    # Font settings
    try:
        # font = ImageFont.truetype("arial.ttf", 180)  # Larger font size
        font = ImageFont.truetype(font_path, size=20)
    except IOError:
        print("LOAD Default Font")
        font = ImageFont.load_default()


    for pid in tqdm(df["PID"].unique()):
        print(f"Processing Polymer ID: {pid}")
        group = df[df["PID"]==pid]

        successful_results = []  # To store successful reactions for visualization        
        polymer_smiles = group["SMILES"].unique()[0]
        # print(polymer_smiles)
        # Process each template
        for reaction_type, subtypes in templates.items():
            for subtype, reaction_template in subtypes.items():
                termination_rule = f"{reaction_type}-{subtype}"
                try:
                    print(f"Trying Template: {termination_rule}")
                    
                    # # **Preliminary Check**
                    # # avoid link breakage on the side chains
                    # # Apply reaction to the original polymer SMILES (without termination rules; Note: Be is used in all templates!)
                    # polymer_ori = Chem.MolFromSmiles(polymer_smiles.replace("*", "[Be]"))
                    # reaction = AllChem.ReactionFromSmarts(reaction_template)
                    # products = reaction.RunReactants((polymer_ori,))
                    # # Check if there is at least one applicable product set
                    # template_applicable = False  # Assume the template is not applicable initially
                    # for product_set in products:
                    #     all_products_valid = True  # Flag to check all products in the set
                    #     for product in product_set:
                    #         smiles_product = Chem.MolToSmiles(product)
                    #         if smiles_product.count('Be') >= 2:
                    #             all_products_valid = False  # A product in this set is invalid
                    #             break  # Exit the inner loop for this product set
                    #     if all_products_valid:
                    #         # At least one valid product set found
                    #         template_applicable = True
                    #         break  # Exit the outer loop
                    # if not template_applicable:
                    #     print(f"Template {termination_rule} is not applicable to polymer {pid}")
                    #     continue  # Skip to the next template
                    
                    # **Proceed with Depolymerization**
                    result = depolymerize(polymer_smiles, reaction_template, termination_rule)
                    if result:
                        for product in result:
                            ### product validity check TODO (add more rules)
                            # Ensure all products are valid molecules  
                            # and do not contain '*'
                            # and do not contain 'Be'
                            valid_products = [
                                Chem.MolFromSmiles(prod) for prod in product 
                                if Chem.MolFromSmiles(prod) is not None and '*' not in prod and "Be" not in prod
                            ]
                            if len(valid_products) == len(product):  # Only add if all products are valid
                                scores = [calculate_monomer_syn_score(prod) for prod in product]
                                successful_results.append({
                                    "template": f"{reaction_type}-{subtype}",
                                    "products": product,
                                    "SAscore_Products": scores,
                                    "PolyScore": hmean(scores) if scores else None,
                                })
                            else:
                                print(f"Invalid product encountered: {product}")
                except Exception as e:
                    print(f"Error processing template {reaction_type}-{subtype}: {str(e)}")
                    continue
        print("#Successful Reactions: ", len(successful_results))
       
        # Add predicted polymerization info as a JSON-like string
        predicted_polymerization_ls.append({
            "PID": pid,
            "predicted": json.dumps(successful_results)
        })

        # check accuracy
        accuracy = check_accuracy(group, successful_results)
        accuracy_ls.append({"PID": pid, "correct": accuracy})
        # Check if this polymer has at least one successful retro result
        success_coverage_ls.append({"PID": pid, "has_success": int(len(successful_results) > 0)})

        # Generate the image summary
        rows = len(successful_results) + len(group)
        row_height = 400  # Height of each row (larger for better resolution)
        img_width = 1800  # Increased width for better clarity
        img_height = rows * row_height

        # Create a blank white image
        summary_img = Image.new("RGB", (img_width, img_height), "white")
        draw = ImageDraw.Draw(summary_img)
        
        # Collect ground truth polymerization
        ground_truth_polymerization = []
        # Draw ground truth from dataframe
        for j, (index, row) in enumerate(group.iterrows()):
            polymerization = {
                "Type": row["Type"],
                "SMILES_CID1": row["SMILES_CID1"] if pd.notna(row["SMILES_CID1"]) else None,
                "SMILES_CID2": row["SMILES_CID2"] if pd.notna(row["SMILES_CID2"]) else None
            }
             # Add SAscore to ground truth monomers
            scores = []
            polymerization["SAscore_CID1"] = (
                calculate_monomer_syn_score(row["SMILES_CID1"]) if pd.notna(row["SMILES_CID1"]) else None
            )
            if polymerization["SAscore_CID1"] is not None:
                scores.append(polymerization["SAscore_CID1"])

            polymerization["SAscore_CID2"] = (
                calculate_monomer_syn_score(row["SMILES_CID2"]) if pd.notna(row["SMILES_CID2"]) else None
            )
            if polymerization["SAscore_CID2"] is not None:
                scores.append(polymerization["SAscore_CID2"])

            # Calculate PolyScore (harmonic mean)
            polymerization["PolyScore"] = hmean(scores) if scores else None

            ground_truth_polymerization.append(polymerization)

            y_offset = j * row_height
            # Polymer image
            polymer_img = Draw.MolsToImage([Chem.MolFromSmiles(row["SMILES"])], subImgSize=(410, 320))  # Larger size
            template_text = row["Type"]
                
            monomer_smiles = []
            if pd.notna(row["SMILES_CID1"]):
                # print(row["SMILES_CID1"])
                monomer_smiles.append(row["SMILES_CID1"])
            if pd.notna(row["SMILES_CID2"]):
                # print(row["SMILES_CID2"])
                monomer_smiles.append(row["SMILES_CID2"])

            # Products image
            if monomer_smiles:
                products_img = Draw.MolsToImage(
                    [Chem.MolFromSmiles(prod) for prod in monomer_smiles],
                    subImgSize=(410, 350),  # Larger size
                    legends=["Monomer 1", "Monomer 2"][:len(monomer_smiles)]  # Adjust legend for single/double products
                )
            
            # Convert PIL images to pasteable formats
            polymer_pil = polymer_img.convert("RGBA")
            products_pil = products_img.convert("RGBA")
            # Paste the polymer image
            summary_img.paste(polymer_pil, (20, y_offset))
            draw.text(
                    (20, y_offset+330),
                    f"(PolyScore: {polymerization['PolyScore']:.2f})" if polymerization["PolyScore"] is not None else "(PolyScore: N/A)",
                    fill="black",
                    font=font
                )
            # Paste the products image
            summary_img.paste(products_pil, (850, y_offset))
            for i, score in enumerate(scores):
                draw.text(
                    (850 + i * 450, y_offset+340),  # Position below each monomer image
                    f"(SAscore: {score:.2f})",
                    fill="black",
                    font=font
                )
            # Draw the reaction template text
            draw.text((400, y_offset + 4), f"GT: {template_text}", fill="black", font=font)
            

        # Draw each reaction row
        for i, result in enumerate(successful_results):
            y_offset = (i+j+1) * row_height
            # Polymer image
            polymer_img = Draw.MolsToImage([Chem.MolFromSmiles(polymer_smiles)], subImgSize=(410, 350))  # Larger size
            template_text = result["template"] 
                
            # Products image
            products_img = Draw.MolsToImage(
                [Chem.MolFromSmiles(prod) for prod in result["products"]],
                subImgSize=(410, 350),  # Larger size
                legends=["Monomer 1", "Monomer 2"][:len(result["products"])]  # Adjust legend for single/double products
            )

            # Convert PIL images to pasteable formats
            polymer_pil = polymer_img.convert("RGBA")
            products_pil = products_img.convert("RGBA")
            # Paste the polymer image
            summary_img.paste(polymer_pil, (20, y_offset))
            draw.text(
                (20, y_offset + 330),
                f"(PolyScore: {result['PolyScore']:.2f})" if result["PolyScore"] is not None else "(PolyScore: N/A)",
                fill="black",
                font=font
            )
            # Paste the products image
            summary_img.paste(products_pil, (850, y_offset))
            monomer_scores = result["SAscore_Products"]
            for i, score in enumerate(monomer_scores):
                draw.text(
                    (850 + i * 450, y_offset + 340),
                    f"(SAscore: {score:.2f})",
                    fill="black",
                    font=font
                )
            # Draw the reaction template text
            draw.text((400, y_offset + 10), f"RS: {template_text}", fill="black", font=font)
            
        #################
        # Save the summary image
        summary_img.save(f"{savepath}/{pid}.png", dpi=(400, 400))  # Save with high resolution
        print("Summary image saved!")

        # Add ground truth info as a JSON-like string
        gt_polymerization_ls.append({
            "PID": pid,
            "ground_truth": json.dumps(ground_truth_polymerization)
        })
    
    # save accuracy and coverage report
    accuracy_df = pd.DataFrame(accuracy_ls)
    success_coverage_df = pd.DataFrame(success_coverage_ls)
    gt_polymerization_df = pd.DataFrame(gt_polymerization_ls)
    predicted_polymerization_df = pd.DataFrame(predicted_polymerization_ls)

    # combined_df = pd.merge(accuracy_df, success_coverage_df, on="PID")
    combined_df = pd.merge(accuracy_df, success_coverage_df, on="PID")
    combined_df = pd.merge(combined_df, gt_polymerization_df, on="PID")
    combined_df = pd.merge(combined_df, predicted_polymerization_df, on="PID")

    output_csv_path = f"{savepath}/reaction_accuracy_coverage_results.csv"
    combined_df.to_csv(output_csv_path, index=False)
    
    # Calculate the average accuracy
    average_accuracy = combined_df["correct"].mean()
    print(f"Average Accuracy: {average_accuracy:.2f}")
    # Calculate the Success Coverage Ratio
    success_coverage_ratio = combined_df["has_success"].mean()
    print(f"Success Coverage Ratio: {success_coverage_ratio:.2f}")

# Example templates dictionary
# reaction type - subtype - termination rule, reaction template
templates = {
    "condensation": {
        "Ester_Cl": "[O:1][C:2](=[O:5])>>[OH:1].[Cl:3][C:2](=[O:5])",#-Cl [DONE]
        "Ester_Cl_split": "[O:1][C:2](=[O:5])>>[OH:1].[Cl:3][C:2](=[O:5])",
        "Ester_Cl_2": "[O:1][C:2](=[O:5])>>[OH:1].[Cl:3][C:2](=[O:5])",#-Cl [DONE]
        "Ester_Cl_3": "[O:1][C:2](=[O:5])>>[OH:1].[Cl:3][C:2](=[O:5])",
        "Ester_OH": "[O:1][C:2](=[O:5])>>[OH:1].[OH:3][C:2](=[O:5])", #-OH [DONE]
        "Ester_OH_split": "[O:1][C:2](=[O:5])>>[OH:1].[OH:3][C:2](=[O:5])", #TODO
        "Ester_O_CH3": "[O:1][C:2](=[O:5])>>[OH:1].[C][O][C:2](=[O:5])",
        "Ester_O_CCH3": "[O:1][C:2](=[O:5])>>[OH:1].[C][C][O][C:2](=[O:5])",
        "Ester_O_CCO": "[O:1][C:2](=[O:5])>>[O:1][C](=[O])[C].[C:2](=[O:5])[O]",
        "Imide": "[#6:1][#6](=[#8:2])[#7:6][#6](=[#8:5])[#6:4]>>[NH2:6].[#6:1][#6](=[#8:2])[#8:3][#6](=[#8:5])[#6:4]", #[Half Done]
        "Imide_split": "[#6:1][#6](=[#8:2])[#7:6][#6](=[#8:5])[#6:4]>>[NH2:6].[#6:1][#6](=[#8:2])[#8:3][#6](=[#8:5])[#6:4]",
        "Imide_split_2": "[#6:1][#6:2](=[#8:3])[#7:4][#6:5](=[#8:6])[#6:7]>>[NH2:4].[#8:3]=[#6:2]([O])[#6:1][#6:7][#6:5]([O])=[#8:6]",
        "Imide_split_3": "[#6:1][#6](=[#8:2])[#7:6][#6](=[#8:5])[#6:4]>>[#7:6]=[#6]=[#8].[#6:1][#6](=[#8:2])[#8:3][#6](=[#8:5])[#6:4]",
        # "Amide_Cl": "[N:1][C:3](=[O:6])>>[N:1].[Cl:2][C:3](=[O:6])", # Amine + Acyl Chloride #[Half Done]
        # "Amide_OH": "[N:1][C:3](=[O:6])>>[NH2:1].[OH:2][C:3](=[O:6])", # Amine + Carboxylic Acid #[Half Done]
        "Amide_Cl": "[#7:1][C:3](=[O:6])>>[#7:1].[Cl:2][C:3](=[O:6])", # Amine + Acyl Chloride #[Half Done]
        "Amide_Cl_split": "[#7:1][C:3](=[O:6])>>[#7:1].[Cl:2][C:3](=[O:6])",
        "Amide_Cl_split_2": "[NH:1][C:2](=[O:3])>>[NH2:1].[Cl][C:2](=[O:3])",
        "Amide_S_Cl": "[#7:1][S:2]>>[#7:1].[S:2][Cl:3]",
        "Amide_S_Cl_split": "[#7:1][S:2]>>[#7:1].[S:2][Cl:3]",
        "Amide_Cl_O_Benz": "[#7:1][C:3](=[O:6])>>[#7:1].[C:3](=[O:6])[O][c]1[c][c][c][c][c]1", #TODO the benzene ring has issue!!! (P100753)
        "Amide_Cl_2": "[#7:1][C:3](=[O:6])>>[#7:1].[Cl:2][C:3](=[O:6])",
        "Amide_OH": "[#7:1][C:3](=[O:6])>>[#7:1].[OH:2][C:3](=[O:6])", # Amine + Carboxylic Acid #[Half Done]
        "Amide_OH_split": "[#7:1][C:3](=[O:6])>>[#7:1].[OH:2][C:3](=[O:6])",
        "Amide_OH_split_2": "[NH:1][C:3](=[O:6])>>[NH2:1].[OH:2][C:3](=[O:6])",
        "Amide_OH_2": "[#7:1][C:3](=[O:6])>>[#7:1].[OH:2][C:3](=[O:6])",
        "S_Cl": "[S:1][#6:2]>>[S:1][Cl].[#6:2]", #
        "S_C=O": "[S:1][C:2](=[O:3])>>[S:1].[C:2](=[O:3])[Cl]", #
        "S_C=O_2": "[S:1][C:2](=[O:3])>>[S:1].[C:2](=[O:3])[Cl]", #
        "c_O_Br": "[C:1][O:2][c:3]>>[C:1][Br].[O:2][c:3]",
        "c_Cl_O": "[C:1][O:2][c:3]>>[C:1][O:2].[c:3][Cl]",
        "N_Cl_c": "[N:1][c:2]>>[N:1][Cl].[c:2]",
        "COOH_OH": "", # no reaction template, only modify the p-smiles; modify depolymerize()
        "COOH_OH_split": "[O:1][C:2](=[O:5])>>[OH:1].[OH:3][C:2](=[O:5])",
        "P_O": "[P:1][O:2]>>[P:1][Cl].[O:2]",
        "P_O_2": "[P:1][O:2]>>[P:1][Cl].[O:2]",
        "c_O_c": "[c:1][O:2][c:3]>>[c:1][O:2].[c:3][F]",
        "c_O_c_Cl": "[c:1][O:2][c:3]>>[c:1][O:2].[c:3][Cl]",
        "c_O_c_2": "[c:1][O:2][c:3]>>[c:1][O:2].[c:3][F]",
        "c_O_c_Cl_2": "[c:1][O:2][c:3]>>[c:1][O:2].[c:3][Cl]",
        "c_O_c_split": "[c:1][O:2][c:3]>>[c:1][O:2].[c:3][F]",
        "c_O_c_split_Cl": "[c:1][O:2][c:3]>>[c:1][O:2].[c:3][Cl]",
        "c_s_c": "", # no reaction template, only modify the p-smiles
        "c_n_c": "",
        "_Si_": "",
        # "c_C(=O)_c": "",
        "c_OH":"",
        "c_CCl":"",
        "s_c_n_OH":"[#6:2]1[#7:3][c:4][c:5][#16:6]1>>[#6:2](=[O])O.[#7:3][c:4][c:5][#16:6]",
        "s_c_n_Cl":"[#6:2]1[#7:3][c:4][c:5][#16:6]1>>[#6:2](=[O])Cl.[#7:3][c:4][c:5][#16:6]",
        "Amide_OCC":"[C:1](=[O:2])[N:3]>>[N:3].[C:1](=[O:2])[O][C][C]",
        "N_=O": "[#7:1]1[#6:2][#6:3][#7:4][c:5]2[c:6]1[c:7][c:8][c:9][c:10]2>>[#7:4][c:5]2[c:6]([c:7][c:8][c:9][c:10]2)[#7:1].[#6:2](=[O])[#6:3](=[O])",
        "O_C=N_ring": "[#8:1]1[#6:2][#7:3][#6:4][#6:5]1>>[#6:2](=[#8])[#8].[#8:1][#6:5][#6:4][#7:3]",
        "O_C=N_ring_Cl": "[#8:1]1[#6:2][#7:3][#6:4][#6:5]1>>[#6:2](=[#8])[Cl].[#8:1][#6:5][#6:4][#7:3]",
        "C=N_c":"[#6:1][#7:2]=[#6:3]>>[#6:1][#7:2].[#6:3]=[#8]",
        "o_c_c_n_n": "[#8:1]1[#6:2]([c:6])[#7:3][#7:4][#6:5]1>>[c:6][#6:2](=[O])[#7:3][#7:4].[#6:5](=[O])[Cl]",
        "Special_NN_split":"[#7:1][#7:2][#6:3][#7:4]>>[#7:1][#7:2].[#6:3](=[O])[O]",
        "c_N_c_split":"[c:1][NH:2][c:3]>>[c:1][NH2:2].[c:3][Br]",
    },
    "addition": {
        "Vinyl": "[Be][C:1][C:2][C:3][C:4][Be]>>[C:1]=[C:2].[C:3]=[C:4]", #[Half Done]
        "Vinyl_2": "[Be][C:1][C:2][C:3][C:4][Be]>>[C:1]=[C:2].[C:3]=[C:4]",
        "Vinyl_3": "[Be][C:1][C:2][C:3][C:4][Be]>>[C:1]=[C:2].[C:3]=[C:4]",
        # "Vinyl_double": "",
        "Urethane": "[O:1][C:3](=[O:5])[N:4]>>[O:1].[O:5]=[C:3]=[N:4]", #[Half Done]
        "Urethane_splt": "[O:1][C:3](=[O:5])[N:4]>>[O:1].[O:5]=[C:3]=[N:4]",
        "Urea": "[N:1][C:3](=[O:5])[N:4]>>[N:1].[O:5]=[C:3]=[N:4]", #[Not working]
        "Urea_split": "[N:1][C:3](=[O:5])[N:4]>>[N:1].[O:5]=[C:3]=[N:4]", 
        "Diene": "[C:1]=[C:2][C:3]=[C:4]>>[C:1]#[C:2].[C:3]#[C:4]", #[Half Done]
        "Special_CCN": "[C:2][C:3][N:4]>>[C:2]#[C:3].[N:4]", #[Not start; cannot find examples]
        "C=O": "[Be][O:1][C:2][O:3][C:4][Be]>>[O:1]=[C:2].[O:3]=[C:4]",
        "Special_o1cc1": "[#8:1][#6:2][#6:3]([#8:4])[#6:5][#8:6]>>[#8:1].[#6:2]1[#6:3]([#8:4]1)[#6:5][#8:6]",
        "C=C_N": "[C:1][C:2][#7:3]>>[C:1]=[C:2].[#7:3]",
        "Special_ONCC": "[Be][O:1][N:2][C:3][C:4][Be]>>[O:1]=[N:2].[C:3]=[C:4]",
        "C=C_C=O": "[Be][O:1][C:2][C:3][C:4][Be]>>[O:1]=[C:2].[C:3]=[C:4]",
        "CN": "[Be][C:1][N:2][C:3][N:4][Be]>>[C:1]=[N:2].[C:3]=[N:4]",# no reaction template, only modify the p-smiles; modify depolymerize()
        "CC_CO":"[Be][C:1][C:2][C:3]([O:4])[Be]>>[C:1]=[C:2].[C:3]=[O:4]",
        "CCO_C":"[Be][C:1][C:2](=[O:4])[C:3][Be]>>[C:1]=[C:2].[C:3]=[O:4]",
    },
    "ring_opening": {
        "Imine": "[*][C:1][C:2][N:3]([*])[C:5]=[O:4]>>[C:1]1[C:2][N:3][C:5][O:4]1".replace("*", "Be"), #This is from Chen's work
        "Imine_new": "[*][C:1][C:2][N:3]([*])[C:5]=[O:4]>>[C:1]1[C:2][N:3]=[C:5][O:4]1".replace("*", "Be"), #This is the template based on PolyInfo
        "Oxide_3cycl": "[*][C:2][C:1][O:3][*]>>[C:2]1[O:3][C:1]1".replace("*", "Be"),
        "Oxide_4cycl": "[*][C:3][C:1][C:2][O:4][*]>>[C:1]1[C:2][O:4][C:3]1".replace("*", "Be"),
        "Oxide_7cycl": "[*][C:5][C:6][C:7][C:3][C:1][C:2][O:4][*]>>[C:5]1[C:6][C:7][C:3][C:1][C:2][O:4]1".replace("*", "Be"),
        "Oxide_4cyclX3": "[*][C:3][C:1][C:2][O:4][*]>>[C:1]1[C:2][O:4][C:3][C:1][C:2][O:4][C:3][C:1][C:2][O:4][C:3]1".replace("*", "Be"),
        "Diene": "[*][C:1]=[C:2][C:3][C:4][C:5][*]>>[C:1]1=[C:2][C:3][C:4][C:5]1".replace("*", "Be"), #[TODO-extend to more carbon in a ring (currently only for 5)]
        "Phosphazene": "[*][P:1]=[N:2][*]>>[P:1]1=[N:2][P:1]=[N:2][P:1]=[N:2]1".replace("*", "Be"),
        "N_4cycl": "[*][C:3][C:1][C:2][N:4][*]>>[C:1]1[C:2][N:4][C:3]1".replace("*", "Be"),
        "N_C=O": "",

    },
}

if __name__ == "__main__":
    args = get_args()
    np.random.seed(42)  # For reproducibility
    # # Load the dataset
    # # file_path = './data/final_chemical_reactions_summary_Nov27_2024_CID_SMILES_cleaned.csv'
    # file_path =  './data/final_chemical_reactions_summary_Dec10_2024_CID_SMILES_cleaned_v2.csv'
    # df = pd.read_csv(file_path)


    # # Get unique PIDs and split them into training (90%) and testing (10%) sets
    # unique_pids = df['PID'].unique()
    # np.random.shuffle(unique_pids)
    # train_pids = unique_pids[:int(0.9 * len(unique_pids))]
    # test_pids = unique_pids[int(0.9 * len(unique_pids)):]
    # # Split the dataframe based on PIDs
    # train_df = df[df['PID'].isin(train_pids)]
    # test_df = df[df['PID'].isin(test_pids)]

    # Save the datasets as CSV files
    train_file_path = './data/final_chemical_reactions_summary_Dec10_2024_CID_SMILES_cleaned_v2_training.csv'
    test_file_path = './data/final_chemical_reactions_summary_Dec10_2024_CID_SMILES_cleaned_v2_testing.csv'
    case_study_path = './data/point2_case_study.csv'
    # train_df.to_csv(train_file_path, index=False)
    # test_df.to_csv(test_file_path, index=False)

    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)
    case_df = pd.read_csv(case_study_path)
    dataset = args.dataset
    if dataset == 'train':
        # Randomly sample 300 unique PIDs and filter the dataframe
        unique_train_pids = train_df['PID'].unique()
        sampled_pids = np.random.choice(unique_train_pids, size=300, replace=False)
        df_draw = train_df[train_df['PID'].isin(sampled_pids)]
        savepath = "./result-retro-train"

        # df_draw = df_draw[df_draw["PID"]=="P500009"]
        # print(df_draw)
    elif dataset == 'test':
        df_draw = test_df
        savepath = "./result-retro-test"
    elif dataset == 'case':
        df_draw = case_df
        savepath = "./result-retro-case"
    summarize_gt_and_results_from_df(df_draw, templates, savepath)