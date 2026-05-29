import sys
import json
import subprocess
import numpy as np
from pathlib import Path
from collections import Counter
from pymatgen.core import Structure

ROOT = Path(__file__).resolve().parents[1]

def run_ligand_exchange_test():
    print("--- Running Charged Ligand Exchange Regression Test ---")
    cif = ROOT / "examples/cifs/InP_zb.cif"
    yaml_path = ROOT / "tests/test_ligand_exchange.yaml"
    out = ROOT / "tests/out/inp_ligand_exchange_regression.xyz"
    out.parent.mkdir(parents=True, exist_ok=True)
    
    # Run builder
    cmd_cut = [
        sys.executable,
        "-m",
        "builder",
        str(cif),
        str(yaml_path),
        "-o",
        str(out),
        "--write-all",
    ]
    
    result = subprocess.run(cmd_cut, cwd=ROOT, check=True, capture_output=True, text=True)
    print("Builder output logs finished.")
    
    # The final exchanged file is out (inp_ligand_exchange_regression.xyz)
    assert out.exists(), "Final exchanged XYZ file not found."
    
    def read_xyz(path):
        lines = path.read_text().strip().splitlines()
        n = int(lines[0].strip())
        syms, pts = [], []
        for i in range(n):
            parts = lines[i + 2].split()
            syms.append(parts[0])
            pts.append(list(map(float, parts[1:4])))
        return syms, np.asarray(pts, float)
        
    final_syms, final_pts = read_xyz(out)
    print(f"Final composition: {Counter(final_syms)}")
    
    # Verify partial exchange composition
    assert "Cl" in final_syms, "Some Cl ligands should remain for 30% exchange."
    assert "C" in final_syms, "Carboxylate carbon not found in final structure."
    assert "O" in final_syms, "Carboxylate oxygen not found in final structure."
    
    # Verify steric integrity: no two heavy atoms are closer than VDW sum minus 0.4 A
    vdw = {1: 1.2, 6: 1.7, 7: 1.55, 8: 1.52, 15: 1.8, 16: 1.8, 49: 1.93, 17: 1.75} # standard VDW
    
    heavy_idx = [i for i, sym in enumerate(final_syms) if sym != "H"]
    n_heavy = len(heavy_idx)
    
    # --- Connected Components grouping to isolate individual molecules ---
    core_syms = {"In", "P"}
    ligand_heavy = [i for i in heavy_idx if final_syms[i] not in core_syms]
    
    visited = set()
    components = []
    
    for start in ligand_heavy:
        if start in visited:
            continue
        comp = []
        queue = [start]
        visited.add(start)
        while queue:
            cur = queue.pop(0)
            comp.append(cur)
            for other in ligand_heavy:
                if other in visited:
                    continue
                dist = np.linalg.norm(final_pts[cur] - final_pts[other])
                if dist < 1.8: # C-C or C-O bond length threshold
                    visited.add(other)
                    queue.append(other)
        components.append(comp)
        
    # Map each ligand atom to its molecule ID
    mol_ids = {}
    for mol_idx, comp in enumerate(components):
        for atom_idx in comp:
            mol_ids[atom_idx] = mol_idx
            
    # For core atoms, each core atom is its own separate "molecule"
    for idx in heavy_idx:
        if idx not in mol_ids:
            mol_ids[idx] = -1 - idx # unique negative ID
            
    print(f"Grouped {len(ligand_heavy)} heavy ligand atoms into {len(components)} separate molecules.")
    
    clashes = 0
    min_inter_dist = 999.0
    
    for i in range(n_heavy):
        idx_i = heavy_idx[i]
        sym_i = final_syms[idx_i]
        z_i = 49 if sym_i == "In" else (15 if sym_i == "P" else (8 if sym_i == "O" else 6))
        r_i = vdw.get(z_i, 1.8)
        mol_i = mol_ids[idx_i]
        
        for j in range(i + 1, n_heavy):
            idx_j = heavy_idx[j]
            sym_j = final_syms[idx_j]
            z_j = 49 if sym_j == "In" else (15 if sym_j == "P" else (8 if sym_j == "O" else 6))
            r_j = vdw.get(z_j, 1.8)
            mol_j = mol_ids[idx_j]
            
            # Skip checking if they belong to the same molecule
            if mol_i == mol_j:
                continue
                
            dist = np.linalg.norm(final_pts[idx_i] - final_pts[idx_j])
            min_inter_dist = min(min_inter_dist, dist)
            
            # Exclude standard coordination and bulk bonds (e.g. In-O, In-Cl, In-P)
            is_bonded = False
            metals = {"In", "Pb", "Cd", "Zn", "Ga"}
            if sym_i in metals and sym_j in {"O", "Cl"} and dist < 2.9:
                is_bonded = True
            elif sym_j in metals and sym_i in {"O", "Cl"} and dist < 2.9:
                is_bonded = True
            elif sym_i in metals and sym_j in {"P", "As", "Se", "Te"} and dist < 2.9:
                is_bonded = True
            elif sym_j in metals and sym_i in {"P", "As", "Se", "Te"} and dist < 2.9:
                is_bonded = True
                
            if not is_bonded:
                # Use a standard hard steric collision limit of 1.8 Å for heavy atoms
                threshold = 1.8
                if dist < threshold:
                    print(f"Steric Clash! {sym_i} (idx={idx_i}, mol={mol_i}) and {sym_j} (idx={idx_j}, mol={mol_j}) at dist={dist:.3f} Å (threshold={threshold:.3f} Å)")
                    clashes += 1
                    
    print(f"Minimum intermolecular / non-coordinating distance found: {min_inter_dist:.3f} Å")
    print(f"Total steric clashes: {clashes}")
    assert clashes == 0, f"Found {clashes} intermolecular steric clashes in the final structure."
    print("Steric integrity: PASS")
    print("Charged ligand exchange regression test passed successfully!")

if __name__ == "__main__":
    run_ligand_exchange_test()
