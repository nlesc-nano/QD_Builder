import sys
import subprocess
import numpy as np
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

def test_neutral_ligand_uniform():
    print("--- Running Uniform Neutral Ligand Passivation Test ---")
    cif = ROOT / "examples/cifs/CdSe_zb.cif"
    yaml_path = ROOT / "tests/test_neutral_ligand_uniform.yaml"
    out = ROOT / "tests/out/cdse_neutral_ligand_uniform.xyz"
    out.parent.mkdir(parents=True, exist_ok=True)
    
    # Run builder
    cmd = [
        sys.executable,
        "-m",
        "builder",
        str(cif),
        str(yaml_path),
        "-o",
        str(out),
        "--write-all",
    ]
    
    result = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    print("Builder output log:")
    print(result.stdout)
    
    assert out.exists(), "Output XYZ file not found."
    
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
    comp = Counter(final_syms)
    print(f"Final composition: {comp}")
    
    # Verify that uniform neutral ligand passivation occurred
    assert "N" in final_syms, "Neutral ligand passivation failed: Nitrogen not found."
    assert "C" in final_syms, "Neutral ligand passivation failed: Carbon not found."
    print("Uniform neutral-ligand passivation test: SUCCESS!")

if __name__ == "__main__":
    test_neutral_ligand_uniform()
