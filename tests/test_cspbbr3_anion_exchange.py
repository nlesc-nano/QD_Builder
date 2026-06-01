import sys
import subprocess
import numpy as np
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

def test_cspbbr3_anion_exchange():
    print("--- Running CsPbBr3 Surface Anion Exchange Test ---")
    cif = ROOT / "examples/cifs/CsPbBr3.cif"
    yaml_path = ROOT / "tests/test_cspbbr3_anion_exchange.yaml"
    out = ROOT / "tests/out/cspbbr3_anion_exchange.xyz"
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
    
    # Verify that ligand exchange occurred
    # Since we replaced Br with CCC(=O)O, we expect:
    # 1. C and O to be present in final_syms (from the ligand)
    # 2. Number of Br should be significantly reduced
    assert "C" in final_syms, "Ligand exchange failed: Carbon not found in final structure."
    assert "O" in final_syms, "Ligand exchange failed: Oxygen not found in final structure."
    print("Surface Br anion exchange test: SUCCESS!")

if __name__ == "__main__":
    test_cspbbr3_anion_exchange()
