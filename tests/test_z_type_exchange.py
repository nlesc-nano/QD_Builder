import sys
import subprocess
import numpy as np
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

def test_z_type_exchange():
    print("--- Running CsPbBr3 Z-Type Exchange Test ---")
    cif = ROOT / "examples/cifs/CsPbBr3.cif"
    yaml_path = ROOT / "tests/test_z_type_exchange.yaml"
    out = ROOT / "tests/out/z_type_exchange.xyz"
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
    
    # Verify MXn exchange: InBr3 should place In while preserving Br chemistry.
    assert "In" in final_syms, "MXn exchange failed: In not found."
    
    # Verify zwitterion exchange: the long amino-carboxylate head group has N.
    assert "N" in final_syms, "Zwitterion exchange failed: Nitrogen not found."
    
    # Verify L-type exchange: pentanoic acid should be placed, bringing C and O.
    assert "C" in final_syms, "Carbon not found."
    assert "O" in final_syms, "Oxygen not found."
    
    print("Z-type displacement/exchange test: SUCCESS!")

if __name__ == "__main__":
    test_z_type_exchange()
