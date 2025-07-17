# test/test_xed.py
import os
import subprocess
import numpy as np
import pytest
from pathlib import Path
from test_cases import test_cases, test_ids

def run_command(test_dir, command):
    """Run XED command in the specified test directory."""
    original_dir = os.getcwd()
    os.chdir(test_dir)

    try:
        cmd = f"iamxed {command}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result
    finally:
        os.chdir(original_dir)

def compare_output(output_file, reference_file, rtol=1e-5, atol=1e-8):
    """Compare output file with reference file, ignoring comment lines."""
    if output_file.suffix == '.npz':
        compare_npz(output_file, reference_file, rtol, atol)
    else:
        def load_data_ignoring_comments(path):
            with open(path) as f:
                lines = [line for line in f if not line.strip().startswith("#")]
            from io import StringIO
            return np.loadtxt(StringIO("".join(lines)))
        output_data = load_data_ignoring_comments(output_file)
        reference_data = load_data_ignoring_comments(reference_file)
        np.testing.assert_allclose(output_data, reference_data, rtol=rtol, atol=atol)

def compare_npz(output_file, reference_file, rtol=1e-5, atol=1e-8):
    """Compare NPZ files."""
    with np.load(output_file) as output_data, np.load(reference_file) as reference_data:
        output_keys = set(output_data.keys())
        reference_keys = set(reference_data.keys())
        
        # Ignore metadata
        output_keys.discard('metadata')
        reference_keys.discard('metadata')
        
        assert output_keys == reference_keys, "Different keys in NPZ files"
        
        for key in output_keys:
            np.testing.assert_allclose(output_data[key], reference_data[key], 
                                      rtol=rtol, atol=atol,
                                      err_msg=f"Arrays differ for key '{key}'")

@pytest.mark.parametrize("test_case", test_cases, ids=test_ids)
def test_xed_output(test_case):
    """Test XED output against reference files."""
    test_dir = Path(test_case["dir"])
    output_path = test_dir / test_case["output"]
    reference_path = test_dir / test_case["reference"]
    
    # Remove output file if it exists
    if output_path.exists():
        output_path.unlink()
    
    # Run the command
    result = run_command(test_dir, test_case["command"])
    assert result.returncode == 0, f"Command failed: {result.stderr}"
    
    # Check output file was created
    assert output_path.exists(), f"Output file {output_path} was not created"
    assert reference_path.exists(), f"Reference file {reference_path} does not exist"
    
    # Compare output with reference
    compare_output(output_path, reference_path)
