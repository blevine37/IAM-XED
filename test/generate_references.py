# test/generate_references.py
import os
import subprocess
import shutil
from pathlib import Path
import sys

# Import the test cases from the test file
from test_xed import test_xed_output

def get_test_cases():
    """Extract test cases from pytest parametrized function."""
    # This is a bit of a hack to get the test cases from the parametrize decorator
    for marker in test_xed_output.pytestmark:
        if marker.name == 'parametrize' and marker.args[0] == 'test_case':
            return marker.args[1]
    return []

def main():
    """Generate reference files for all test cases."""
    test_cases = get_test_cases()

    for test in test_cases:
        # Remove "test/" prefix if running from test directory
        dir_path = test["dir"]
        if dir_path.startswith("test/"):
            dir_path = dir_path[5:]  # Remove "test/" prefix
        
        test_dir = Path(dir_path)
        print(f"Generating reference for: {dir_path}")

        # Check if directory exists
        if not test_dir.exists():
            print(f"Error: Directory {test_dir} does not exist. Please create it first.")
            continue
            
        output_path = Path(test["output"])
        reference_path = Path(test["reference"])

        # Run XED command
        original_dir = os.getcwd()
        os.chdir(test_dir)

        try:
            cmd = f"iamxed {test['command']}"
            print(f"Running: {cmd}")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"Error running command: {result.stderr}")
                continue

            if not output_path.exists():
                print(f"Output file {output_path} not created")
                continue

            # Copy output to reference
            shutil.copy2(output_path, reference_path)
            print(f"Created reference file: {reference_path}")

        finally:
            os.chdir(original_dir)

if __name__ == "__main__":
    main()