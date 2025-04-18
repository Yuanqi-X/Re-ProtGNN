import os
import pytest
from data_processing import dataUtils
from torch_geometric.data import InMemoryDataset

# T1: Test that missing raw file raises FileNotFoundError
def test_missing_raw_file_raises():
    broken_root = "./tests/fake_data/mutag_missing_file/"
    dataset_name = "MUTAG"
    dataset_path = os.path.join(broken_root, dataset_name, "raw")

    # Create directory structure without actual files
    os.makedirs(dataset_path, exist_ok=True)

    # Expect a FileNotFoundError because required files are missing
    with pytest.raises(FileNotFoundError) as excinfo:
        _ = dataUtils._MUTAGDataset(root=broken_root, name=dataset_name)

    assert "is missing" in str(excinfo.value)


# T2: Test that corrupted input format raises ValueError
def test_invalid_format_raises(tmp_path):
    # Create a directory with corrupted raw files
    raw_dir = tmp_path / "MUTAG" / "raw"
    raw_dir.mkdir(parents=True)

    # Required file names (must match .process())
    required_files = [
        "MUTAG_A.txt",
        "MUTAG_node_labels.txt",
        "MUTAG_graph_labels.txt",
        "MUTAG_graph_indicator.txt"
    ]

    # Write invalid content to simulate a format error
    for fname in required_files:
        with open(raw_dir / fname, "w") as f:
            f.write("corrupted_content\n")

    # Expect a ValueError due to parsing failure
    with pytest.raises(ValueError) as excinfo:
        _ = dataUtils._MUTAGDataset(root=str(tmp_path), name="MUTAG")

    assert "Failed to parse" in str(excinfo.value) or "Mismatch" in str(excinfo.value)


# T3: Success case with real MUTAG data
def test_valid_dataset_loads_successfully():
    dataset = dataUtils._MUTAGDataset(root="data", name="MUTAG")
    assert isinstance(dataset, InMemoryDataset)
    assert len(dataset) > 0



"""
In terminal (project's root), run tests with:

PYTHONPATH=./src pytest tests/sys_test_input_validation.py
"""