import os
import torch
import pickle
import numpy as np
import pytest
from torch.utils.data import Dataset
from torch_geometric.data import Data
from unittest.mock import MagicMock, patch, mock_open
from data_processing import dataUtils


# ======== load_dataset + get_dataloader ==========

@patch("data_processing.dataUtils._get_dataset")
@patch("data_processing.dataUtils._get_dataloader")
@patch("data_processing.dataUtils.append_record")
def test_load_dataset(mock_log, mock_loader, mock_get_dataset):
    dummy_data = MagicMock()
    dummy_data.num_node_features = 10
    dummy_data.num_classes = 2
    mock_get_dataset.return_value = dummy_data
    mock_loader.return_value = {"train": MagicMock(), "eval": MagicMock(), "test": MagicMock()}

    dataUtils.data_args.dataset_name = "DummySet"
    dataUtils.data_args.dataset_dir = "any_dir"
    dataUtils.data_args.task = None
    dataUtils.train_args.batch_size = 8
    dataUtils.data_args.data_split_ratio = [0.6, 0.2, 0.2]

    dataset, in_dim, out_dim, loader = dataUtils.load_dataset()
    assert in_dim == 10
    assert out_dim == 2
    assert "train" in loader


def test_get_dataloader_random_split():
    dummy_data = [Data(x=torch.rand(3, 2)) for _ in range(10)]
    loader = dataUtils._get_dataloader(dummy_data, batch_size=2, random_split_flag=True,
                                       data_split_ratio=[0.6, 0.2, 0.2])
    assert set(loader.keys()) == {"train", "eval", "test"}


def test_get_dataloader_split_indices():
    dataset = MagicMock()
    dataset.__len__.return_value = 10
    dataset.supplement = {"split_indices": torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])}
    loader = dataUtils._get_dataloader(dataset, batch_size=2, random_split_flag=False)
    assert set(loader.keys()) == {"train", "eval", "test"}


# ========== get_dataset ==========

@patch("data_processing.dataUtils._load_MUTAG")
def test_get_dataset_mutag(mock_mutag):
    mock_mutag.return_value = MagicMock()
    result = dataUtils._get_dataset("path", "MUTAG")
    assert result is not None


@patch("data_processing.dataUtils._load_syn_data")
def test_get_dataset_ba2motifs(mock_loader):
    mock_loader.return_value = MagicMock()
    result = dataUtils._get_dataset("path", "BA_2Motifs")
    assert result is not None


@patch("data_processing.dataUtils.MoleculeNet")
def test_get_dataset_moleculenet(mock_mol):
    mock_data = MagicMock()
    mock_data.data = MagicMock()
    mock_data.data.x = torch.tensor([[1.0]])
    mock_data.data.y = torch.tensor([[1]])
    mock_mol.return_value = mock_data
    mock_mol.names.keys.return_value = ["BBBP"]

    result = dataUtils._get_dataset("path", "BBBP")
    assert result is not None


def test_get_dataset_invalid():
    with pytest.raises(NotImplementedError):
        dataUtils._get_dataset("path", "Unknown")


# ========== read_ba2motif_data ==========

@patch("builtins.open", new_callable=mock_open)
@patch("pickle.load")
def test_read_ba2motif_data(mock_pickle, mock_open_file):
    dense = np.zeros((1, 3, 3))
    feat = np.ones((1, 3, 5))
    labels = np.array([[0, 1]])
    mock_pickle.return_value = (dense, feat, labels)

    result = dataUtils._read_ba2motif_data("folder", "prefix")
    assert isinstance(result[0], Data)


# ========== _load_MUTAG + process ==========

@patch("data_processing.dataUtils.torch.load")
@patch("data_processing.dataUtils.torch.save")
@patch("data_processing.dataUtils.open", new_callable=mock_open)
def test_load_mutag(mock_open_file, mock_save, mock_load):
    mock_load.return_value = (MagicMock(), MagicMock())
    with patch("data_processing.dataUtils._MUTAGDataset.process", autospec=True):
        ds = dataUtils._load_MUTAG("fake", "MUTAG")
        assert isinstance(ds, dataUtils._MUTAGDataset)


# ========== _load_syn_data wrapper ==========

@patch("data_processing.dataUtils._BA2MotifDataset")
def test_load_syn_data_wrapper(mock_cls):
    dummy = MagicMock()
    dummy.num_classes = 3
    mock_cls.return_value = dummy
    ds = dataUtils._load_syn_data("dir", "ba_2motifs")
    assert hasattr(ds, "node_type_dict")


# ========== _load_MolecueNet wrapper ==========

@patch("data_processing.dataUtils.MoleculeNet")
def test_load_molecuenet_wrapper(mock_molnet):
    dummy = MagicMock()
    dummy.data.x = torch.rand(3, 4)
    dummy.data.y = torch.randint(0, 2, (3,))
    mock_molnet.return_value = dummy
    mock_molnet.names.keys.return_value = ["BBBP"]

    ds = dataUtils._load_MolecueNet("dir", "bbbp")
    assert ds.node_color is None



@patch("data_processing.dataUtils.MoleculeNet")
def test_moleculenet_multitask(mock_molnet):
    dummy = MagicMock()
    dummy.data.x = torch.rand(5, 4)
    dummy.data.y = torch.randint(0, 2, (5, 3))
    mock_molnet.return_value = dummy
    mock_molnet.names.keys.return_value = ["BBBP"]

    ds = dataUtils._load_MolecueNet("dir", "bbbp", task="classification")
    assert ds.node_color is None
    assert ds.node_type_dict is None
