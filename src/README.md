# Re-ProtGNN
A re-implementation of the AAAI 2022 paper:  
**ProtGNN: Towards Self-Explaining Graph Neural Networks**  
[arXiv:2112.00911](https://arxiv.org/abs/2112.00911)

## Requirements
```
pytorch                   1.8.1             
torch-geometric           2.0.2
```
For the full list of dependencies, see [`requirements.txt`](../requirements.txt).

## Included Results

A Jupyter notebook containing pre-run results is available in the `src` directory.


## Usage

1. **Download the required datasets**  
   Download from [this link](https://mailustceducn-my.sharepoint.com/:u:/g/personal/yhy12138_mail_ustc_edu_cn/ET69UPOa9jxAlob03sWzJ50BeXM-lMjoKh52h6aFc8E8Jw?e=lglJcP) and place the contents in the `./data` directory.

2. **Set hyperparameters**  
   Edit `./utils/Configures.py` to customize model and training settings.

3. **Run the model**

```bash
python src/main.py
```

> This implementation has been tested and verified on the **MUTAG** datasets.

## Reference

Original Paper:

```
@article{zhang2021protgnn,
  title={ProtGNN: Towards Self-Explaining Graph Neural Networks},
  author={Zhang, Zaixi and Liu, Qi and Wang, Hao and Lu, Chengqiang and Lee, Cheekong},
  journal={arXiv preprint arXiv:2112.00911},
  year={2021}
}
```

