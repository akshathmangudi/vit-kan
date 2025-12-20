# ViKANformer: Embedding Kolmogorov–Arnold Networks in Vision Transformers for Pattern-based Learning

ViKANformer replaces standard MLP blocks in Vision Transformers with Kolmogorov-Arnold expansions (SineKAN, Fast-KAN, etc.), leveraging the Kolmogorov-Arnold theorem for enhanced representational power. For full details on the methodology and experiments, please refer to the [arXiv paper](https://arxiv.org/abs/2503.01124).

## Installation

### Step 1: Setup a virtualenv / conda environment. 
For conda: 
```bash
$ conda create -n <env_name> 
```

For virtualenv:
```bash
$ python -m venv <env_name>
```
where `env_name` will be the name of the directory that you will use as the virtual environment. 

### Step 2: Activate the environment. 
For conda: 
```bash
$ conda install --file requirements.txt
```

For virualenv: 
```bash
$ pip install -r requirements.txt
```

### Step 3: Run train.py 
```bash
$ python train.py
```

To follow these steps correctly, make sure you are in the root directory of the repository. 


Control variables:
- **Dataset used: MNIST**
- **Transformations: None**
- **GPU Used: Tesla P100**

## Citation

If you find ViKANformer helpful in your research, please cite our paper:

```bibtex
@misc{s2025vikanformerembeddingkolmogorovarnold,
      title={ViKANformer: Embedding Kolmogorov Arnold Networks in Vision Transformers for Pattern-Based Learning}, 
      author={Shreyas S and Akshath M},
      year={2025},
      eprint={2503.01124},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.01124}, 
}
```

## License
We give credit to all the other papers and projects that we have referenced in order to write the paper. This paper is covered by the MIT License, granting full access to using this repository for any use whatsoever. 
