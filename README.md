# KDD2022CLARE


This is our implementation for the paper:

**CLARE: A Semi-supervised Community Detection Algorithm**




## What are in this Repository

This repository contains the following contents:

```
.
├── Locator                       --> (The folder containing Community Locator source code)
├── Rewriter                      --> (The folder containing Community Rewriter source code)
├── ckpts                         --> (The folder saving checkpoint files)
├── dataset                       --> (The folder containing 7 used datasets)
├── run.py                        --> (The main code file. The code is run through this file)
└── utils                         --> (The folder containing utils functions)

```
Note that you have to create a `ckpts` folder to save contents.



## Datasets

Raw datasets are available at SNAP(http://snap.stanford.edu/data/index.html) and pre-processing details are explained in our paper.

> We select LiveJournal, DBLP and Amazon, in the **Networks with ground-truth communities** part.



We provide 7 datasets as below. Each of them contains a community file `{name}-1.90.cmty.txt` and an edge file `{name}-1.90.ungraph.txt`.

```
├── dataset
│   ├── amazon
│   ├── amazon_dblp
│   ├── dblp
│   ├── dblp_amazon
│   ├── dblp_lj
│   ├── lj
│   ├── lj_dblp
```





## Run our code

### Environmental Requirement

0. You need to set up the environment for running the experiments (Python 3.7 or above)

1. Install **Pytorch** with version 1.8.0 or later

2.  Install **torch-geometric** package with version 2.0.1

    Note that it may need to appropriately install the package `torch-geometric` based on the CUDA version (or CPU version if GPU is not available). Please refer to the official website https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html for more information of installing prerequisites.

    For example (Mac / CPU)

    ```
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.0+cpu.html
    ```

3. Install **Deepsnap** with version 0.2.1

   ```
   pip install deepsnap
   ```



### Run the code

Execute the `run.py` file

```
python run.py --dataset=amazon --conv_type=GCN 
```

Main arguments:

```
--dataset [amazon, dblp, lj, amazon_dblp, dblp_amazon, dblp_lj, lj_dblp]: the dataset to run
--conv_type [GCN, GIN, SAGE]: GNN type in Community Locator
--n_layers: ego-net dimensions & number of GNN layers
--pred_size: total number of predicted communities
--agent_lr: the learning rate of Community Rewriter
```

  For more argument options, please refer to `run.py`
