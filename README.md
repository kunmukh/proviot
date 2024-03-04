# ProvIoT: Detecting Stealthy Attacks in IoT through Federated Edge-Cloud Security

Reproducibility artifacts for the paper _ProvIoT: Detecting Stealthy Attacks in IoT through Federated Edge-Cloud Security_.


## Folder structure

| Folder | Description|
| -------|-----------|
| `data`| Folder containing the data files for IDS execution. |
| `dco2vec`| Folder containing the code for doc2vec implementation. |
| `AE`| Folder containing the code for AutoEncoder (AE) execution. |

### Environment Setup

We will use `conda` as the python environment manager. Install the project dependencies from the [proviot.yml](proviot.yml) using this command:

```bash
conda env update --name proviot --file proviot.yml
```

Activate the conda environment before running the experiments by running this command

```bash
conda activate proviot
```

### ProvIot

* [provIoT.py](AE/provIoT.py)
  * Driver script for ProvIoT, which is an Autoencoder-based IDS that detects anomalous paths.
  * Sample causal paragraphs and feature vectors for APT attack available in [anomaly-paragraph](data/example-paragraph/anomaly-paragraph.csv) directory.
  
Running the ProvIoT script:

```bash
python proviot.py
```

## Citing us

```
@inproceedings{mukherjee2024acns,
	title        = {ProvIoT: Detecting Stealthy Attacks in IoT through Federated Edge-Cloud Security},
	author       = {Kunal Mukherjee and Joshua Wiedemeier and Qi Wang and Junpei Kamimura and John Junghwan Rhee and James Wei and Zhichun Li and Xiao Yu and Lu-An Tang and Jiaping Gui and Kangkook Jee},
	year         = 2024,
	booktitle    = {22nd International Conference on Applied Cryptography and Network Security},
	series       = {ACNS '24}
}
```
