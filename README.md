# ProvIoT: Detecting Stealthy Attacks in IoT through Federated Edge-Cloud Security

Reproducibility artifacts for the paper _ProvIoT: Detecting Stealthy Attacks in IoT through Federated Edge-Cloud Security_.

## Folder structure

| Folder | Description|
| -------|-----------|
| `data`| Folder containing the data files for IDS execution. |
| `dco2vec`| Folder containing the code for doc2vec implementation. |
| `AE`| Folder containing the code for AutoEncoder (AE) execution. |


### ProvIot

* [proviot.py](AE/provIoT.py)
  * Driver script for ProvIoT, which is an Autoencoder-based IDS that detects anomalous paths.
  * Sample causal paragraphs and feature vectors for APT attack using _EXCEL.EXE_ available in [anomaly-paragraph](data/example-paragraph/anomaly-paragraph.csv) directory.
  
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
