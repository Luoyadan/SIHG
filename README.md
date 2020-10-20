Anonymous Submission ID 1326 for WWW 2021
============================================
This is an official PyTorch implementation of paper "Interpretable Signed Link Prediction with Signed Infomax Hyperbolic Graph"


## Contents
* [Requirements](#requirements)
* [Dataset Preparation](#dataset-preparation)
* [Usage](#usage)
* [Logs](#logs)




## Requirements

```
python            3.7.3
texttable         1.5.0
tqdm              4.32.1
numpy             1.15.4
scikit-learn      0.1.2
scipy             1.3.0
sklearn           0.20.0
torch             1.4.0
torch-scatter     2.0.4
torch-sparse      0.6.1
torch-cluster     1.5.4
torch-geometric   1.5.0
torchvision       0.5.0
tensorboardX      1.8
```

## Dataset Preparation
Put the edge source files in ./input


## Usage

To run SIHG model with the default setting:
```
python src/main.py
```


## Logs

We save the embedding and evaluation scores as scalar in ./src/logs
