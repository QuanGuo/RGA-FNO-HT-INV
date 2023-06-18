# RGA-FNO-HT-INV
RGA+FNO for inverse modeling of hydraulic tomography 

Author: Quan G

## About this notebook

This notebook combines Reformulated Geostatistical Approach (RGA) and Fourier Neural Operator (FNO) for inverse modeling of hydraulic tomography (HT). The RGA-FNO model uses RGA for dimensionality reduction of Gaussian random field (GRF) and FNO as forward model surrogate, the inverse process optimizes latent variables of random fields to match monitored hydraulic head measurements.

Please note this work:
* Assumes the reader is comfortable with Python, especially, python notebook and pytorch.
* Google Colab is recommended as the programming platform.

## Data
Data used for training DL models in paper experiments is stored at 
[Google Drive (under 'Data' directory)](https://drive.google.com/drive/folders/1lkJAHBljYBIwrukrJyit-iu9_yhlo7Jf).

## Train FNO Forward Model Surrogate
- **Fourier-Neural-Operatorl.ipynb**
    - save the trained model under **models** directory for further use in optimization

## Inverse Modeling with RGA-FNO Model
- **RGA-FNO.ipynb**
    - **RGA** decoder requires the Geostatistical information to be known, i.e., **Z** and **mu**

##
## Citations
```
@article{GUO2023128828,
      title = {Reformulated geostatistical approach with a Fourier neural operator surrogate model for inverse modeling of hydraulic tomography},
      journal = {under review},
}
```
[comment]: <> (      volume = {616},
      pages = {128828},
      year = {2023},
      issn = {0022-1694},
      doi = {https://doi.org/10.1016/j.jhydrol.2022.128828},
      url = {https://www.sciencedirect.com/science/article/pii/S0022169422013981},
      author = {Quan Guo and Yue Zhao and Chunhui Lu and Jian Luo})

