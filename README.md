# SAND
The official PyTorch implementation of "Learning to Simulate Daily Activities via Modeling Dynamic Human Needs" (WWW'23)

The code is tested under a Linux desktop with torch 1.7 and Python 3.7.10.

## Installation

### Environment
- Tested OS: Linux
- Python >= 3.7
- PyTorch == 1.7.1

### Dependencies
1. Install PyTorch 1.7.1 with the correct CUDA version.
2. Use the ``pip install -r requirements. txt`` command to install all of the Python modules and packages used in this project.

## Model Training

Use the following command to train DSTPP on `Foursquare` dataset: 

``
cd SAND
``

``
python app.py --dataset 'Foursquare'
``

## Note

The implemention is based on *[NJSDE](https://github.com/000Justin000/torchdiffeq/tree/jj585)*.
