#!/bin/bash

conda create -y -n nmbg python=3.7
conda activate nmbg
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install jupyter
pip install scikit-image matplotlib imageio plotly opencv-python
pip install black usort flake8 flake8-bugbear flake8-comprehensions
conda install pytorch3d -c pytorch3d

pip install \
    numpy \
    pyyaml \
    tensorboardX \
    munch \
    scipy \
    matplotlib \
    Cython \
    trimesh \
    huepy \
    Pillow \
    tqdm \
    scikit-learn

