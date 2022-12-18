#!/bin/bash

# conda create -n nlp python=3.9
# conda activate nlp
conda install pip
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchtext==0.8.1
pip install ipdb
pip install weightwatcher
pip install tqdm
pip install wandb
pip install opencv-python
pip install scikit-image
pip install pandas
pip install tensorboard
pip install spacy
python -m spacy download en_core_web_sm
pip install torchaudio==0.9.1
pip install torchvision==0.10.1
pip install tornado==6.2
pip install numpy