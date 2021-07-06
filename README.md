# Learning to Follow Instruction in Text-Based Games
This is the code repository for the paper Learning to Follow Instruction in Text-Based Games, which will appear in NeurIPS 2022.

## Installing Dependencies
We used `python-3.9` in our experiments.

The main dependencies can be installed as
`pip install -r requirements.txt`

We use the Spot library for LTL progression, which can be installed by following the instructions [here](https://spot.lrde.epita.fr/install.html).
It should be installed to `<path-to-python-env>/lib/python??/site-packages/spot`

Once that's complete, we can download spacy
```console
python -m spacy download en
```
and also download the FastText Word Embeddings
```console
cd src/support-files
curl -L -o crawl-300d-2M.vec.h5 "https://bit.ly/2U3Mde2"
```
The dataset itself can be downloaded as
```console
cd src/support-files/rl.0.2
wget https://aka.ms/twkg/rl.0.2.zip
unzip rl.0.2.zip
```
Note finally that vocabularies can be found in [src/support-files/vocabularies](support-files/vocabularies).
The location of the dataset and vocabularies is not important, and can be specified in the `config.yaml` file.

## Training Agents
Instructions for training and evaluating agents on is available in the [src](src) folder.

## Citation


