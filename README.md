# Readability: extracting linguistic features and predict text complexity 

This repo contains code for evaluating linguistic features and readability levels in Russian (English will be added).

## Available features

1. Linguistic features evaluation:
   - traditional features and readability indices
   - morphological
   - syntax
   - lexical
   - punctuation
   - topic modelling 
   - key phrase generation (TBA)

2. Readability assessment models:
   - SVM
   - RandomForest
   - CatBoost (TBA)
   - Feedforward NN
   - Convolutional NN (TBA)
   - BERT (TBA)

## Installation

The library has been tested on Python 3.10. When using newer versions of Python, you may need to replace pymorphy2 with [pymorphy3](https://github.com/no-plagiarism/pymorphy3), as the original library is no longer supported and is outdated.

To install, you need to:
1. Install the libraries listed in `requirements.txt`
2. Load the spacy model with the command
```
python -m spacy download ru_core_news_lg
```
3. Download morpheme model [Morphberta-K from the Russian National Corpus](https://ruscorpora.ru/license-content/neuromodels#section-15) and place model files in `external_resources/morphemes/morphberta-k` folder for morpheme labeling.

## Usage

The root of the library contains examples of its use: data labeling (`label.py`), model training (`train.py`), running on test data and quality assessment (`inference.py`).

## External sources

List of projects and sources used in this repo:
 - Frequency dictionary of the Russian National Corpus. [О. Н. Ляшевская, С. А. Шаров. Частотный словарь современного русского языка (на материалах Национального корпуса русского языка)](http://dict.ruslang.ru/freq.php).
 - [Tokenizer and morpheme segmentation model from Russian National Corpus](https://ruscorpora.ru/en/license-content/neuromodels)
 - [Stanza](https://stanfordnlp.github.io/stanza/)
 - [Spacy (ru_core_news_lg model)](https://spacy.io/models)

## Citing
If you use this dataset, please cite this paper:

``Text complexity and linguistic features: their correlation in English and Russian - Dmitry A. Morozov , Anna V. Glazkova, and Boris L. Iomdin // Russian Journal of Linguistics 26 (2). 425–447, 2022.``

