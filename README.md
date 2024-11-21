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
   - Feedforward NN (TBA)
   - Convolutional NN (TBA)
   - BERT (TBA)

## Installation

# TODO: add guide

```
python -m spacy download ru_core_news_lg
```

## Usage

# TODO: add some examples

## External sources

List of projects and sources used in this repo:
 - Frequency dictionary of the Russian National Corpus. [О. Н. Ляшевская, С. А. Шаров. Частотный словарь современного русского языка (на материалах Национального корпуса русского языка)](http://dict.ruslang.ru/freq.php).
 - [Deep Convolutional Networks for Supervised Morpheme Segmentation of Russian Language](https://github.com/AlexeySorokin/NeuralMorphemeSegmentation)
 - [Tokenizer and morpheme segmentation model from Russian National Corpus](https://ruscorpora.ru/en/license-content/neuromodels)
 - [Stanza](https://stanfordnlp.github.io/stanza/)
 - [Spacy (ru_core_news_lg model)](https://spacy.io/models)

## Citing
If you use this dataset, please cite this paper:

``Text complexity and linguistic features: their correlation in English and Russian - Dmitry A. Morozov , Anna V. Glazkova, and Boris L. Iomdin // Russian Journal of Linguistics 26 (2). 425–447, 2022.``

