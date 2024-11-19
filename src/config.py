from pathlib import Path

RESOURCES_DIR = (Path(__name__).parent.parent / "external_resources").resolve()
STANZA_PATH = RESOURCES_DIR / 'stanza_rubicdata_tokenizer.pt'
FREQUENCY_RUS_PATH = RESOURCES_DIR / 'frequency/rus/freqrnc2011.tsv'
FREQUENCY_ENG_PATH = RESOURCES_DIR / 'frequency/eng/all_words.txt'
MORPHODICT_PATH = RESOURCES_DIR / 'morphemes/lex2morphemes.json'
MORPHEME_CONFIG_PATH = RESOURCES_DIR / 'morphemes/morphodict_10_07_2023.json'

DATA_PATH = (Path(__name__).parent.parent / "data").resolve()

# If MIN_CLASS_SIZE > 0, most statistics will be count on randomly chosen MIN_CLASS_SIZE tokens
MIN_CLASS_SIZE = 0
