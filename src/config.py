from pathlib import Path

MODELS_PATH = (Path(__name__).parent.parent / "pretrained_models").resolve()
STANZA_PATH = MODELS_PATH / 'stanza_rubicdata_tokenizer.pt'

DATA_PATH = (Path(__name__).parent.parent / "data").resolve()

# If MIN_CLASS_SIZE > 0, most statistics will be count on randomly chosen MIN_CLASS_SIZE tokens
MIN_CLASS_SIZE = 0
