import pandas as pd

from src.pipelines.config import PipelineConfig
from src.pipelines.tokenize_pipeline import TokenizePipeline

from src.tokenize.stanza_tokenizer import StanzaTokenizer
from src.preprocess.base_preprocessor import BasePreprocessor

from src.label.labels import LabelType
from src.label.pymorphy import PymorphyLemmatizer

from src.predict.base_predictor import BasePredictor

from src.config import DATA_PATH


import argparse

parser = argparse.ArgumentParser(description='Experiment configuration')
parser.add_argument('--corpus', type=str, help='corpus')
parser.add_argument('--dataset', type=str, help='dataset')
args = parser.parse_args()


simple_config = PipelineConfig(
    preprocessor=BasePreprocessor(tokenizer=StanzaTokenizer()),
    labelers=[
        (PymorphyLemmatizer(), {LabelType.LEMMA}),
    ],
    extractors=[],
    predictor=BasePredictor()
)


if __name__ == "__main__":
    df = pd.read_csv(str(DATA_PATH / f'corpora/{args.corpus}/{args.dataset}.csv'))
    simple_config.labeled_path = DATA_PATH / f'pretokenized/{args.corpus}/{args.dataset}.csv'
    simple_config.labeled_path.parent.mkdir(exist_ok=True, parents=True)
    tokenize_pipeline = TokenizePipeline(simple_config)
    tokenize_pipeline.run(raw_texts=df['text'].to_list())
