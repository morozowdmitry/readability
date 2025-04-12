import pandas as pd

from src.pipelines.config import PipelineConfig
from src.pipelines.label_pipeline import LabelPipeline

from src.tokenize.stanza_tokenizer import StanzaTokenizer
from src.preprocess.base_preprocessor import BasePreprocessor

from src.label.labels import LabelType
from src.label.pymorphy import PymorphyLemmatizer
from src.label.frequency import FrequencyLabeler
from src.label.spacy_syntax import SpacySyntaxLabeler
from src.label.morpheme import MorphemeBERTLabeler

from src.feature_extract.variety_extractor import NAVExtractor, LexicalVarietyExtractor
from src.feature_extract.lengths_extractor import LengthsExtractor
from src.feature_extract.indices_extractor import LongWordsExtractor, IndicesExtractor
from src.feature_extract.frequency_extractor import FrequencyExtractor
from src.feature_extract.punctuation_extractor import PunctuationExtractor
from src.feature_extract.morphology_extractor import MorphologyExtractor
from src.feature_extract.syntax_extractor import SimpleSyntaxExtractor
from src.feature_extract.morpheme_extractor import MorphemeVarietyExtractor

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
        (PymorphyLemmatizer(), {LabelType.MORPH, LabelType.LEMMA}),
        (FrequencyLabeler(), {LabelType.FREQUENCY, }),
        (SpacySyntaxLabeler(), {LabelType.SYNTAX, }),
        (MorphemeBERTLabeler(), {LabelType.MORPHEME}),
    ],
    extractors=[
        NAVExtractor(),
        LexicalVarietyExtractor(),
        LengthsExtractor(),
        LongWordsExtractor(),
        IndicesExtractor(),
        FrequencyExtractor(),
        PunctuationExtractor(),
        MorphologyExtractor(),
        SimpleSyntaxExtractor(),
        MorphemeVarietyExtractor(),
    ],
    predictor=BasePredictor()
)


if __name__ == "__main__":
    df = pd.read_csv(str(DATA_PATH / f'corpora/{args.corpus}/{args.dataset}.csv'))
    simple_config.labeled_path = DATA_PATH / f'prelabeled/{args.corpus}/{args.dataset}.csv'
    simple_config.labeled_path.parent.mkdir(exist_ok=True, parents=True)
    label_pipeline = LabelPipeline(simple_config)
    label_pipeline.run(raw_texts=df['text'].to_list())
