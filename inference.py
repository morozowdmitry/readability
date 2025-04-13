import pandas as pd

from src.pipelines.config import PipelineConfig
from src.pipelines.inference_pipeline import InferencePipeline

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

from src.vectorize.vectorizer import BoWVectorizer
from src.vectorize.sentencebert import SentenceBertVectorizer
from src.vectorize.lda import LDAVectorizer

from src.predict.sklearn_predictors import SVCPredictor, RandomForestPredictor
from src.predict.mlp_predictor import MLPPredictor

from src.config import DATA_PATH, MODELS_PATH

from src.utils.scaler import SKLearnMinMaxScaler
from src.utils.metric import evaluate_metrics
from src.utils.labels import label2class


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
    predictor=RandomForestPredictor(),
    scaler=SKLearnMinMaxScaler(model_path=DATA_PATH / f'models/test_scaler.pt'),
    vectorizer=BoWVectorizer(model_path=DATA_PATH / f'models/test_vectorizer.pt'),
    topic_modeler=LDAVectorizer(model_path=DATA_PATH / f'models/test_topic_modeler.pt'),
)



if __name__ == "__main__":
    df = pd.read_csv(str(DATA_PATH / f'corpora/{args.corpus}/{args.dataset}.csv'))
    targets = [label2class(x) for x in df["category"].tolist()]
    simple_config.model_path = MODELS_PATH / f'models/test_model.pt'
    simple_config.model_path.parent.mkdir(exist_ok=True, parents=True)
    inference_pipeline = InferencePipeline(simple_config)
    predicts, _ = inference_pipeline.run(raw_texts=df['text'].to_list())
    metrics = evaluate_metrics(targets, predicts)
    print(metrics)
