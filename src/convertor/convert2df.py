from typing import List, Optional
import pandas as pd
import numpy as np

from src.data_handlers.text import Text

from tqdm import tqdm


def texts2df(texts: List[Text], prelabeled_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    concat_list = list()
    print('start converting to DF')
    if texts:
        if texts[0].embedding is not None:
            columns = list(texts[0].embedding.keys())
            data = list()
            for _t in tqdm(texts):
                _t_row = [_t.embedding[column_name] for column_name in columns]
                data.append(_t_row)
            embedding_df = pd.DataFrame(np.array(data), columns=columns)
            concat_list.append(embedding_df)
            print('embeddings collected')
        if texts[0].topic is not None:
            topic_df = pd.DataFrame([_t.topic for _t in texts],
                                    columns=[f'topic_{i}' for i in range(len(texts[0].topic))])
            concat_list.append(topic_df)
            print('topics collected')
        if texts[0].features is not None:
            features_df = pd.DataFrame([_t.features for _t in texts])
            concat_list.append(features_df)
            print('new features collected')
    if prelabeled_df is not None:
        concat_list.append(prelabeled_df)
        print('prelabeled features collected')
    result_df = pd.concat(concat_list, axis=1)
    result_df = result_df.reindex(sorted(result_df.columns), axis=1)
    return result_df


def labels2series(labels: List[str]) -> pd.Series:
    return pd.Series(labels)
