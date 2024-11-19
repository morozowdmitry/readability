from typing import List, Dict

import statistics
from collections import Counter, defaultdict

from src.data_handlers.text import Text, Token, Sentence, TokenType
from src.feature_extract.base_extractor import BaseExtractor
from src.label.labels import LabelType


class BaseSyntaxExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        self.required_labels = {LabelType.SYNTAX}

    def _count_dep2sent(self, text: Text, deps: List[str], label: str) -> Dict[str, float]:
        sent_dep_num = [
            len([
                _t for _t in _s.tokens
                if _t.syntax.dep in deps
            ])
            for _s in text.sentences
        ]
        return self._stats(sent_dep_num, label)

    def _max_linear_distance(self, text: Text) -> Dict[str, float]:
        max_distances = list()
        for sent in text.sentences:
            max_distance = max(abs(token.syntax.idx - token.syntax.head_idx) for token in sent.tokens)
            max_distances.append(max_distance)
        return self._stats(max_distances, 'max_distance')

    def _mean_linear_distance(self, text: Text) -> Dict[str, float]:
        mean_distances = list()
        for sent in text.sentences:
            mean_distance = statistics.mean(abs(token.syntax.idx - token.syntax.head_idx) for token in sent.tokens)
            mean_distances.append(mean_distance)
        return self._stats(mean_distances, 'mean_distance')

    def _tree_depth(self, text: Text) -> Dict[str, float]:
        tree_depths = list()
        for sent in text.sentences:
            root_token = [token for token in sent.tokens if token.syntax.is_root][0]
            tree_depths.append(self._tree_step(sentence=sent, token=root_token, current_depth=0))
        return self._stats(tree_depths, 'tree_depth')

    def _tree_step(self, sentence: Sentence, token: Token, current_depth: int) -> int:
        if token.syntax.children_idx:
            return max(
                self._tree_step(sentence=sentence, token=sentence.tokens[idx], current_depth=current_depth)
                for idx in token.syntax.children_idx
            ) + 1
        else:
            return current_depth + 1


class SimpleSyntaxExtractor(BaseSyntaxExtractor):
    def __init__(self):
        super().__init__()
        self.extractor_label = 'SIMPLE_SYNTAX'

    def _count_features(self, text: Text) -> Text:
        features = dict()
        features.update(self._count_dep2sent(text, deps=['nmod', 'nmod:poss'], label='nmod'))
        features.update(self._count_dep2sent(text, deps=['ccomp'], label='ccomp'))
        features.update(self._count_dep2sent(text, deps=['xcomp'], label='xcomp'))
        features.update(self._count_dep2sent(text, deps=['acl'], label='acl'))
        features.update(self._count_dep2sent(text, deps=['advcl'], label='advcl'))
        features.update(self._max_linear_distance(text))
        features.update(self._tree_depth(text))
        features.update(self._count_clauses(text))
        features.update(self._nmod_depth(text))
        for k, v in features.items():
            self._add_feature(text, k, v)
        return text

    def _count_clauses(self, text: Text) -> Dict[str, float]:
        sent_clauses_num = [
            len([
                _t for _t in _s.tokens
                if _t.syntax.dep == 'conj' and _s.tokens[_t.syntax.head_idx].syntax.is_root
            ]) + 1
            for _s in text.sentences
        ]
        return self._stats(sent_clauses_num, 'clauses')

    def _nmod_depth(self, text: Text) -> Dict[str, float]:
        nmod_depths = list()
        for sent in text.sentences:
            for token in sent.tokens:
                if token.syntax.dep in ["nmod", "nmod:poss"]:
                    if any(sent.tokens[idx].syntax.dep in ["nmod", "nmod:poss"] for idx in token.syntax.children_idx):
                        continue
                    nmod_depths.append(self._nmod_step(sentence=sent, token=token, current_depth=0))
        return self._stats(nmod_depths, 'nmod_depth')

    def _nmod_step(self, sentence: Sentence, token: Token, current_depth: int) -> int:
        if sentence.tokens[token.syntax.head_idx].syntax.dep in ["nmod", "nmod:poss"]:
            return self._nmod_step(sentence=sentence,
                                   token=sentence.tokens[token.syntax.head_idx],
                                   current_depth=current_depth + 1)
        else:
            return current_depth + 1


class RichSyntaxExtractor(BaseSyntaxExtractor):
    def __init__(self):
        super().__init__()
        self.extractor_label = 'RICH_SYNTAX'

    def _count_features(self, text: Text) -> Text:
        features = self._evaluate_features(text)
        for k, v in features.items():
            self._add_feature(text, k, v)
        return text

    def _evaluate_features(self, text: Text) -> Dict[str, int | float]:
        features = dict()
        features.update(self._count_edge_types(text))
        features.update(self._count_children(text))
        features.update(self._max_linear_distance(text))
        features.update(self._mean_linear_distance(text))
        features.update(self._tree_depth(text))
        return features

    def _count_edge_types(self, text: Text) -> Dict[str, int | float]:
        edge_types_features = dict()
        edge_types = [
            'acl', 'acl:relcl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'aux:pass',
            'case', 'cc', 'ccomp', 'compound', 'conj', 'cop', 'csubj', 'csubj:pass',
            'det', 'discourse', 'expl', 'fixed', 'flat', 'flat:foreign', 'flat:name', 'iobj',
            'mark', 'nmod', 'nsubj', 'nsubj:pass', 'nummod', 'nummod:gov', 'nummod:entity',
            'obj', 'obl', 'orphan', 'parataxis', 'xcomp',
        ]
        sent_stats = defaultdict(list)
        for sent in text.sentences:
            sent_edges = Counter()
            for token in sent.tokens:
                sent_edges[token.syntax.dep] += 1
            for k, v in sent_edges.items():
                sent_stats[k].append(v)
        for k in edge_types:
            v = sent_stats[k]
            if len(v) < text.sentences_number():
                v += [0] * (text.sentences_number() - len(v))
            edge_types_features.update(self._stats(v, label=f'edge_{k}'))
        return edge_types_features

    def _count_vertices(self, text: Text) -> Dict[str, int | float]:
        vertices_number = list()
        for sent in text.sentences:
            vertices_number += sent.words_number()
        return self._stats(vertices_number, 'vertices')

    def _count_children(self, text: Text) -> Dict[str, int | float]:
        children_features = dict()
        sent_stats = defaultdict(list)
        for sent in text.sentences:
            sent_children = Counter()
            for token in sent.tokens:
                children_num = len(token.syntax.children_idx)
                if children_num < 4:
                    sent_children[children_num] += 1
                else:
                    sent_children['>=4'] += 1
                if children_num > 0:
                    sent_children['>0'] += 1
            for k, v in sent_children.items():
                sent_stats[k].append(v)
        for k, v in sent_stats.items():
            if len(v) < text.sentences_number():
                v += [0] * (text.sentences_number() - len(v))
            children_features.update(self._stats(v, label=f'children_{k}'))
        return children_features

    @staticmethod
    def _stats(accumulated, label, sampling=True) -> Dict[str, int | float]:
        return {
            f'average_{label}': statistics.mean(accumulated) if len(accumulated) >= 1 else 0,
            f'median_{label}': statistics.median(accumulated) if len(accumulated) >= 1 else 0,
            f'max_{label}': max(accumulated) if len(accumulated) >= 1 else 0,
            f'std_{label}': statistics.stdev(accumulated) if len(accumulated) >= 2 else 0
        }


class BestSyntaxExtractor(RichSyntaxExtractor):
    def __init__(self):
        super().__init__()
        self.extractor_label = 'BEST_SYNTAX'
        self.features_set = {
            'average_children_2',
            'average_edge_amod',
            'average_edge_compound',
            'average_edge_conj',
            'average_edge_det',
            'average_edge_nmod',
            'average_edge_nsubj',
            'average_children_>0',
            'average_vertices',
            'average_tree_depth',
            'max_children_0',
            'max_children_>0',
            'max_vertices',
            'median_edge_nmod',
            'median_children_>0',
            'median_vertices',
            'std_edge_conj',
            'std_edge_det',
            'std_children_0',
            'std_mean_distance',
            'std_vertices'
        }

    def _count_features(self, text: Text) -> Text:
        features = self._evaluate_features(text)
        for k, v in features.items():
            if k in self.features_set:
                self._add_feature(text, k, v)
        return text

