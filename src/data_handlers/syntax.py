from typing import Optional, List


class SyntaxParsing:
    def __init__(self,
                 dep: Optional[str] = None,
                 idx: Optional[int] = None,
                 head_idx: Optional[int] = None,
                 children_idx: Optional[List[int]] = None,
                 is_root: Optional[bool] = False):
        self.dep = dep
        self.idx = idx
        self.head_idx = head_idx
        self.children_idx = children_idx
        self.is_root = is_root
