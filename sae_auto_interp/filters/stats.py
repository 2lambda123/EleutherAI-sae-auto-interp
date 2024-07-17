from typing import Dict, List
from ..features import FeatureRecord

class Stat:

    def refresh(self, **kwargs):
        pass

    def compute(self, records: List[FeatureRecord], *args, **kwargs):
        pass

class CombinedStat(Stat):
    def __init__(self, **kwargs):
        self._objs: Dict[str, Stat] = kwargs

    def refresh(self, **kwargs):
        for obj in self._objs.values():
            obj.refresh(**kwargs)

    def compute(self, records: List[FeatureRecord], *args, **kwargs):
        for obj in self._objs.values():
            obj.compute(records, *args, **kwargs)
