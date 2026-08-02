from typing import List, Dict, Tuple, Set
from src.models.observation import IdentityObservation
from src.models.common import SourceType

class IdentityRegistry:
    def __init__(self):
        # Index: (identity_type, normalized_value) -> List[IdentityObservation]
        self._index: Dict[Tuple[str, str], List[IdentityObservation]] = {}
        # Deduplication set
        self._dedup_set: Set[Tuple[str, str, str, str, str]] = set()
        
    def register(self, obs: IdentityObservation):
        key = (obs.identity.identity_type.value, obs.identity.normalized_value)
        
        # Dedup key prevents uncontrolled duplication of the exact same observation occurrence.
        dedup_key = (
            obs.identity.identity_type.value,
            obs.identity.normalized_value,
            obs.source_type.value,
            obs.source_record_id,
            obs.source_field
        )
        
        if dedup_key in self._dedup_set:
            return # skip exact duplicate
            
        self._dedup_set.add(dedup_key)
        
        if key not in self._index:
            self._index[key] = []
            
        self._index[key].append(obs)
        
    def get_observations(self, identity_type: str, normalized_value: str) -> List[IdentityObservation]:
        return self._index.get((identity_type, normalized_value), [])
        
    def has_identity(self, identity_type: str, normalized_value: str) -> bool:
        return (identity_type, normalized_value) in self._index
        
    def get_sources(self, identity_type: str, normalized_value: str) -> Set[SourceType]:
        obs = self.get_observations(identity_type, normalized_value)
        return {o.source_type for o in obs}
        
    def is_cross_source(self, identity_type: str, normalized_value: str) -> bool:
        return len(self.get_sources(identity_type, normalized_value)) > 1
        
    def get_unique_identities(self, identity_type: str) -> int:
        return sum(1 for k in self._index.keys() if k[0] == identity_type)
        
    def get_total_observations(self) -> int:
        return sum(len(obs) for obs in self._index.values())

    def get_all_identities_by_type(self, identity_type: str) -> List[str]:
        return [k[1] for k in self._index.keys() if k[0] == identity_type]
