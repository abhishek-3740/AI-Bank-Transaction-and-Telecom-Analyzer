from typing import List, Dict
from src.models.bank import BankTransaction
from src.models.cdr import CDREvent
from src.resolution.registry import IdentityRegistry
from src.models.common import IdentityType, SourceType
from src.correlation.config import CorrelationConfig
from src.correlation.models import (
    CorrelationRecord,
    MatchStrength,
    RelationshipType,
    IdentityMatchEvidence
)

def correlate_bank_to_cdr(
    bank_txns: List[BankTransaction],
    cdr_dict: Dict[str, CDREvent],
    registry: IdentityRegistry,
    config: CorrelationConfig
) -> List[CorrelationRecord]:
    
    correlations = []
    
    for txn in bank_txns:
        # 1. Candidate Generation
        # Aggregate candidates by CDR ID to deduplicate multiple paths (e.g. sender and receiver both match)
        candidates: Dict[str, List[IdentityMatchEvidence]] = {}
        
        # Helper to search
        def search_phone(phone_val: str, bank_role: str):
            if not phone_val:
                return
            # Normalize just in case, though registry values should be canonical
            obs_list = registry.get_observations(IdentityType.PHONE.value, phone_val)
            for obs in obs_list:
                if obs.source_type == SourceType.CDR:
                    # Pre-filter by time!
                    time_diff = (obs.timestamp - txn.transaction_timestamp).total_seconds()
                    if abs(time_diff) > config.bank_cdr_window_seconds:
                        continue
                        
                    evidence = IdentityMatchEvidence(
                        identity_type=IdentityType.PHONE,
                        normalized_value=phone_val,
                        source_role=bank_role,
                        target_role=obs.role.value
                    )
                    
                    if obs.source_record_id not in candidates:
                        candidates[obs.source_record_id] = []
                    candidates[obs.source_record_id].append(evidence)

        if txn.sender.phone:
            search_phone(txn.sender.phone, "BANK_SENDER")
        if txn.receiver.phone:
            search_phone(txn.receiver.phone, "BANK_RECEIVER")
            
        # 2. Evaluation
        for cdr_id, evidence_list in candidates.items():
            if cdr_id not in cdr_dict:
                continue
                
            cdr_event = cdr_dict[cdr_id]
            
            # target - source
            time_diff = (cdr_event.event_timestamp - txn.transaction_timestamp).total_seconds()
            
            accepted = abs(time_diff) <= config.bank_cdr_window_seconds
            if not accepted:
                continue
                
            strength = MatchStrength.STRONG
            correlation_id = f"{RelationshipType.BANK_CDR.value}_{txn.transaction_id}_{cdr_id}"
            
            record = CorrelationRecord(
                correlation_id=correlation_id,
                relationship_type=RelationshipType.BANK_CDR,
                source_type=SourceType.BANK,
                source_event_id=txn.transaction_id,
                source_timestamp=txn.transaction_timestamp,
                target_type=SourceType.CDR,
                target_event_id=cdr_id,
                target_timestamp=cdr_event.event_timestamp,
                time_difference_seconds=int(time_diff),
                identity_evidence=evidence_list,
                conflicting_evidence=[],
                match_strength=strength,
                accepted=accepted
            )
            correlations.append(record)
            
    # Sort for deterministic output
    correlations.sort(key=lambda x: x.correlation_id)
    return correlations
