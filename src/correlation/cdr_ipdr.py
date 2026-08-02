from typing import List, Dict, Set
from src.models.cdr import CDREvent
from src.models.ipdr import IPDRSession
from src.resolution.registry import IdentityRegistry
from src.models.common import IdentityType, SourceType
from src.correlation.config import CorrelationConfig
from src.correlation.models import (
    CorrelationRecord,
    MatchStrength,
    RelationshipType,
    IdentityMatchEvidence,
    IdentityConflictEvidence
)

def correlate_cdr_to_ipdr(
    cdr_events: List[CDREvent],
    ipdr_dict: Dict[str, IPDRSession],
    registry: IdentityRegistry,
    config: CorrelationConfig
) -> List[CorrelationRecord]:
    
    correlations = []
    
    for cdr in cdr_events:
        # 1. Candidate Generation
        # Map: IPDR_ID -> list of IdentityMatchEvidence
        candidates: Dict[str, List[IdentityMatchEvidence]] = {}
        
        def search_identity(id_type: IdentityType, id_val: str, cdr_role: str):
            if not id_val:
                return
            obs_list = registry.get_observations(id_type.value, id_val)
            for obs in obs_list:
                if obs.source_type == SourceType.IPDR:
                    # Pre-filter by time!
                    time_diff = (obs.timestamp - cdr.event_timestamp).total_seconds()
                    if abs(time_diff) > config.cdr_ipdr_window_seconds:
                        continue
                        
                    evidence = IdentityMatchEvidence(
                        identity_type=id_type,
                        normalized_value=id_val,
                        source_role=cdr_role,
                        target_role=obs.role.value
                    )
                    if obs.source_record_id not in candidates:
                        candidates[obs.source_record_id] = []
                    candidates[obs.source_record_id].append(evidence)

        # Retrieve evidence from registry matches
        search_identity(IdentityType.PHONE, cdr.a_party_phone, "CDR_A_PARTY")
        search_identity(IdentityType.PHONE, cdr.b_party_phone, "CDR_B_PARTY")
        search_identity(IdentityType.IMSI, cdr.imsi, "CDR_IMSI")
        search_identity(IdentityType.IMEI, cdr.imei, "CDR_IMEI")
        search_identity(IdentityType.CELL_ID, cdr.cell_id, "CDR_CELL_ID")
        
        # 2. Evaluation
        for ipdr_id, evidence_list in candidates.items():
            if ipdr_id not in ipdr_dict:
                continue
                
            ipdr = ipdr_dict[ipdr_id]
            time_diff = (ipdr.session_timestamp - cdr.event_timestamp).total_seconds()
            
            # Detect conflicts
            conflicts: List[IdentityConflictEvidence] = []
            
            # Helper for strict conflict checking
            def check_conflict(id_type: IdentityType, cdr_val: str, ipdr_val: str, cdr_role: str, ipdr_role: str):
                # A conflict only exists if BOTH are present and they DO NOT match.
                # If they match, the registry will have provided the evidence.
                # If one is missing, it's just missing evidence.
                if cdr_val and ipdr_val and cdr_val != ipdr_val:
                    conflicts.append(IdentityConflictEvidence(
                        identity_type=id_type,
                        source_value=cdr_val,
                        target_value=ipdr_val,
                        source_role=cdr_role,
                        target_role=ipdr_role
                    ))
            
            # Check IMSI conflict
            check_conflict(IdentityType.IMSI, cdr.imsi, ipdr.subscriber_imsi, "CDR_IMSI", "IPDR_SUBSCRIBER_IMSI")
            # Check IMEI conflict
            check_conflict(IdentityType.IMEI, cdr.imei, ipdr.device_imei, "CDR_IMEI", "IPDR_DEVICE_IMEI")
            # Check Cell ID conflict
            check_conflict(IdentityType.CELL_ID, cdr.cell_id, ipdr.cell_id, "CDR_CELL_ID", "IPDR_CELL_ID")
            # We don't check Phone conflict strictly here because CDR has two phones (A/B party) 
            # while IPDR has one (subscriber). If the IPDR phone matches NEITHER A nor B party,
            # that's a conflict.
            if ipdr.subscriber_msisdn and (cdr.a_party_phone or cdr.b_party_phone):
                if ipdr.subscriber_msisdn != cdr.a_party_phone and ipdr.subscriber_msisdn != cdr.b_party_phone:
                    conflicts.append(IdentityConflictEvidence(
                        identity_type=IdentityType.PHONE,
                        source_value=f"A:{cdr.a_party_phone}|B:{cdr.b_party_phone}",
                        target_value=ipdr.subscriber_msisdn,
                        source_role="CDR_A_B_PARTY",
                        target_role="IPDR_SUBSCRIBER_MSISDN"
                    ))
            
            # 3. Acceptance Policy
            in_window = abs(time_diff) <= config.cdr_ipdr_window_seconds
            
            # Check strong identifiers matched
            strong_matches = sum(1 for e in evidence_list if e.identity_type in {
                IdentityType.PHONE, IdentityType.IMSI, IdentityType.IMEI
            })
            
            # Only Cell ID?
            cell_only = len(evidence_list) == 1 and evidence_list[0].identity_type == IdentityType.CELL_ID
            
            accepted = in_window and strong_matches > 0
            
            if not accepted:
                continue
                
            # Determine match strength
            strength = MatchStrength.NONE
            if accepted:
                # Strong conflicts in strong identifiers
                strong_conflicts = sum(1 for c in conflicts if c.identity_type in {
                    IdentityType.PHONE, IdentityType.IMSI, IdentityType.IMEI
                })
                
                if strong_matches >= 2:
                    strength = MatchStrength.STRONG
                elif strong_matches == 1 and strong_conflicts == 0:
                    strength = MatchStrength.STRONG
                else:
                    strength = MatchStrength.MODERATE
            
            correlation_id = f"{RelationshipType.CDR_IPDR.value}_{cdr.cdr_id}_{ipdr_id}"
            
            record = CorrelationRecord(
                correlation_id=correlation_id,
                relationship_type=RelationshipType.CDR_IPDR,
                source_type=SourceType.CDR,
                source_event_id=cdr.cdr_id,
                source_timestamp=cdr.event_timestamp,
                target_type=SourceType.IPDR,
                target_event_id=ipdr_id,
                target_timestamp=ipdr.session_timestamp,
                time_difference_seconds=int(time_diff),
                identity_evidence=evidence_list,
                conflicting_evidence=conflicts,
                match_strength=strength,
                accepted=accepted
            )
            correlations.append(record)
            
    # Sort for deterministic output
    correlations.sort(key=lambda x: x.correlation_id)
    return correlations
