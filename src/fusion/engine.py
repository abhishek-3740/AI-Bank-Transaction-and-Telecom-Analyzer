from typing import List, Dict
from src.models.bank import BankTransaction
from src.models.cdr import CDREvent
from src.models.ipdr import IPDRSession
from src.correlation.models import CorrelationRecord, RelationshipType
from src.fusion.models import TransactionContext, TimelineEvent, EventType

class FusionEngine:
    def __init__(
        self,
        bank_transactions: List[BankTransaction],
        cdr_events: List[CDREvent],
        ipdr_sessions: List[IPDRSession],
        bank_cdr_correlations: List[CorrelationRecord],
        cdr_ipdr_correlations: List[CorrelationRecord],
    ):
        self.bank_transactions = bank_transactions
        self.cdr_events = cdr_events
        self.ipdr_sessions = ipdr_sessions
        # Only preserve accepted Stage 4 correlations
        self.bank_cdr_correlations = [c for c in bank_cdr_correlations if c.accepted]
        self.cdr_ipdr_correlations = [c for c in cdr_ipdr_correlations if c.accepted]
        
        # Build indexes
        self._cdr_by_id: Dict[str, CDREvent] = {}
        for cdr in self.cdr_events:
            if cdr.cdr_id in self._cdr_by_id:
                raise ValueError(f"Duplicate CDREvent ID found: {cdr.cdr_id}")
            self._cdr_by_id[cdr.cdr_id] = cdr
            
        self._ipdr_by_id: Dict[str, IPDRSession] = {}
        for ipdr in self.ipdr_sessions:
            if ipdr.ipdr_id in self._ipdr_by_id:
                raise ValueError(f"Duplicate IPDRSession ID found: {ipdr.ipdr_id}")
            self._ipdr_by_id[ipdr.ipdr_id] = ipdr
            
        self._bank_cdr_by_txn_id: Dict[str, List[CorrelationRecord]] = {}
        for corr in self.bank_cdr_correlations:
            if corr.relationship_type != RelationshipType.BANK_CDR:
                raise ValueError(f"Invalid relationship type in bank_cdr_correlations: {corr.relationship_type}")
            if corr.source_type.value != "BANK" or corr.target_type.value != "CDR":
                 raise ValueError("Bank-CDR correlation must have source=BANK and target=CDR")
            self._bank_cdr_by_txn_id.setdefault(corr.source_event_id, []).append(corr)
            
        self._cdr_ipdr_by_cdr_id: Dict[str, List[CorrelationRecord]] = {}
        for corr in self.cdr_ipdr_correlations:
            if corr.relationship_type != RelationshipType.CDR_IPDR:
                raise ValueError(f"Invalid relationship type in cdr_ipdr_correlations: {corr.relationship_type}")
            if corr.source_type.value != "CDR" or corr.target_type.value != "IPDR":
                 raise ValueError("CDR-IPDR correlation must have source=CDR and target=IPDR")
            self._cdr_ipdr_by_cdr_id.setdefault(corr.source_event_id, []).append(corr)

    def _build_timeline(self, 
                        txn: BankTransaction, 
                        cdrs: List[CDREvent], 
                        ipdrs: List[IPDRSession]) -> List[TimelineEvent]:
        events = []
        events.append(TimelineEvent(
            event_type=EventType.BANK,
            event_id=txn.transaction_id,
            timestamp=txn.transaction_timestamp,
            event=txn
        ))
        for cdr in cdrs:
            events.append(TimelineEvent(
                event_type=EventType.CDR,
                event_id=cdr.cdr_id,
                timestamp=cdr.event_timestamp,
                event=cdr
            ))
        for ipdr in ipdrs:
            events.append(TimelineEvent(
                event_type=EventType.IPDR,
                event_id=ipdr.ipdr_id,
                timestamp=ipdr.session_timestamp,
                event=ipdr
            ))
            
        # Sort deterministicly: timestamp -> event_type enum value (BANK < CDR < IPDR alphabetically) -> event_id
        events.sort(key=lambda e: (e.timestamp, e.event_type.value, e.event_id))
        return events
            

    def build_transaction_contexts(self) -> List[TransactionContext]:
        contexts = []
        
        for txn in self.bank_transactions:
            txn_id = txn.transaction_id
            
            # Step 1: Find related CDRs
            txn_bank_cdr_corrs = self._bank_cdr_by_txn_id.get(txn_id, [])
            
            unique_cdr_ids = set()
            distinct_bank_cdr_corrs = {}
            for corr in txn_bank_cdr_corrs:
                if corr.target_event_id not in self._cdr_by_id:
                    raise ValueError(f"Correlation references nonexistent CDR: {corr.target_event_id}")
                unique_cdr_ids.add(corr.target_event_id)
                # Deduplicate relationships exactly using correlation_id
                distinct_bank_cdr_corrs[corr.correlation_id] = corr
                
            cdrs = [self._cdr_by_id[cdr_id] for cdr_id in unique_cdr_ids]
            
            # Step 2: Find related IPDRs (via CDRs)
            unique_ipdr_ids = set()
            distinct_cdr_ipdr_corrs = {}
            
            for cdr_id in unique_cdr_ids:
                cdr_corrs = self._cdr_ipdr_by_cdr_id.get(cdr_id, [])
                for corr in cdr_corrs:
                    if corr.target_event_id not in self._ipdr_by_id:
                        raise ValueError(f"Correlation references nonexistent IPDR: {corr.target_event_id}")
                    unique_ipdr_ids.add(corr.target_event_id)
                    distinct_cdr_ipdr_corrs[corr.correlation_id] = corr
                    
            ipdrs = [self._ipdr_by_id[ipdr_id] for ipdr_id in unique_ipdr_ids]
            
            # Sort items deterministically 
            cdrs.sort(key=lambda c: c.cdr_id)
            ipdrs.sort(key=lambda i: i.ipdr_id)
            final_bank_cdr_corrs = sorted(distinct_bank_cdr_corrs.values(), key=lambda c: c.correlation_id)
            final_cdr_ipdr_corrs = sorted(distinct_cdr_ipdr_corrs.values(), key=lambda c: c.correlation_id)
            
            # Step 3: Build Timeline
            timeline = self._build_timeline(txn, cdrs, ipdrs)
            
            context = TransactionContext(
                transaction=txn,
                cdr_events=cdrs,
                ipdr_sessions=ipdrs,
                bank_cdr_correlations=final_bank_cdr_corrs,
                cdr_ipdr_correlations=final_cdr_ipdr_corrs,
                timeline=timeline
            )
            contexts.append(context)
            
        return contexts

def build_transaction_contexts(
    bank_transactions: List[BankTransaction],
    cdr_events: List[CDREvent],
    ipdr_sessions: List[IPDRSession],
    bank_cdr_correlations: List[CorrelationRecord],
    cdr_ipdr_correlations: List[CorrelationRecord],
) -> List[TransactionContext]:
    """
    Builds transaction-centric contexts from canonical events and correlation records.
    """
    engine = FusionEngine(
        bank_transactions,
        cdr_events,
        ipdr_sessions,
        bank_cdr_correlations,
        cdr_ipdr_correlations
    )
    return engine.build_transaction_contexts()
