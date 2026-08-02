import statistics
import bisect
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime

from src.models.bank import BankTransaction, BankParty
from src.models.cdr import CDREvent
from src.models.ipdr import IPDRSession
from src.fusion.models import TransactionContext
from .models import FeatureRow

class FeatureEngine:
    def __init__(self, historical_bank: List[BankTransaction], historical_cdr: List[CDREvent], historical_ipdr: List[IPDRSession]):
        self._historical_bank = historical_bank
        self._historical_cdr = historical_cdr
        self._historical_ipdr = historical_ipdr
        
        self._bank_history: Dict[Tuple[str, str], List[BankTransaction]] = {}
        self._bank_running_medians: Dict[Tuple[str, str], List[float]] = {}
        self._bank_running_mads: Dict[Tuple[str, str], List[float]] = {}
        for txn in self._historical_bank:
            sender_id = self._get_bank_identity(txn.sender)
            if sender_id:
                if sender_id not in self._bank_history:
                    self._bank_history[sender_id] = []
                self._bank_history[sender_id].append(txn)
                
        for k in self._bank_history:
            self._bank_history[k].sort(key=lambda t: t.transaction_timestamp)
            txns = self._bank_history[k]
            sorted_amounts = []
            medians = []
            mads = []
            for t in txns:
                amt = float(t.amount)
                bisect.insort(sorted_amounts, amt)
                n = len(sorted_amounts)
                if n % 2 == 1:
                    med = sorted_amounts[n // 2]
                else:
                    med = (sorted_amounts[n // 2 - 1] + sorted_amounts[n // 2]) / 2.0
                medians.append(med)
                
                abs_devs = [abs(x - med) for x in sorted_amounts]
                abs_devs.sort()
                if n % 2 == 1:
                    mad = abs_devs[n // 2]
                else:
                    mad = (abs_devs[n // 2 - 1] + abs_devs[n // 2]) / 2.0
                mads.append(mad)
            self._bank_running_medians[k] = medians
            self._bank_running_mads[k] = mads

        self._cdr_history: Dict[str, List[CDREvent]] = {}
        for cdr in self._historical_cdr:
            if cdr.a_party_phone:
                if cdr.a_party_phone not in self._cdr_history:
                    self._cdr_history[cdr.a_party_phone] = []
                self._cdr_history[cdr.a_party_phone].append(cdr)
            if cdr.b_party_phone and cdr.b_party_phone != cdr.a_party_phone:
                if cdr.b_party_phone not in self._cdr_history:
                    self._cdr_history[cdr.b_party_phone] = []
                self._cdr_history[cdr.b_party_phone].append(cdr)
                
        for k in self._cdr_history:
            self._cdr_history[k].sort(key=lambda c: c.event_timestamp)

        self._ipdr_history: Dict[Tuple[str, str], List[IPDRSession]] = {}
        self._ipdr_running_medians: Dict[Tuple[str, str], List[float]] = {}
        self._ipdr_running_mads: Dict[Tuple[str, str], List[float]] = {}
        for ipdr in self._historical_ipdr:
            sub_id = self._get_ipdr_identity(ipdr)
            if sub_id:
                if sub_id not in self._ipdr_history:
                    self._ipdr_history[sub_id] = []
                self._ipdr_history[sub_id].append(ipdr)
                
        for k in self._ipdr_history:
            self._ipdr_history[k].sort(key=lambda i: i.session_timestamp)
            sessions = self._ipdr_history[k]
            medians = []
            mads = []
            valid_durs = []
            for sess in sessions:
                if sess.duration_seconds is not None:
                    bisect.insort(valid_durs, sess.duration_seconds)
                
                n = len(valid_durs)
                if n == 0:
                    medians.append(0.0)
                    mads.append(0.0)
                else:
                    if n % 2 == 1:
                        med = valid_durs[n // 2]
                    else:
                        med = (valid_durs[n // 2 - 1] + valid_durs[n // 2]) / 2.0
                    medians.append(med)
                    
                    abs_devs = [abs(x - med) for x in valid_durs]
                    abs_devs.sort()
                    if n % 2 == 1:
                        mad = abs_devs[n // 2]
                    else:
                        mad = (abs_devs[n // 2 - 1] + abs_devs[n // 2]) / 2.0
                    mads.append(mad)
            self._ipdr_running_medians[k] = medians
            self._ipdr_running_mads[k] = mads

    def _get_bank_identity(self, party: BankParty) -> Optional[Tuple[str, str]]:
        if party.customer_id:
            return ("CUSTOMER_ID", party.customer_id)
        if party.account_number:
            return ("ACCOUNT_NUMBER", party.account_number)
        if party.phone:
            return ("PHONE", party.phone)
        return None

    def _get_ipdr_identity(self, ipdr: IPDRSession) -> Optional[Tuple[str, str]]:
        if ipdr.subscriber_imsi:
            return ("IMSI", ipdr.subscriber_imsi)
        if ipdr.subscriber_msisdn:
            return ("MSISDN", ipdr.subscriber_msisdn)
        return None
        
    def _get_history_end_index(self, history: List, timestamp: datetime, type_flag: str) -> int:
        left = 0
        right = len(history)
        while left < right:
            mid = (left + right) // 2
            if type_flag == 'bank':
                evt_time = history[mid].transaction_timestamp
            elif type_flag == 'cdr':
                evt_time = history[mid].event_timestamp
            else:
                evt_time = history[mid].session_timestamp
                
            if evt_time < timestamp:
                left = mid + 1
            else:
                right = mid
        return left

    def _mad(self, data: List[float], median: float) -> float:
        abs_devs = [abs(x - median) for x in data]
        return statistics.median(abs_devs)

    def generate_feature_row(self, context: TransactionContext) -> FeatureRow:
        txn = context.transaction
        t = txn.transaction_timestamp
        row = FeatureRow(transaction_id=txn.transaction_id)
        
        # 10. BANK FEATURES
        row.transaction_amount = float(txn.amount)
        row.transaction_hour = t.hour
        
        sender_id = self._get_bank_identity(txn.sender)
        bank_hist_list = []
        bank_hist_end = 0
        if sender_id and sender_id in self._bank_history:
            bank_hist_list = self._bank_history[sender_id]
            bank_hist_end = self._get_history_end_index(bank_hist_list, t, 'bank')
            
        row.customer_history_count = bank_hist_end
        
        if bank_hist_end > 0:
            med = self._bank_running_medians[sender_id][bank_hist_end - 1]
            if med > 0:
                row.amount_vs_customer_median = float(txn.amount) / med
            
            mad = self._bank_running_mads[sender_id][bank_hist_end - 1]
            if mad > 0:
                row.amount_robust_zscore = 0.6745 * (float(txn.amount) - med) / mad
                
            less_count = sum(1 for i in range(bank_hist_end) if float(bank_hist_list[i].amount) < float(txn.amount))
            row.amount_percentile = less_count / bank_hist_end
            
            prev_txn = bank_hist_list[bank_hist_end - 1]
            row.time_since_previous_transaction = (t - prev_txn.transaction_timestamp).total_seconds()
            
            hour_count = sum(1 for i in range(bank_hist_end) if bank_hist_list[i].transaction_timestamp.hour == t.hour)
            row.hour_rarity = 1.0 - (hour_count / bank_hist_end)
            
            recv_id = self._get_bank_identity(txn.receiver)
            if recv_id:
                recv_count = sum(1 for i in range(bank_hist_end) if self._get_bank_identity(bank_hist_list[i].receiver) == recv_id)
                row.receiver_historical_count = recv_count
                row.receiver_seen_before = 1 if recv_count > 0 else 0
                row.receiver_frequency = recv_count / bank_hist_end
        
        # Velocity
        if bank_hist_end > 0:
            t_10m = t.timestamp() - 600
            t_30m = t.timestamp() - 1800
            t_1h = t.timestamp() - 3600
            
            for i in range(bank_hist_end - 1, -1, -1):
                h = bank_hist_list[i]
                ht = h.transaction_timestamp.timestamp()
                if ht < t_1h:
                    break
                amt = float(h.amount)
                if ht >= t_10m:
                    row.txn_count_previous_10m += 1
                if ht >= t_30m:
                    row.txn_count_previous_30m += 1
                    row.amount_velocity_30m += amt
                row.txn_count_previous_1h += 1
                row.amount_velocity_1h += amt

        # 11. CDR FEATURES
        row.has_cdr_context = 1 if context.has_cdr_context else 0
        
        pre_cdrs = [c for c in context.cdr_events if c.event_timestamp < t]
        if len(pre_cdrs) > 0:
            # nearest
            nearest = max(pre_cdrs, key=lambda c: c.event_timestamp)
            row.nearest_call_before_seconds = (t - nearest.event_timestamp).total_seconds()
            
            t_10m = t.timestamp() - 600
            t_30m = t.timestamp() - 1800
            t_1h = t.timestamp() - 3600
            
            durations_30m = []
            
            for c in pre_cdrs:
                ct = c.event_timestamp.timestamp()
                if ct >= t_10m:
                    row.calls_previous_10m += 1
                if ct >= t_30m:
                    row.calls_previous_30m += 1
                    if c.duration_seconds is not None:
                        durations_30m.append(c.duration_seconds)
                if ct >= t_1h:
                    row.calls_previous_1h += 1
                    
            if durations_30m:
                row.total_call_duration_30m = sum(durations_30m)
                row.max_call_duration_30m = float(max(durations_30m))
                
            sender_phone = txn.sender.phone
            
            # Context counterparties
            counterparties = set()
            imeis = set()
            cells = set()
            roamings = set()
            
            for c in pre_cdrs:
                if sender_phone:
                    if c.a_party_phone == sender_phone:
                        counterparties.add(c.b_party_phone)
                    elif c.b_party_phone == sender_phone:
                        counterparties.add(c.a_party_phone)
                
                if c.imei: imeis.add(c.imei)
                if c.cell_id: cells.add(c.cell_id)
                if c.roaming_circle: roamings.add(c.roaming_circle)
                
            # History check
            if sender_phone and sender_phone in self._cdr_history:
                cdr_hist_list = self._cdr_history[sender_phone]
                cdr_hist_end = self._get_history_end_index(cdr_hist_list, t, 'cdr')
            else:
                cdr_hist_list = []
                cdr_hist_end = 0
                
            hist_counterparties = []
            hist_imeis = set()
            hist_cells = set()
            hist_roamings = set()
            
            for i in range(cdr_hist_end):
                c = cdr_hist_list[i]
                if c.a_party_phone == sender_phone:
                    hist_counterparties.append(c.b_party_phone)
                elif c.b_party_phone == sender_phone:
                    hist_counterparties.append(c.a_party_phone)
                if c.imei: hist_imeis.add(c.imei)
                if c.cell_id: hist_cells.add(c.cell_id)
                if c.roaming_circle: hist_roamings.add(c.roaming_circle)
                
            # Caller novelty
            if counterparties:
                novel = False
                freqs = []
                for cp in counterparties:
                    if cp not in hist_counterparties:
                        novel = True
                        freqs.append(0.0)
                    else:
                        if cdr_hist_end > 0:
                            count = hist_counterparties.count(cp)
                            freqs.append(count / cdr_hist_end)
                row.caller_novelty = 1 if novel else 0
                if freqs:
                    row.caller_historical_frequency = min(freqs)
                    
            if imeis:
                row.imei_novelty = 1 if any(i not in hist_imeis for i in imeis) else 0
            if cells:
                row.cell_novelty = 1 if any(c not in hist_cells for c in cells) else 0
            if roamings:
                row.roaming_change = 1 if any(r not in hist_roamings for r in roamings) else 0

        # 18. IPDR FEATURES
        row.has_ipdr_context = 1 if context.has_ipdr_context else 0
        
        pre_ipdrs = [i for i in context.ipdr_sessions if i.session_timestamp < t]
        if len(pre_ipdrs) > 0:
            nearest = max(pre_ipdrs, key=lambda i: i.session_timestamp)
            row.nearest_session_before_seconds = (t - nearest.session_timestamp).total_seconds()
            
            t_10m = t.timestamp() - 600
            t_30m = t.timestamp() - 1800
            
            for i in pre_ipdrs:
                it = i.session_timestamp.timestamp()
                if it >= t_10m:
                    row.sessions_previous_10m += 1
                if it >= t_30m:
                    row.sessions_previous_30m += 1
                    
            sub_ids = {self._get_ipdr_identity(i) for i in pre_ipdrs}
            sub_id = next(iter(sub_ids)) if sub_ids else None
            
            if sub_id and sub_id in self._ipdr_history:
                ipdr_hist_list = self._ipdr_history[sub_id]
                ipdr_hist_end = self._get_history_end_index(ipdr_hist_list, t, 'ipdr')
            else:
                ipdr_hist_list = []
                ipdr_hist_end = 0
                
            hist_src_ips = {ipdr_hist_list[i].source_ip for i in range(ipdr_hist_end) if ipdr_hist_list[i].source_ip}
            hist_dst_ips = {ipdr_hist_list[i].destination_ip for i in range(ipdr_hist_end) if ipdr_hist_list[i].destination_ip}
            hist_dst_ports = {ipdr_hist_list[i].destination_port for i in range(ipdr_hist_end) if ipdr_hist_list[i].destination_port is not None}
            hist_pairs = {(ipdr_hist_list[i].subscriber_imsi, ipdr_hist_list[i].device_imei) for i in range(ipdr_hist_end) if ipdr_hist_list[i].subscriber_imsi and ipdr_hist_list[i].device_imei}
            
            src_ips = {i.source_ip for i in pre_ipdrs if i.source_ip}
            dst_ips = {i.destination_ip for i in pre_ipdrs if i.destination_ip}
            dst_ports = {i.destination_port for i in pre_ipdrs if i.destination_port is not None}
            pairs = {(i.subscriber_imsi, i.device_imei) for i in pre_ipdrs if i.subscriber_imsi and i.device_imei}
            
            if src_ips:
                row.source_ip_novelty = 1 if any(sip not in hist_src_ips for sip in src_ips) else 0
            if dst_ips:
                row.destination_ip_novelty = 1 if any(dip not in hist_dst_ips for dip in dst_ips) else 0
            if dst_ports:
                row.destination_port_novelty = 1 if any(dp not in hist_dst_ports for dp in dst_ports) else 0
            if pairs:
                row.imsi_imei_pair_novelty = 1 if any(p not in hist_pairs for p in pairs) else 0
                
            # Duration deviation
            if ipdr_hist_end > 0:
                hmed = self._ipdr_running_medians[sub_id][ipdr_hist_end - 1]
                hmad = self._ipdr_running_mads[sub_id][ipdr_hist_end - 1]
                
                if hmad > 0:
                    devs = []
                    for i in pre_ipdrs:
                        if i.duration_seconds is not None:
                            dev = 0.6745 * (i.duration_seconds - hmed) / hmad
                            devs.append(abs(dev))
                    if devs:
                        row.session_duration_deviation = max(devs)
                            
        # Consistency features based on CorrelationRecord
        # We only check correlations strictly before the transaction
        # Wait, the prompt says "For relevant pre-transaction IPDR sessions" implies only the correlations 
        # involving pre-transaction events. We'll filter the correlations where target/source timestamp < t.
        
        has_imei_conflict = False
        has_imei_match = False
        has_cell_conflict = False
        has_cell_match = False
        
        for corr in context.cdr_ipdr_correlations:
            if corr.source_timestamp >= t or corr.target_timestamp >= t:
                continue
                
            for conf in corr.conflicting_evidence:
                if conf.identity_type.value == "IMEI":
                    has_imei_conflict = True
                elif conf.identity_type.value == "CELL_ID":
                    has_cell_conflict = True
                    
            for match in corr.identity_evidence:
                if match.identity_type.value == "IMEI":
                    has_imei_match = True
                elif match.identity_type.value == "CELL_ID":
                    has_cell_match = True
                    
        if has_imei_conflict:
            row.device_consistency = 0
        elif has_imei_match:
            row.device_consistency = 1
            
        if has_cell_conflict:
            row.cell_consistency = 0
        elif has_cell_match:
            row.cell_consistency = 1
        
        return row

    def generate_feature_sets(self, contexts: List[TransactionContext]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        # Returns sets A, B, C as lists of dicts
        rows = []
        for idx, context in enumerate(contexts):
            if idx > 0 and idx % 10000 == 0:
                print(f"Processed {idx}/{len(contexts)} feature rows...")
            rows.append(self.generate_feature_row(context))
            
        set_A = []
        set_B = []
        set_C = []
        
        for row_obj in rows:
            row = row_obj.model_dump(exclude_none=False)
            
            # A: Bank only
            bank_keys = [
                'transaction_id', 'transaction_amount', 'transaction_hour', 'customer_history_count',
                'amount_vs_customer_median', 'amount_robust_zscore', 'amount_percentile',
                'receiver_seen_before', 'receiver_historical_count', 'receiver_frequency',
                'hour_rarity', 'txn_count_previous_10m', 'txn_count_previous_30m', 'txn_count_previous_1h',
                'amount_velocity_30m', 'amount_velocity_1h', 'time_since_previous_transaction'
            ]
            row_A = {k: row[k] for k in bank_keys}
            
            # B: Bank + CDR
            cdr_keys = [
                'has_cdr_context', 'calls_previous_10m', 'calls_previous_30m', 'calls_previous_1h',
                'nearest_call_before_seconds', 'total_call_duration_30m', 'max_call_duration_30m',
                'caller_novelty', 'caller_historical_frequency', 'imei_novelty', 'cell_novelty', 'roaming_change'
            ]
            row_B = {**row_A, **{k: row[k] for k in cdr_keys}}
            
            # C: Bank + CDR + IPDR
            ipdr_keys = [
                'has_ipdr_context', 'sessions_previous_10m', 'sessions_previous_30m',
                'nearest_session_before_seconds', 'source_ip_novelty', 'destination_ip_novelty',
                'destination_port_novelty', 'imsi_imei_pair_novelty', 'device_consistency',
                'cell_consistency', 'session_duration_deviation'
            ]
            row_C = {**row_B, **{k: row[k] for k in ipdr_keys}}
            
            set_A.append(row_A)
            set_B.append(row_B)
            set_C.append(row_C)
            
        return set_A, set_B, set_C
