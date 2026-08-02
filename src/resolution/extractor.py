from typing import List
from src.models.common import EntityIdentity, IdentityType, SourceType
from src.models.observation import IdentityObservation, RoleType
from src.models.bank import BankTransaction
from src.models.cdr import CDREvent
from src.models.ipdr import IPDRSession

def extract_bank_identities(txn: BankTransaction) -> List[IdentityObservation]:
    obs = []
    ts = txn.transaction_timestamp
    src_type = SourceType.BANK
    src_id = txn.transaction_id
    
    # SENDER
    if txn.sender.phone:
        obs.append(IdentityObservation(
            identity=EntityIdentity(identity_type=IdentityType.PHONE, raw_value=txn.sender.phone, normalized_value=txn.sender.phone),
            source_type=src_type, source_record_id=src_id, source_field="sender.phone", role=RoleType.SENDER, timestamp=ts
        ))
    if txn.sender.account_number:
        obs.append(IdentityObservation(
            identity=EntityIdentity(identity_type=IdentityType.BANK_ACCOUNT, raw_value=txn.sender.account_number, normalized_value=txn.sender.account_number),
            source_type=src_type, source_record_id=src_id, source_field="sender.account_number", role=RoleType.SENDER, timestamp=ts
        ))
    if txn.sender.customer_id:
        obs.append(IdentityObservation(
            identity=EntityIdentity(identity_type=IdentityType.CUSTOMER_ID, raw_value=txn.sender.customer_id, normalized_value=txn.sender.customer_id),
            source_type=src_type, source_record_id=src_id, source_field="sender.customer_id", role=RoleType.SENDER, timestamp=ts
        ))
        
    # RECEIVER
    if txn.receiver.phone:
        obs.append(IdentityObservation(
            identity=EntityIdentity(identity_type=IdentityType.PHONE, raw_value=txn.receiver.phone, normalized_value=txn.receiver.phone),
            source_type=src_type, source_record_id=src_id, source_field="receiver.phone", role=RoleType.RECEIVER, timestamp=ts
        ))
    if txn.receiver.account_number:
        obs.append(IdentityObservation(
            identity=EntityIdentity(identity_type=IdentityType.BANK_ACCOUNT, raw_value=txn.receiver.account_number, normalized_value=txn.receiver.account_number),
            source_type=src_type, source_record_id=src_id, source_field="receiver.account_number", role=RoleType.RECEIVER, timestamp=ts
        ))
    if txn.receiver.customer_id:
        obs.append(IdentityObservation(
            identity=EntityIdentity(identity_type=IdentityType.CUSTOMER_ID, raw_value=txn.receiver.customer_id, normalized_value=txn.receiver.customer_id),
            source_type=src_type, source_record_id=src_id, source_field="receiver.customer_id", role=RoleType.RECEIVER, timestamp=ts
        ))
        
    return obs

def extract_cdr_identities(event: CDREvent) -> List[IdentityObservation]:
    obs = []
    ts = event.event_timestamp
    src_type = SourceType.CDR
    src_id = event.cdr_id
    
    if event.a_party_phone:
        obs.append(IdentityObservation(
            identity=EntityIdentity(identity_type=IdentityType.PHONE, raw_value=event.a_party_phone, normalized_value=event.a_party_phone),
            source_type=src_type, source_record_id=src_id, source_field="a_party_phone", role=RoleType.A_PARTY, timestamp=ts
        ))
    if event.b_party_phone:
        obs.append(IdentityObservation(
            identity=EntityIdentity(identity_type=IdentityType.PHONE, raw_value=event.b_party_phone, normalized_value=event.b_party_phone),
            source_type=src_type, source_record_id=src_id, source_field="b_party_phone", role=RoleType.B_PARTY, timestamp=ts
        ))
    if event.imsi:
        obs.append(IdentityObservation(
            identity=EntityIdentity(identity_type=IdentityType.IMSI, raw_value=event.imsi, normalized_value=event.imsi),
            source_type=src_type, source_record_id=src_id, source_field="imsi", role=RoleType.SUBSCRIBER, timestamp=ts
        ))
    if event.imei:
        obs.append(IdentityObservation(
            identity=EntityIdentity(identity_type=IdentityType.IMEI, raw_value=event.imei, normalized_value=event.imei),
            source_type=src_type, source_record_id=src_id, source_field="imei", role=RoleType.DEVICE, timestamp=ts
        ))
    if event.cell_id:
        obs.append(IdentityObservation(
            identity=EntityIdentity(identity_type=IdentityType.CELL_ID, raw_value=event.cell_id, normalized_value=event.cell_id),
            source_type=src_type, source_record_id=src_id, source_field="cell_id", role=RoleType.CELL, timestamp=ts
        ))
        
    return obs

def extract_ipdr_identities(session: IPDRSession) -> List[IdentityObservation]:
    obs = []
    ts = session.session_timestamp
    src_type = SourceType.IPDR
    src_id = session.ipdr_id
    
    if session.subscriber_msisdn:
        # Note: IPDR MSISDN mapped to PHONE to bridge with CDR
        obs.append(IdentityObservation(
            identity=EntityIdentity(identity_type=IdentityType.PHONE, raw_value=session.subscriber_msisdn, normalized_value=session.subscriber_msisdn),
            source_type=src_type, source_record_id=src_id, source_field="subscriber_msisdn", role=RoleType.SUBSCRIBER, timestamp=ts
        ))
    if session.subscriber_imsi:
        obs.append(IdentityObservation(
            identity=EntityIdentity(identity_type=IdentityType.IMSI, raw_value=session.subscriber_imsi, normalized_value=session.subscriber_imsi),
            source_type=src_type, source_record_id=src_id, source_field="subscriber_imsi", role=RoleType.SUBSCRIBER, timestamp=ts
        ))
    if session.device_imei:
        obs.append(IdentityObservation(
            identity=EntityIdentity(identity_type=IdentityType.IMEI, raw_value=session.device_imei, normalized_value=session.device_imei),
            source_type=src_type, source_record_id=src_id, source_field="device_imei", role=RoleType.DEVICE, timestamp=ts
        ))
    if session.cell_id:
        obs.append(IdentityObservation(
            identity=EntityIdentity(identity_type=IdentityType.CELL_ID, raw_value=session.cell_id, normalized_value=session.cell_id),
            source_type=src_type, source_record_id=src_id, source_field="cell_id", role=RoleType.CELL, timestamp=ts
        ))
    if session.source_ip:
        obs.append(IdentityObservation(
            identity=EntityIdentity(identity_type=IdentityType.IP_ADDRESS, raw_value=session.source_ip, normalized_value=session.source_ip),
            source_type=src_type, source_record_id=src_id, source_field="source_ip", role=RoleType.SOURCE_IP, timestamp=ts
        ))
    if session.destination_ip:
        obs.append(IdentityObservation(
            identity=EntityIdentity(identity_type=IdentityType.IP_ADDRESS, raw_value=session.destination_ip, normalized_value=session.destination_ip),
            source_type=src_type, source_record_id=src_id, source_field="destination_ip", role=RoleType.DESTINATION_IP, timestamp=ts
        ))
        
    return obs
