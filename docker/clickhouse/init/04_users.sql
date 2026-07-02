-- users: LOCAL minimal copy used ONLY by the UI cohort queries
-- (CohortQuery / CohortDetailQuery) for IP and email-domain grouping.
-- Mirrors the production PeerDB shape: dedup by _peerdb_version, soft-delete
-- via _peerdb_is_deleted. Populate locally with SELECT-from-prod + INSERT-local
-- (read-only prod user); never write to production.
--
-- Columns are the minimal set the UI references:
--   id, email, current_sign_in_ip, _peerdb_is_deleted, _peerdb_version
-- created_at is included for parity with the production table shape.

CREATE TABLE IF NOT EXISTS pbp_productionDB_optimized.users
(
    id                  UInt64,
    email               String DEFAULT '',
    current_sign_in_ip  String DEFAULT '',
    created_at          DateTime DEFAULT now(),
    _peerdb_is_deleted  UInt8 DEFAULT 0,
    _peerdb_version     UInt64 DEFAULT 0
)
ENGINE = ReplacingMergeTree(_peerdb_version)
ORDER BY id
SETTINGS index_granularity = 8192;
