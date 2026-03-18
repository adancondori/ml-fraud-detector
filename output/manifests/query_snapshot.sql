
SELECT
    id,
    user_id,
    effective_user_id,
    facility_id,
    facility_name,
    created_at,
    captured_at,
    payment_method,
    gateway,
    source_enum,
    status,
    reservation_paid_out,
    discount,
    tax,
    tip,
    card_brand,
    currency,
    paid_by_manager,
    reversed_id,
    debit_refund,
    _peerdb_version
FROM pbp_productionDB_optimized.payments FINAL
WHERE created_at >= %(start)s
  AND created_at < %(end)s
  AND payment_method != 'reversal'
  AND payment_method != 'free'
  AND user_id != 0
  AND _peerdb_is_deleted = 0
ORDER BY created_at, id
