
SELECT
    p.id AS id,
    p.user_id AS user_id,
    p.effective_user_id AS effective_user_id,
    p.facility_id AS facility_id,
    p.facility_name AS facility_name,
    p.created_at AS created_at,
    p.captured_at AS captured_at,
    p.payment_method AS payment_method,
    p.gateway AS gateway,
    p.source_enum AS source_enum,
    p.status AS status,
    p.reservation_paid_out AS reservation_paid_out,
    p.discount AS discount,
    p.tax AS tax,
    p.tip AS tip,
    p.card_brand AS card_brand,
    p.currency AS currency,
    p.paid_by_manager AS paid_by_manager,
    p.reversed_id AS reversed_id,
    p.debit_refund AS debit_refund,
    p.category AS category,
    p.club_credit_flag AS club_credit_flag,
    p._peerdb_version AS _peerdb_version,
    CASE
        WHEN fu.role IN ('court_manager', 'court_operator', 'teacher') THEN 1
        ELSE 0
    END AS is_staff,
    coalesce(fu.role, 'player') AS user_role,
    u.created_at AS user_created_at
FROM pbp_productionDB_optimized.payments AS p FINAL
LEFT ANY JOIN (
    SELECT user_id, facility_id, role
    FROM pbp_productionDB_optimized.facilities_users FINAL
    WHERE _peerdb_is_deleted = 0
) AS fu
    ON p.user_id = fu.user_id
   AND p.facility_id = fu.facility_id
LEFT ANY JOIN (
    SELECT id, created_at
    FROM pbp_productionDB_optimized.users FINAL
    WHERE _peerdb_is_deleted = 0
) AS u
    ON p.user_id = u.id
WHERE p.created_at >= %(start)s
  AND p.created_at < %(end)s
  AND p.payment_method != 'reversal'
  AND p.payment_method != 'free'
  AND p.user_id != 0
  AND p._peerdb_is_deleted = 0
ORDER BY p.created_at, p.id
