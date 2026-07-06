"""Tests for ShadowDualRunner and BatchScorer dual-run mode.

All ClickHouse and scorer interactions are mocked — no real connections or
model artifacts required.

SHAD-01 invariants tested:
  (a) score_pair returns error-result for a failing champion without affecting challenger.
  (b) score_pair returns error-result for a failing challenger without affecting champion.
  (c) Dual-run mode produces exactly 2 rows per payment (shadow_old + shadow_new).
  (d) Each row has exactly 26 values matching _INSERT_COLUMNS order.
  (e) model_version is 'IF-40-v1' for shadow_old and 'frame-v1' for shadow_new.
  (f) Dedup tokens for the two INSERTs differ by shadow-old-/shadow-new- prefix.
  (g) assert_write_target_is_safe is invoked before any INSERT in dual mode.
  (h) Active mode (scorer_shadow=None) remains backward-compatible (26-column rows).
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fraud_detector.scoring.classifier import ScoringResult
from fraud_detector.scoring.context import UserContext
from scorer.batch.scorer import BatchScorer, _INSERT_COLUMNS
from scorer.shadow.dual_runner import ShadowDualRunner, _error_result


# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

READ_FP = ("prod-clickhouse-host", 8443, "pbp_productionDB_optimized", True, "readonly")
WRITE_FP = ("clickhouse", 8123, "pbp_productionDB_optimized", False, "default")
WRITE_HOST = "clickhouse"
TABLE = "pbp_productionDB_optimized.anomaly_scores"


def _make_mock_ch_client(result_rows=None):
    mock = MagicMock()
    qr = MagicMock()
    qr.result_rows = result_rows or []
    mock.query.return_value = qr
    return mock


def _mock_single_scorer(
    model_version: str = "IF-40-v1",
    feature_version: str = "enriched-40",
    threshold_version: str = "v2",
    risk_level: str = "minimal",
    percentile: float = 0.3,
    n_features: int = 40,
) -> MagicMock:
    """Build a mock SingleTransactionScorer that produces a deterministic ScoringResult."""
    mock = MagicMock()
    mock._model_version = model_version
    mock._feature_version = feature_version
    mock._threshold_version = threshold_version
    mock._feature_names = [f"f{i}" for i in range(n_features)]
    mock._feature_calc.calculate.return_value = np.zeros(n_features)
    mock.score_features.return_value = (0.5, np.zeros((1, n_features), dtype=np.float32))
    mock._classifier.classify.return_value = (False, risk_level, percentile)

    def _score(payment, context=None):
        return ScoringResult(
            score=0.5,
            is_anomaly=False,
            risk_level=risk_level,
            percentile=percentile,
            factors=[],
            model_version=model_version,
            feature_version=feature_version,
            threshold_version=threshold_version,
        )

    mock.score.side_effect = _score
    return mock


def _mock_frame_v1_scorer() -> MagicMock:
    mock = _mock_single_scorer(
        model_version="frame-v1",
        feature_version="frame-operational-v1",
        threshold_version="thresholds-segmented-v1",
        n_features=30,
    )

    def _score(payment, context=None):
        return ScoringResult(
            score=0.4,
            is_anomaly=False,
            risk_level="minimal",
            percentile=0.25,
            factors=[],
            model_version="frame-v1",
            feature_version="frame-operational-v1",
            threshold_version="thresholds-segmented-v1",
            calibration_segment="global",
            fallback_level="global",
            frame_flags={"timezone_missing": False, "currency_missing": False},
        )

    mock.score.side_effect = _score
    return mock


def _sample_payment(payment_id: int = 101, user_id: int = 42) -> dict:
    return {
        "payment_id": payment_id,
        "user_id": user_id,
        "facility_id": 7,
        "facility_name": "Test Facility",
        "reservation_paid_out": 100.0,
        "created_at": datetime(2026, 4, 28, 14, 30, 0),
        "discount": 0.0,
        "tip": 0.0,
        "payment_method": "card",
        "category": "reservation",
        "club_credit_flag": False,
        "paid_by_manager": False,
        "currency": "USD",
        "status": "paid",
        "gateway": "stripe",
        "source_enum": "pbp_web",
    }


def _make_batch_scorer_dual(scorer_champion, scorer_shadow, read_client, write_client):
    return BatchScorer(
        scorer=scorer_champion,
        read_ch_client=read_client,
        write_ch_client=write_client,
        scorer_shadow=scorer_shadow,
        anomaly_scores_table=TABLE,
        read_fingerprint=READ_FP,
        write_fingerprint=WRITE_FP,
        write_host=WRITE_HOST,
        allow_nonlocal_write=False,
    )


# ---------------------------------------------------------------------------
# ShadowDualRunner unit tests
# ---------------------------------------------------------------------------


class TestShadowDualRunnerScorePair:
    """(a)(b) score_pair: partial failure isolation."""

    def test_champion_fails_challenger_succeeds(self):
        """If champion raises, result_old is an error-result; result_new is valid."""
        champion = MagicMock()
        champion._model_version = "IF-40-v1"
        champion.score.side_effect = RuntimeError("model exploded")

        challenger = _mock_frame_v1_scorer()

        runner = ShadowDualRunner(champion, challenger)
        payment = _sample_payment()
        ctx = UserContext()

        r_old, r_new = runner.score_pair(payment, ctx, ctx)

        # Champion produced an error result.
        assert r_old.model_version == "IF-40-v1"
        assert r_old.score == 0.0
        assert r_old.is_anomaly is False
        assert r_old.risk_level == "minimal"
        assert r_old.feature_version == "error"

        # Challenger succeeded normally.
        assert r_new.model_version == "frame-v1"
        assert r_new.score == 0.4
        assert r_new.calibration_segment == "global"

    def test_challenger_fails_champion_succeeds(self):
        """If challenger raises, result_new is an error-result; result_old is valid."""
        champion = _mock_single_scorer()
        challenger = MagicMock()
        challenger._model_version = "frame-v1"
        challenger.score.side_effect = ValueError("scaler mismatch")

        runner = ShadowDualRunner(champion, challenger)
        payment = _sample_payment()
        ctx = UserContext()

        r_old, r_new = runner.score_pair(payment, ctx, ctx)

        assert r_old.model_version == "IF-40-v1"
        assert r_old.score == 0.5  # champion succeeded

        assert r_new.model_version == "frame-v1"
        assert r_new.score == 0.0  # error result
        assert r_new.feature_version == "error"

    def test_both_succeed(self):
        """Both models succeed: no exceptions, valid results."""
        champion = _mock_single_scorer()
        challenger = _mock_frame_v1_scorer()

        runner = ShadowDualRunner(champion, challenger)
        payment = _sample_payment()
        ctx = UserContext()

        r_old, r_new = runner.score_pair(payment, ctx, ctx)

        assert r_old.score == 0.5
        assert r_new.score == 0.4

    def test_score_pair_never_raises(self):
        """score_pair must not propagate any exception."""
        champion = MagicMock()
        champion.score.side_effect = Exception("catastrophic")
        challenger = MagicMock()
        challenger.score.side_effect = Exception("also catastrophic")

        runner = ShadowDualRunner(champion, challenger)
        # Should not raise.
        r_old, r_new = runner.score_pair(_sample_payment(), UserContext(), UserContext())
        assert r_old.feature_version == "error"
        assert r_new.feature_version == "error"

    def test_error_result_model_version_preserved(self):
        """_error_result carries the correct model_version."""
        err = _error_result("IF-40-v1", RuntimeError("oops"))
        assert err.model_version == "IF-40-v1"
        assert err.score == 0.0
        assert err.feature_version == "error"


# ---------------------------------------------------------------------------
# BatchScorer dual-run mode tests
# ---------------------------------------------------------------------------


class TestBatchScorerDualMode:
    """(c)(d)(e) Dual-run: 2 rows/payment, 26 columns, correct model_version."""

    def _setup_read_client(self, n_payments: int = 2):
        """Build a mock read client for n_payments."""
        cursor_end = datetime(2026, 4, 28, 15, 0, 0)
        fetch_rows = []
        for i in range(n_payments):
            fetch_rows.append((
                100 + i, 42 + i, 7, "Test Facility",
                100.0, datetime(2026, 4, 28, 14, 30, 0),
                0.0, 0.0, "card", "reservation", False, False,
                "USD", "paid", 42 + i, datetime(2026, 4, 28, 14, 30, 0),
                "stripe", "pbp_web",
            ))
        mock_read = _make_mock_ch_client()
        cursor_end_result = MagicMock()
        cursor_end_result.result_rows = [(cursor_end,)]
        fetch_result = MagicMock()
        fetch_result.result_rows = fetch_rows
        ctx_result = MagicMock()
        ctx_result.result_rows = []
        mock_read.query.side_effect = (
            [cursor_end_result, fetch_result] + [ctx_result] * 6
        )
        return mock_read, cursor_end

    def test_dual_mode_produces_two_inserts(self):
        """score_batch in dual mode calls write_ch.insert exactly twice."""
        champion = _mock_single_scorer()
        challenger = _mock_frame_v1_scorer()
        mock_read, _ = self._setup_read_client(n_payments=2)
        mock_write = MagicMock()

        from fraud_detector.scoring.scorer import SingleTransactionScorer

        with patch.object(SingleTransactionScorer, "_explain_top_factors", return_value=[]):
            bs = _make_batch_scorer_dual(champion, challenger, mock_read, mock_write)
            bs.score_batch(datetime(2026, 4, 28, 0, 0, 0, tzinfo=timezone.utc))

        # Two INSERT calls: one for shadow_old, one for shadow_new.
        assert mock_write.insert.call_count == 2

    def test_dual_rows_have_26_columns(self):
        """Every row passed to INSERT has exactly 26 values (len(_INSERT_COLUMNS))."""
        assert len(_INSERT_COLUMNS) == 26  # guard the constant itself

        champion = _mock_single_scorer()
        challenger = _mock_frame_v1_scorer()
        mock_read, _ = self._setup_read_client(n_payments=1)
        mock_write = MagicMock()

        from fraud_detector.scoring.scorer import SingleTransactionScorer

        with patch.object(SingleTransactionScorer, "_explain_top_factors", return_value=[]):
            bs = _make_batch_scorer_dual(champion, challenger, mock_read, mock_write)
            bs.score_batch(datetime(2026, 4, 28, 0, 0, 0, tzinfo=timezone.utc))

        for insert_call in mock_write.insert.call_args_list:
            rows = insert_call.args[1]
            for row in rows:
                assert len(row) == 26, (
                    f"Expected 26 columns, got {len(row)}: {row}"
                )

    def test_dual_model_versions(self):
        """shadow_old rows carry model_version='IF-40-v1'; shadow_new carry 'frame-v1'."""
        champion = _mock_single_scorer()
        challenger = _mock_frame_v1_scorer()
        mock_read, _ = self._setup_read_client(n_payments=1)
        mock_write = MagicMock()

        from fraud_detector.scoring.scorer import SingleTransactionScorer

        with patch.object(SingleTransactionScorer, "_explain_top_factors", return_value=[]):
            bs = _make_batch_scorer_dual(champion, challenger, mock_read, mock_write)
            bs.score_batch(datetime(2026, 4, 28, 0, 0, 0, tzinfo=timezone.utc))

        mv_idx = _INSERT_COLUMNS.index("model_version")
        sm_idx = _INSERT_COLUMNS.index("scoring_mode")

        all_rows = []
        for insert_call in mock_write.insert.call_args_list:
            all_rows.extend(insert_call.args[1])

        old_rows = [r for r in all_rows if r[sm_idx] == "shadow_old"]
        new_rows = [r for r in all_rows if r[sm_idx] == "shadow_new"]

        assert len(old_rows) == 1
        assert len(new_rows) == 1
        assert old_rows[0][mv_idx] == "IF-40-v1"
        assert new_rows[0][mv_idx] == "frame-v1"

    def test_dual_scored_count_is_double(self):
        """scored count in the result equals 2× the number of payments."""
        champion = _mock_single_scorer()
        challenger = _mock_frame_v1_scorer()
        mock_read, _ = self._setup_read_client(n_payments=3)
        mock_write = MagicMock()

        from fraud_detector.scoring.scorer import SingleTransactionScorer

        with patch.object(SingleTransactionScorer, "_explain_top_factors", return_value=[]):
            bs = _make_batch_scorer_dual(champion, challenger, mock_read, mock_write)
            result = bs.score_batch(datetime(2026, 4, 28, 0, 0, 0, tzinfo=timezone.utc))

        assert result["processed"] == 3
        assert result["scored"] == 6  # 3 payments × 2 models


# ---------------------------------------------------------------------------
# Dedup token tests  (f)
# ---------------------------------------------------------------------------


class TestDualDedupTokens:
    """(f) The two INSERT tokens differ by shadow-old-/shadow-new- prefix."""

    def test_tokens_differ_by_prefix(self):
        """The two dedup tokens start with shadow-old- and shadow-new- respectively."""
        champion = _mock_single_scorer()
        challenger = _mock_frame_v1_scorer()

        cursor = datetime(2026, 4, 28, 0, 0, 0, tzinfo=timezone.utc)
        cursor_end = datetime(2026, 4, 28, 15, 0, 0)

        fetch_rows = [(
            101, 42, 7, "Test Facility",
            100.0, datetime(2026, 4, 28, 14, 30, 0),
            0.0, 0.0, "card", "reservation", False, False,
            "USD", "paid", 42, datetime(2026, 4, 28, 14, 30, 0),
            "stripe", "pbp_web",
        )]
        mock_read = _make_mock_ch_client()
        cursor_end_result = MagicMock()
        cursor_end_result.result_rows = [(cursor_end,)]
        fetch_result = MagicMock()
        fetch_result.result_rows = fetch_rows
        ctx_result = MagicMock()
        ctx_result.result_rows = []
        mock_read.query.side_effect = [cursor_end_result, fetch_result] + [ctx_result] * 6
        mock_write = MagicMock()

        from fraud_detector.scoring.scorer import SingleTransactionScorer

        with patch.object(SingleTransactionScorer, "_explain_top_factors", return_value=[]):
            bs = _make_batch_scorer_dual(champion, challenger, mock_read, mock_write)
            bs.score_batch(cursor)

        tokens = [
            c.kwargs["settings"]["insert_deduplication_token"]
            for c in mock_write.insert.call_args_list
        ]
        assert len(tokens) == 2
        old_tokens = [t for t in tokens if t.startswith("shadow-old-")]
        new_tokens = [t for t in tokens if t.startswith("shadow-new-")]
        assert len(old_tokens) == 1, f"Expected 1 shadow-old- token, got: {tokens}"
        assert len(new_tokens) == 1, f"Expected 1 shadow-new- token, got: {tokens}"
        # Tokens must differ.
        assert old_tokens[0] != new_tokens[0]


# ---------------------------------------------------------------------------
# Guardrail test  (g)
# ---------------------------------------------------------------------------


class TestGuardrailInDualMode:
    """(g) assert_write_target_is_safe is the first call in _insert_chunks_dual."""

    def test_guardrail_called_before_insert(self):
        """With identical READ/WRITE fingerprints, dual mode raises before inserting."""
        champion = _mock_single_scorer()
        challenger = _mock_frame_v1_scorer()

        cursor_end = datetime(2026, 4, 28, 15, 0, 0)
        fetch_rows = [(
            101, 42, 7, "Test Facility",
            100.0, datetime(2026, 4, 28, 14, 30, 0),
            0.0, 0.0, "card", "reservation", False, False,
            "USD", "paid", 42, datetime(2026, 4, 28, 14, 30, 0),
            "stripe", "pbp_web",
        )]
        mock_read = _make_mock_ch_client()
        cursor_end_result = MagicMock()
        cursor_end_result.result_rows = [(cursor_end,)]
        fetch_result = MagicMock()
        fetch_result.result_rows = fetch_rows
        ctx_result = MagicMock()
        ctx_result.result_rows = []
        mock_read.query.side_effect = [cursor_end_result, fetch_result] + [ctx_result] * 6
        mock_write = MagicMock()

        from fraud_detector.scoring.scorer import SingleTransactionScorer

        with patch.object(SingleTransactionScorer, "_explain_top_factors", return_value=[]):
            bs = BatchScorer(
                scorer=champion,
                read_ch_client=mock_read,
                write_ch_client=mock_write,
                scorer_shadow=challenger,
                anomaly_scores_table=TABLE,
                read_fingerprint=READ_FP,
                write_fingerprint=READ_FP,  # identical → must abort
                write_host="prod-clickhouse-host",
                allow_nonlocal_write=False,
            )
            with pytest.raises(ValueError, match="same ClickHouse"):
                bs.score_batch(datetime(2026, 4, 28, 0, 0, 0, tzinfo=timezone.utc))

        mock_write.insert.assert_not_called()


# ---------------------------------------------------------------------------
# Active mode retrocompat test  (h)
# ---------------------------------------------------------------------------


class TestActiveModeRetrocompat:
    """(h) scorer_shadow=None keeps the existing active-mode behavior."""

    def test_active_mode_single_insert(self):
        """Active mode (scorer_shadow=None) calls write_ch.insert exactly once."""
        champion = _mock_single_scorer()

        cursor_end = datetime(2026, 4, 28, 15, 0, 0)
        fetch_rows = [(
            101, 42, 7, "Test Facility",
            100.0, datetime(2026, 4, 28, 14, 30, 0),
            0.0, 0.0, "card", "reservation", False, False,
            "USD", "paid", 42, datetime(2026, 4, 28, 14, 30, 0),
            "stripe", "pbp_web",
        )]
        mock_read = _make_mock_ch_client()
        cursor_end_result = MagicMock()
        cursor_end_result.result_rows = [(cursor_end,)]
        fetch_result = MagicMock()
        fetch_result.result_rows = fetch_rows
        ctx_result = MagicMock()
        ctx_result.result_rows = []
        mock_read.query.side_effect = [cursor_end_result, fetch_result] + [ctx_result] * 6
        mock_write = MagicMock()

        from fraud_detector.scoring.scorer import SingleTransactionScorer

        with patch.object(SingleTransactionScorer, "_explain_top_factors", return_value=[]):
            bs = BatchScorer(
                scorer=champion,
                read_ch_client=mock_read,
                write_ch_client=mock_write,
                scorer_shadow=None,  # active mode
                anomaly_scores_table=TABLE,
                read_fingerprint=READ_FP,
                write_fingerprint=WRITE_FP,
                write_host=WRITE_HOST,
                allow_nonlocal_write=False,
            )
            result = bs.score_batch(datetime(2026, 4, 28, 0, 0, 0, tzinfo=timezone.utc))

        mock_write.insert.assert_called_once()
        assert result["scored"] == 1

    def test_active_mode_rows_have_26_columns(self):
        """Active mode rows include the 3 frame-v1 columns as empty strings."""
        assert len(_INSERT_COLUMNS) == 26

        champion = _mock_single_scorer()

        cursor_end = datetime(2026, 4, 28, 15, 0, 0)
        fetch_rows = [(
            101, 42, 7, "Test Facility",
            100.0, datetime(2026, 4, 28, 14, 30, 0),
            0.0, 0.0, "card", "reservation", False, False,
            "USD", "paid", 42, datetime(2026, 4, 28, 14, 30, 0),
            "stripe", "pbp_web",
        )]
        mock_read = _make_mock_ch_client()
        cursor_end_result = MagicMock()
        cursor_end_result.result_rows = [(cursor_end,)]
        fetch_result = MagicMock()
        fetch_result.result_rows = fetch_rows
        ctx_result = MagicMock()
        ctx_result.result_rows = []
        mock_read.query.side_effect = [cursor_end_result, fetch_result] + [ctx_result] * 6
        mock_write = MagicMock()

        from fraud_detector.scoring.scorer import SingleTransactionScorer

        with patch.object(SingleTransactionScorer, "_explain_top_factors", return_value=[]):
            bs = BatchScorer(
                scorer=champion,
                read_ch_client=mock_read,
                write_ch_client=mock_write,
                scorer_shadow=None,
                anomaly_scores_table=TABLE,
                read_fingerprint=READ_FP,
                write_fingerprint=WRITE_FP,
                write_host=WRITE_HOST,
                allow_nonlocal_write=False,
            )
            bs.score_batch(datetime(2026, 4, 28, 0, 0, 0, tzinfo=timezone.utc))

        insert_call = mock_write.insert.call_args
        rows = insert_call.args[1]
        for row in rows:
            assert len(row) == 26

        # Trailing 3 columns must be empty strings for active/IF-40 mode.
        cs_idx = _INSERT_COLUMNS.index("calibration_segment")
        fl_idx = _INSERT_COLUMNS.index("fallback_level")
        ff_idx = _INSERT_COLUMNS.index("frame_flags")
        for row in rows:
            assert row[cs_idx] == ""
            assert row[fl_idx] == ""
            assert row[ff_idx] == ""

    def test_active_mode_token_starts_with_batch(self):
        """Active mode dedup tokens keep the existing 'batch-' prefix."""
        champion = _mock_single_scorer()

        cursor_end = datetime(2026, 4, 28, 15, 0, 0)
        fetch_rows = [(
            101, 42, 7, "Test Facility",
            100.0, datetime(2026, 4, 28, 14, 30, 0),
            0.0, 0.0, "card", "reservation", False, False,
            "USD", "paid", 42, datetime(2026, 4, 28, 14, 30, 0),
            "stripe", "pbp_web",
        )]
        mock_read = _make_mock_ch_client()
        cursor_end_result = MagicMock()
        cursor_end_result.result_rows = [(cursor_end,)]
        fetch_result = MagicMock()
        fetch_result.result_rows = fetch_rows
        ctx_result = MagicMock()
        ctx_result.result_rows = []
        mock_read.query.side_effect = [cursor_end_result, fetch_result] + [ctx_result] * 6
        mock_write = MagicMock()

        from fraud_detector.scoring.scorer import SingleTransactionScorer

        with patch.object(SingleTransactionScorer, "_explain_top_factors", return_value=[]):
            bs = BatchScorer(
                scorer=champion,
                read_ch_client=mock_read,
                write_ch_client=mock_write,
                scorer_shadow=None,
                anomaly_scores_table=TABLE,
                read_fingerprint=READ_FP,
                write_fingerprint=WRITE_FP,
                write_host=WRITE_HOST,
                allow_nonlocal_write=False,
            )
            bs.score_batch(datetime(2026, 4, 28, 0, 0, 0, tzinfo=timezone.utc))

        token = mock_write.insert.call_args.kwargs["settings"]["insert_deduplication_token"]
        assert token.startswith("batch-"), f"Expected 'batch-' prefix, got: {token!r}"


# ---------------------------------------------------------------------------
# _INSERT_COLUMNS structure test
# ---------------------------------------------------------------------------


def test_insert_columns_length():
    """_INSERT_COLUMNS must have exactly 26 entries after adding frame-v1 columns."""
    assert len(_INSERT_COLUMNS) == 26


def test_insert_columns_frame_v1_at_end():
    """The 3 frame-v1 columns must be the last 3 entries in _INSERT_COLUMNS."""
    assert _INSERT_COLUMNS[-3] == "calibration_segment"
    assert _INSERT_COLUMNS[-2] == "fallback_level"
    assert _INSERT_COLUMNS[-1] == "frame_flags"
