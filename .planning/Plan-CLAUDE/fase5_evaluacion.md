# Fase 5: Evaluacion

## Archivo: `src/fraud_detector/evaluation/metrics.py`

### Eliminar
Class `FraudMetrics`, functions `evaluate_fraud_model()`, `find_optimal_threshold()`.

### Nueva clase: `HypothesisEvaluator`

CRITICAL CHANGES from original plan:
1. **Rank-biserial r formula FIXED** — use `r = 2*U/(n1*n2) - 1` for positive r when anomaly>normal
2. **CLES added** (Common Language Effect Size): P(anomaly score > normal score) = U/(n1*n2)
3. **Multiple k values** for top-K (1%, 2%, 5%, 10%) instead of just 5%
4. **AP threshold raised** from base_rate to 2*base_rate
5. **EF threshold raised** from >1.0 to >2.0
6. **Holm-Bonferroni correction** applied across HE1-HE4
7. **Temporal stability analysis** added (monthly AUC on test set)
8. **Per-status proxy evaluation** added
9. **KS statistic** added as complementary separation metric
10. **Feature 17 sensitivity** includes Jaccard similarity on top-K, not just AUC delta

```python
class HypothesisEvaluator:
    """Evaluates anomaly detection models against thesis hypotheses HE1-HE4."""

    # --- HE1: Statistical Separation ---
    def test_mann_whitney(self, scores, proxy) -> dict:
        """
        Mann-Whitney U test. With N>2.5M, p-values are always tiny.
        Focus on effect sizes: rank-biserial r and CLES.

        Returns: U_statistic, p_value, rank_biserial_r, cles, n_anomaly, n_normal, he1_pass

        FIX: r = 2*U/(n1*n2) - 1 (positive when anomaly scores are higher)
        Added: CLES = U/(n1*n2) (probability that random anomaly > random normal)
        """
        scores_anomaly = scores[proxy == 1]
        scores_normal = scores[proxy == 0]
        U, p = mannwhitneyu(scores_anomaly, scores_normal, alternative="greater")
        n1, n2 = len(scores_anomaly), len(scores_normal)
        r = 2 * U / (n1 * n2) - 1  # FIXED: positive when anomaly > normal
        cles = U / (n1 * n2)  # NEW: Common Language Effect Size
        he1_pass = (p < 0.05) and (r > 0.10)
        return {"U_statistic": U, "p_value": p, "rank_biserial_r": r,
                "cles": cles, "n_anomaly": n1, "n_normal": n2, "he1_pass": he1_pass}

    # --- HE2: Discriminative Capacity ---
    def compute_discrimination(self, scores, proxy) -> dict:
        """
        AUC-ROC and Average Precision.
        Criteria: AUC > 0.70 AND AP > 2*base_rate (raised from base_rate)
        """
        auc = roc_auc_score(proxy, scores)
        ap = average_precision_score(proxy, scores)
        base_rate = proxy.mean()
        ap_ratio = ap / base_rate if base_rate > 0 else 0
        he2_pass = (auc > 0.70) and (ap > 2 * base_rate)
        return {"auc_roc": auc, "average_precision": ap, "base_rate": base_rate,
                "ap_over_baseline": ap_ratio, "he2_pass": he2_pass}

    # --- HE3: Top-K Concentration ---
    def compute_topk(self, scores, proxy, k_values=[0.01, 0.02, 0.05, 0.10]) -> dict:
        """
        Evaluate at MULTIPLE k values (was only 5%).
        Criteria: EF > 2.0 at k=5% (raised from >1.0)
        """
        results = {}
        for k in k_values:
            n = len(scores)
            k_count = max(1, int(n * k))
            top_k_idx = np.argsort(scores)[-k_count:]
            proxy_in_topk = proxy[top_k_idx].sum()
            total_anomalies = proxy.sum()
            base_rate = proxy.mean()
            precision_at_k = proxy_in_topk / k_count
            recall_at_k = proxy_in_topk / total_anomalies if total_anomalies > 0 else 0
            ef = precision_at_k / base_rate if base_rate > 0 else 0
            k_pct = int(k * 100)
            results[f"precision_at_{k_pct}pct"] = precision_at_k
            results[f"recall_at_{k_pct}pct"] = recall_at_k
            results[f"ef_at_{k_pct}pct"] = ef

        # Primary criterion at k=5%
        results["he3_pass"] = results.get("ef_at_5pct", 0) > 2.0
        # Also store convenience values
        results["precision_at_k"] = results.get("precision_at_5pct", 0)
        results["recall_at_k"] = results.get("recall_at_5pct", 0)
        results["enrichment_factor"] = results.get("ef_at_5pct", 0)
        return results

    # --- HE4: Model Comparison ---
    def compare_models(self, results_dict) -> dict:
        """
        Compare IF vs LOF/OC-SVM.
        IF >= best competitor in >=2 of 4 metrics.

        NOTE: All models are now tuned (IF grid, LOF n_neighbors, OC-SVM nu/gamma).
        Thesis states: "All models received hyperparameter tuning proportional to
        their search space complexity."
        """
        # (same logic as before)
        ...

    # --- Bootstrap CI ---
    def bootstrap_ci(self, scores, proxy, metric_fn, n_iterations=1000,
                     ci=0.95, random_seed=42) -> dict:
        """
        Bootstrap with tqdm progress bar.
        Returns: mean, lower, upper, std, n_iterations
        """
        from tqdm import tqdm
        rng = np.random.RandomState(random_seed)
        n = len(scores)
        values = []
        for _ in tqdm(range(n_iterations), desc="Bootstrap", leave=False):
            idx = rng.choice(n, n, replace=True)
            values.append(metric_fn(scores[idx], proxy[idx]))
        values = np.array(values)
        alpha = (1 - ci) / 2
        return {"mean": np.mean(values), "lower": np.percentile(values, alpha*100),
                "upper": np.percentile(values, (1-alpha)*100),
                "std": np.std(values), "n_iterations": n_iterations}

    # --- NEW: Temporal Stability ---
    def temporal_stability(self, scores, proxy, dates, model_name) -> dict:
        """
        Compute monthly AUC on test set to verify model doesn't degrade over time.
        Answers: "Will this model still work 3 months after training?"
        """
        df_temp = pd.DataFrame({"scores": scores, "proxy": proxy, "date": dates})
        df_temp["month"] = df_temp["date"].dt.to_period("M")
        monthly = {}
        for month, group in df_temp.groupby("month"):
            if group["proxy"].nunique() < 2:
                continue
            auc = roc_auc_score(group["proxy"], group["scores"])
            monthly[str(month)] = {"auc_roc": auc, "n_samples": len(group),
                                   "proxy_rate": group["proxy"].mean()}
        return {"model_name": model_name, "monthly_auc": monthly}

    # --- NEW: Per-Status Proxy Evaluation ---
    def per_status_evaluation(self, scores, df_test) -> dict:
        """
        Compute AUC using each status individually as positive label.
        Answers: "Is the model better at detecting total refunds vs credit refunds?"
        """
        statuses = ["totally_refunded", "refunded_to_credit", "partially_refunded"]
        results = {}
        for status in statuses:
            proxy = (df_test["status"] == status).astype(int)
            if proxy.sum() < 10:
                continue
            auc = roc_auc_score(proxy, scores)
            results[status] = {"auc_roc": auc, "count": int(proxy.sum())}
        return results

    # --- NEW: KS Statistic ---
    def ks_test(self, scores, proxy) -> dict:
        """Kolmogorov-Smirnov test for score distribution separation."""
        from scipy.stats import ks_2samp
        stat, p = ks_2samp(scores[proxy == 1], scores[proxy == 0])
        return {"ks_statistic": stat, "p_value": p}

    # --- Sensitivity: Proxy ---
    def sensitivity_proxy(self, scores, df_test) -> dict:
        """Compare strict vs wide proxy. Plus per-status breakdown."""
        # (same as before, plus per-status)
        ...

    # --- Sensitivity: Feature 17 (ENHANCED) ---
    def sensitivity_feature17(self, trainer, X_train_20, X_train_19,
                              X_val, X_val_19, y_val_proxy, best_params) -> dict:
        """
        Compare AUC with/without feature 17.
        NEW: Also compute Jaccard similarity on top-5% sets and Spearman correlation.
        """
        # Train both models
        # Compare AUC delta < 0.02
        # NEW: Jaccard similarity of top-5% flagged transactions
        scores_20 = ...
        scores_19 = ...

        top_5pct_20 = set(np.argsort(scores_20)[-k:])
        top_5pct_19 = set(np.argsort(scores_19)[-k:])
        jaccard = len(top_5pct_20 & top_5pct_19) / len(top_5pct_20 | top_5pct_19)

        # NEW: Spearman rank correlation
        from scipy.stats import spearmanr
        spearman_r, _ = spearmanr(scores_20, scores_19)

        return {
            "auc_20_features": auc_20, "auc_19_features": auc_19,
            "delta_auc": delta, "low_sensitivity": delta < 0.02,
            "jaccard_top5pct": jaccard, "spearman_r": spearman_r,
        }

    # --- Holm-Bonferroni Correction ---
    def apply_holm_bonferroni(self, p_values: list) -> list:
        """
        Apply Holm-Bonferroni correction for multiple hypothesis testing.
        Returns adjusted p-values.
        """
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        adjusted = np.zeros(n)
        for rank, idx in enumerate(sorted_indices):
            adjusted[idx] = p_values[idx] * (n - rank)
        adjusted = np.minimum(adjusted, 1.0)
        return adjusted.tolist()

    # --- Full Evaluation ---
    def full_evaluation(self, model_name, scores, proxy,
                        dates=None, bootstrap_n=1000,
                        top_k_values=[0.01, 0.02, 0.05, 0.10]) -> dict:
        """
        Complete evaluation: HE1 + HE2 + HE3 + KS + bootstrap CIs + temporal stability
        """
        he1 = self.test_mann_whitney(scores, proxy)
        he2 = self.compute_discrimination(scores, proxy)
        he3 = self.compute_topk(scores, proxy, k_values=top_k_values)
        ks = self.ks_test(scores, proxy)

        # Bootstrap CIs
        ci_auc = self.bootstrap_ci(scores, proxy, lambda s,p: roc_auc_score(p,s), bootstrap_n)
        ci_ap = self.bootstrap_ci(scores, proxy, lambda s,p: average_precision_score(p,s), bootstrap_n)

        result = {
            "model_name": model_name,
            "he1": he1, "he2": he2, "he3": he3, "ks": ks,
            "bootstrap_ci_auc": ci_auc, "bootstrap_ci_ap": ci_ap,
        }

        # Temporal stability if dates provided
        if dates is not None:
            result["temporal_stability"] = self.temporal_stability(scores, proxy, dates, model_name)

        return result
```

### Gate D: Robustness
Before proceeding to Phase 6:
1. All 3 models evaluated on test set
2. HE1-HE4 computed with Holm-Bonferroni correction
3. Bootstrap CIs computed (lower < mean < upper for all)
4. Temporal stability: monthly AUC computed (flag if any month drops >0.10 from overall)
5. Sensitivity proxy: delta_auc < 0.05 between strict and wide
6. Sensitivity feature17: delta_auc < 0.02 and Jaccard > 0.80
7. Per-status evaluation computed
8. results.json saved with complete structure
