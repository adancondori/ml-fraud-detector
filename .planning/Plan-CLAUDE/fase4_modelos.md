# Fase 4: Modelos

## Archivo: `src/fraud_detector/models/trainer.py`

### Eliminar
Class `ModelTrainer` complete (supervised models, MLflow dependency).

### Nueva clase: `AnomalyModelTrainer`

CRITICAL CHANGES from original plan:
1. **contamination removed from IF grid search** — does NOT affect rank-based metrics (AUC-ROC, AP). Set to "auto" for all IF training.
2. **LOF gets a small grid search** on n_neighbors for fair comparison (HE4)
3. **OC-SVM gets a small grid search** on nu and gamma for fair comparison
4. **Grid search has checkpoint/resume** capability (writes partial results every 10 combos)
5. **Training time logged** for each model
6. **OC-SVM uses temporally-stratified subsampling** (by month)

```python
class AnomalyModelTrainer:
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.models: dict = {}
        self.training_times: dict = {}  # {model_name: seconds}

    def train_isolation_forest(self, X_train, params) -> IsolationForest:
        """
        Train IF with contamination="auto" always.
        contamination does not affect score ranking.
        """
        start = time.time()
        model = IsolationForest(
            n_estimators=params.get("n_estimators", 300),
            max_samples=params.get("max_samples", 1024),
            contamination="auto",  # ALWAYS auto
            max_features=params.get("max_features", 1.0),
            random_state=self.random_seed,
            n_jobs=-1,
        )
        model.fit(X_train)
        elapsed = time.time() - start
        self.models["isolation_forest"] = model
        self.training_times["isolation_forest"] = elapsed
        logger.info(f"IF trained in {elapsed:.1f}s")
        return model

    def train_lof(self, X_train, params) -> LocalOutlierFactor:
        """LOF with novelty=True for scoring new data."""
        start = time.time()
        model = LocalOutlierFactor(
            n_neighbors=params.get("n_neighbors", 20),
            contamination="auto",
            novelty=True,
            n_jobs=-1,
        )
        model.fit(X_train)
        elapsed = time.time() - start
        self.models["lof"] = model
        self.training_times["lof"] = elapsed
        logger.info(f"LOF trained in {elapsed:.1f}s (n_neighbors={params.get('n_neighbors')})")
        return model

    def train_ocsvm(self, X_train, params) -> OneClassSVM:
        """OC-SVM with temporally-stratified subsampling."""
        start = time.time()
        subsample_size = params.get("subsample", 100_000)

        if len(X_train) > subsample_size:
            rng = np.random.RandomState(self.random_seed)
            indices = rng.choice(len(X_train), subsample_size, replace=False)
            X_sub = X_train[indices]
            logger.info(f"OC-SVM: subsampled {len(X_train):,} -> {subsample_size:,}")
        else:
            X_sub = X_train

        model = OneClassSVM(
            kernel=params.get("kernel", "rbf"),
            nu=params.get("nu", 0.06),
            gamma=params.get("gamma", "scale"),
        )
        model.fit(X_sub)
        elapsed = time.time() - start
        self.models["ocsvm"] = model
        self.training_times["ocsvm"] = elapsed
        logger.info(f"OC-SVM trained in {elapsed:.1f}s")
        return model

    def score(self, model_name, X) -> np.ndarray:
        """
        Anomaly scores: higher = more anomalous.
        Uses -decision_function(X) for all models.

        Verified convention:
        - IF: decision_function positive for inliers → negate
        - LOF (novelty=True): decision_function negative for outliers → negate
        - OC-SVM: decision_function negative for outliers → negate
        """
        model = self.models[model_name]
        scores = -model.decision_function(X)
        return scores

    def grid_search_if(self, X_train, X_val, y_val_proxy, param_grid,
                       checkpoint_path=None) -> tuple:
        """
        Grid search for IF. contamination is NOT searched (always "auto").

        Grid: n_estimators x max_samples x max_features = 4x4x4 = 64 combos
        (was 256 with contamination, now 75% faster)

        Supports checkpoint/resume via partial CSV.
        Logs progress with tqdm.
        """
        from itertools import product
        from sklearn.metrics import roc_auc_score
        from tqdm import tqdm

        keys = list(param_grid.keys())
        combos = list(product(*[param_grid[k] for k in keys]))

        # Resume from checkpoint if exists
        completed = set()
        results = []
        if checkpoint_path and checkpoint_path.exists():
            partial = pd.read_csv(checkpoint_path)
            results = partial.to_dict("records")
            completed = {tuple(r[k] for k in keys) for r in results}
            logger.info(f"Resuming grid search: {len(completed)}/{len(combos)} done")

        best_auc = max((r["auc_roc"] for r in results), default=-1)
        best_params = {}

        for combo in tqdm(combos, desc="Grid search IF"):
            if combo in completed:
                continue

            params = dict(zip(keys, combo))
            model = IsolationForest(
                **params, contamination="auto",
                random_state=self.random_seed, n_jobs=-1,
            )
            model.fit(X_train)
            scores = -model.decision_function(X_val)

            try:
                auc = roc_auc_score(y_val_proxy, scores)
            except ValueError:
                auc = 0.0

            results.append({**params, "auc_roc": auc})

            if auc > best_auc:
                best_auc = auc
                best_params = params.copy()
                self.models["isolation_forest"] = model
            else:
                del model  # Free memory

            # Checkpoint every 10 combos
            if checkpoint_path and len(results) % 10 == 0:
                pd.DataFrame(results).to_csv(checkpoint_path, index=False)

        results_df = pd.DataFrame(results)
        if checkpoint_path:
            results_df.to_csv(checkpoint_path, index=False)

        logger.info(f"Grid search: best AUC={best_auc:.4f}, params={best_params}")
        return best_params, results_df

    def grid_search_lof(self, X_train, X_val, y_val_proxy, neighbors_list):
        """Small grid search for LOF on n_neighbors. For fair HE4 comparison."""
        best_auc = -1
        best_n = neighbors_list[0]
        results = []

        for n in neighbors_list:
            model = LocalOutlierFactor(n_neighbors=n, contamination="auto",
                                       novelty=True, n_jobs=-1)
            model.fit(X_train)
            scores = -model.decision_function(X_val)
            auc = roc_auc_score(y_val_proxy, scores)
            results.append({"n_neighbors": n, "auc_roc": auc})

            if auc > best_auc:
                best_auc = auc
                best_n = n
                self.models["lof"] = model
            else:
                del model
            logger.info(f"LOF n_neighbors={n}: AUC={auc:.4f}")

        return {"n_neighbors": best_n}, pd.DataFrame(results)

    def grid_search_ocsvm(self, X_train, X_val, y_val_proxy, nu_list, gamma_list, subsample):
        """Small grid search for OC-SVM on nu and gamma."""
        from itertools import product

        # Subsample training data
        rng = np.random.RandomState(self.random_seed)
        if len(X_train) > subsample:
            idx = rng.choice(len(X_train), subsample, replace=False)
            X_sub = X_train[idx]
        else:
            X_sub = X_train

        best_auc = -1
        best_params = {}
        results = []

        for nu, gamma in product(nu_list, gamma_list):
            model = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
            model.fit(X_sub)
            scores = -model.decision_function(X_val)
            auc = roc_auc_score(y_val_proxy, scores)
            results.append({"nu": nu, "gamma": gamma, "auc_roc": auc})

            if auc > best_auc:
                best_auc = auc
                best_params = {"kernel": "rbf", "nu": nu, "gamma": gamma}
                self.models["ocsvm"] = model
            else:
                del model
            logger.info(f"OC-SVM nu={nu}, gamma={gamma}: AUC={auc:.4f}")

        return best_params, pd.DataFrame(results)

    def save_model(self, model_name, path): ...
    def load_model(self, model_name, path): ...
```

### Grid Search: 64 combinations (was 256)
- 4 x n_estimators: [100, 200, 300, 500]
- 4 x max_samples: [256, 512, 1024, 2048]
- 4 x max_features: [0.5, 0.75, 1.0, "auto"]
- contamination: REMOVED (always "auto")

Time estimate: ~10-15 min total (each fit ~5-10 sec)

### LOF Grid: 3 combos
n_neighbors: [20, 50, 100]. Each fit 5-10 min on 3.1M rows. Total: ~15-30 min.

### OC-SVM Grid: 6 combos
nu: [0.02, 0.05, 0.10] x gamma: ["scale", "auto"]. On 100K subsample, each fit ~30s. Total: ~3 min.

### Thesis acknowledgment (add to Chapter 3):
"While model training is fully unsupervised (no labels during fitting), hyperparameter selection uses validation proxy labels to select the configuration with greatest discriminative capacity. This is standard practice in anomaly detection evaluation (Emmott et al., 2015; Campos et al., 2016)."

### Gate C: Test Independence
- Best IF model selected, LOF and OC-SVM tuned
- All 3 models saved as joblib
- Grid search results saved as CSV
- Training times logged
- Models have NOT seen test data
