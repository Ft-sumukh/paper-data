import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from typing import Tuple, Optional

class GaussianMixtureRegimeDetector:
    """
    Classifies market regimes using unsupervised Gaussian Mixture clustering
    on joint rolling returns and realized volatilities.
    """
    def __init__(self, n_regimes: int = 3, window: int = 63, random_state: int = 42):
        self.n_regimes = n_regimes
        self.window = window
        self.random_state = random_state
        self.gmm = GaussianMixture(
            n_components=n_regimes,
            covariance_type="full",
            random_state=random_state,
            n_init=5
        )

    def fit_predict(self, benchmark_returns: pd.Series) -> pd.DataFrame:
        """
        Fits GMM on rolling return & volatility features, ordering clusters by volatility.
        """
        roll_ret = benchmark_returns.rolling(self.window).mean() * 252.0
        roll_vol = benchmark_returns.rolling(self.window).std() * np.sqrt(252.0)
        
        feature_df = pd.DataFrame({"return": roll_ret, "volatility": roll_vol}).dropna()
        X = feature_df.values

        raw_labels = self.gmm.fit_predict(X)
        
        # Sort cluster labels by ascending mean volatility
        # (Cluster 0: Low Vol, Cluster 1: Normal Vol, Cluster 2: High Vol)
        cluster_vols = [float(X[raw_labels == k, 1].mean()) for k in range(self.n_regimes)]
        sorted_indices = np.argsort(cluster_vols)
        mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_indices)}
        
        ordered_codes = np.array([mapping[lbl] for lbl in raw_labels])
        label_names = {0: "LOW_VOL", 1: "NORMAL_VOL", 2: "HIGH_VOL"}
        ordered_labels = [label_names.get(c, f"REGIME_{c}") for c in ordered_codes]

        res_df = pd.DataFrame({
            "rolling_return": feature_df["return"],
            "realized_volatility": feature_df["volatility"],
            "regime_code": ordered_codes,
            "regime": ordered_labels
        }, index=feature_df.index)

        return res_df
