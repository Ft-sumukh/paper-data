import os
import logging
import pandas as pd
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

class DataCleaner:
    """
    Cleans raw price time series, handles missing values via forward/backward filling,
    removes duplicate dates, and formats clean price matrices.
    """
    def __init__(self, processed_data_dir: str = "data/processed"):
        self.processed_data_dir = processed_data_dir
        os.makedirs(self.processed_data_dir, exist_ok=True)

    def clean_prices(
        self,
        raw_df: pd.DataFrame,
        ffill_limit: int = 5,
        drop_threshold: float = 0.20
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Cleans the price DataFrame:
        1. Deduplicates dates keeping the last observation.
        2. Drops assets with missing values exceeding drop_threshold.
        3. Forward fills missing prices up to ffill_limit days (e.g. market holidays).
        4. Backward fills any remaining leading NaNs.
        """
        df = raw_df.copy()
        
        # Ensure DatetimeIndex
        df.index = pd.to_datetime(df.index)
        
        # Deduplicate
        initial_len = len(df)
        df = df[~df.index.duplicated(keep="last")]
        dedup_count = initial_len - len(df)
        
        # Sort chronologically
        df = df.sort_index()

        # Check missing proportion per asset
        nan_proportions = df.isna().mean()
        valid_cols = nan_proportions[nan_proportions <= drop_threshold].index.tolist()
        dropped_cols = nan_proportions[nan_proportions > drop_threshold].index.tolist()
        
        if dropped_cols:
            logger.warning(f"Dropping assets due to excessive missing data (> {drop_threshold*100}%): {dropped_cols}")
            df = df[valid_cols]

        # Forward fill up to ffill_limit, then backward fill leading observations
        df = df.ffill(limit=ffill_limit).bfill()

        # Drop any remaining rows that still contain NaNs
        final_df = df.dropna()

        cleaning_log = {
            "initial_rows": initial_len,
            "final_rows": len(final_df),
            "duplicates_removed": dedup_count,
            "dropped_assets": dropped_cols,
            "retained_assets_count": len(final_df.columns),
            "retained_assets": list(final_df.columns)
        }

        # Save to processed directory
        output_file = os.path.join(self.processed_data_dir, "prices_clean.csv")
        final_df.to_csv(output_file)
        logger.info(f"Cleaned prices saved to {output_file}. Shape: {final_df.shape}")

        return final_df, cleaning_log
