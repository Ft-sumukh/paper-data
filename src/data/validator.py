import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class DataValidator:
    """
    Validates financial time-series data to ensure mathematical and data-integrity standards.
    """
    @staticmethod
    def validate_prices(df: pd.DataFrame, max_nan_ratio: float = 0.10) -> Dict[str, Any]:
        """
        Runs comprehensive validation checks on an asset price DataFrame.
        """
        report = {
            "is_valid": True,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "duplicate_dates": int(df.index.duplicated().sum()),
            "is_monotonic_increasing": bool(df.index.is_monotonic_increasing),
            "negative_or_zero_prices": {},
            "nan_percentages": {},
            "extreme_return_anomalies": {},
            "warnings": [],
            "errors": []
        }

        # 1. Check duplicate dates
        if report["duplicate_dates"] > 0:
            report["errors"].append(f"Found {report['duplicate_dates']} duplicate timestamp observations.")
            report["is_valid"] = False

        # 2. Check monotonic time index
        if not report["is_monotonic_increasing"]:
            report["errors"].append("Date index is not monotonically increasing.")
            report["is_valid"] = False

        # 3. Check for non-positive prices
        for col in df.columns:
            invalid_count = int((df[col] <= 0).sum())
            if invalid_count > 0:
                report["negative_or_zero_prices"][col] = invalid_count
                report["errors"].append(f"Asset {col} contains {invalid_count} non-positive prices.")
                report["is_valid"] = False

        # 4. Check NaN ratio
        for col in df.columns:
            nan_ratio = float(df[col].isna().mean())
            report["nan_percentages"][col] = round(nan_ratio * 100, 2)
            if nan_ratio > max_nan_ratio:
                report["warnings"].append(f"Asset {col} has {report['nan_percentages'][col]}% missing values (threshold: {max_nan_ratio*100}%).")

        # 5. Check for extreme single-day returns (potential unadjusted stock splits or data errors)
        returns = df.pct_change().abs()
        for col in df.columns:
            extreme_events = int((returns[col] > 0.50).sum())  # > 50% single day move
            if extreme_events > 0:
                report["extreme_return_anomalies"][col] = extreme_events
                report["warnings"].append(f"Asset {col} has {extreme_events} daily moves exceeding 50%.")

        logger.info(f"Data validation completed. Valid: {report['is_valid']}, Warnings: {len(report['warnings'])}, Errors: {len(report['errors'])}")
        return report
