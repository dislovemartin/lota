# sample_data.py (Sample Data for Testing)
import pandas as pd
import numpy as np

def get_sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame({
        "feature1": [
            1,
            2,
            np.nan,
            4
        ],
        "feature2": [
            "A",
            "B",
            "A",
            "B"
        ]
    })

