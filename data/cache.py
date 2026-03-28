import pandas as pd
from pathlib import Path


class CacheManager:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, asset: str, timeframe: str) -> Path:
        safe_asset = asset.replace("/", "_")
        return self.cache_dir / safe_asset / f"{timeframe}.parquet"

    def write(self, asset: str, timeframe: str, df: pd.DataFrame) -> None:
        path = self._path(asset, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)

    def read(self, asset: str, timeframe: str) -> pd.DataFrame | None:
        path = self._path(asset, timeframe)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def last_timestamp(self, asset: str, timeframe: str) -> pd.Timestamp | None:
        path = self._path(asset, timeframe)
        if not path.exists():
            return None
        df = pd.read_parquet(path, columns=[])  # index only, no column data
        if len(df.index) == 0:
            return None
        return df.index.max()

    def append(self, asset: str, timeframe: str, new_df: pd.DataFrame) -> None:
        if new_df.empty:
            return
        existing = self.read(asset, timeframe)
        if existing is None:
            self.write(asset, timeframe, new_df)
        else:
            combined = pd.concat([existing, new_df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
            self.write(asset, timeframe, combined)
