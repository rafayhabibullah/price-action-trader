import pandas as pd
from pathlib import Path


class ResultsReporter:
    def __init__(self, reports_dir: Path):
        self.reports_dir = Path(reports_dir)

    def save(self, results: pd.DataFrame, run_id: str) -> Path:
        out_dir = self.reports_dir / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / "leaderboard.csv"
        results.to_csv(csv_path, index=False)

        html_path = out_dir / "leaderboard.html"
        html = results.to_html(index=False, float_format="%.4f")
        styled = f"""<!DOCTYPE html>
<html><head><style>
  body {{ font-family: monospace; padding: 20px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 12px; text-align: right; }}
  th {{ background: #222; color: #fff; }}
  tr:nth-child(even) {{ background: #f5f5f5; }}
</style></head><body>
<h2>Price Action Trader — Leaderboard ({run_id})</h2>
{html}
</body></html>"""
        html_path.write_text(styled)

        return out_dir

    def top(self, results: pd.DataFrame, n: int = 20) -> pd.DataFrame:
        return results.sort_values("score", ascending=False).head(n).reset_index(drop=True)
