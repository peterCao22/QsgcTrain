"""从 lgb_classifier.importance.csv 更新 feature_library.json。"""
import json
import pandas as pd
from datetime import date
from pathlib import Path

imp = pd.read_csv("models/saved/lgb_classifier.importance.csv")
imp = imp.rename(columns={"Unnamed: 0": "feature", "gain": "importance"})
imp = imp[["feature", "importance"]].sort_values("importance", ascending=False)

print("Feature importances:")
print(imp.to_string(index=False))

selected = imp[imp["importance"] > 0]["feature"].tolist()
print(f"\nAll {len(selected)} features with importance > 0 selected.")

lib = {
    "version": str(date.today()),
    "method": "lgb_gain_importance / target=is_strong_pos / all_37_features",
    "target": "is_strong_pos",
    "n_features": len(selected),
    "selected_features": selected,
    "feature_importance": dict(zip(imp["feature"], imp["importance"].astype(float))),
}

lib_path = Path("data/feature_library.json")
lib_path.parent.mkdir(parents=True, exist_ok=True)
with open(lib_path, "w", encoding="utf-8") as f:
    json.dump(lib, f, ensure_ascii=False, indent=2)
print(f"\nSaved: {lib_path}")
