"""Command-line run of layers 1-3 (no Streamlit needed).
   python run_case.py ALR-073         -> one case, printed
   python run_case.py --scorecard     -> Lookout KPIs on all 80 alerts
"""
import json, sys
from agents import Store, DeskManager, kpis

dm = DeskManager(Store.load())
if len(sys.argv) > 1 and sys.argv[1] == "--scorecard":
    print(json.dumps(kpis(dm.scorecard()), indent=2, default=str))
else:
    case = dm.run_case(sys.argv[1] if len(sys.argv) > 1 else "ALR-073")
    case.pop("alert")
    print(json.dumps(case, indent=2, default=str))
