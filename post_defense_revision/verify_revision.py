from pathlib import Path
import hashlib, json

HERE=Path(__file__).resolve().parent
prov=json.loads((HERE/"provenance/revision_run_provenance.json").read_text(encoding="utf-8"))
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()
bad=[]
for name,expected in prov["protected_input_hashes_after"].items():
    p=Path(name)
    if not p.exists() or sha(p)!=expected: bad.append(name)
if bad: raise SystemExit("FAIL protected-input mismatch: "+", ".join(bad))
required=["R1_zero_missing/R1_zero_missing_summary.csv","R2_p99_robustness/R2_annual_station_wethour_p99.csv","R3_event_sensitivity/R3_scenario_comparison.csv","R4_spatial_interpolation/R4_spatial_CV_metrics.csv","POST_DEFENSE_REVISION_REPORT.md","REVISION_DECISION_LOG.md"]
missing=[x for x in required if not (HERE/x).is_file()]
if missing: raise SystemExit("FAIL missing outputs: "+", ".join(missing))
print("PASS: protected inputs unchanged and required revision outputs exist")
