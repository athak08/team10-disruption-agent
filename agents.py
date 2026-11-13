"""
Layers 1-3 of the Team 10 multi-agent system (IIM Udaipur GSCM industry project).

  Layer 1  Lookout    - reads one outside alert, decides if it is a disruption, which
                        scenario, which suppliers. Text is information only (injection guard).
  Layer 2  Analyst    - links the alert to open orders and stock, computes days of stock,
                        run-out date, products affected and a 0-100 risk score.
  Layer 3  Librarian  - answers: is the delay excused? can we claim damages? can we buy
                        elsewhere? - quoting the contract clause - and finds similar past cases.
  DeskManager         - runs the layers in order, keeps the case file and audit log, and
                        stops alert-only scenarios (S3, S4, S6) after layer 1.

Everything industry-specific (keywords, thresholds, weights) is read from profiles/<industry>.json.
Deterministic, rule-based and free to run. An LLM can be added later for wording only.
"""
from __future__ import annotations
import json, math, re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"


# --------------------------------------------------------------------------- data
@dataclass
class Store:
    sup: pd.DataFrame; mat: pd.DataFrame; ctr: pd.DataFrame; po: pd.DataFrame
    stock: pd.DataFrame; inc: pd.DataFrame; alerts: pd.DataFrame; stab: pd.DataFrame; bom: pd.DataFrame
    profile: dict

    @classmethod
    def load(cls, data_dir: Path = DATA, profile: str = "pharma") -> "Store":
        r = lambda f: pd.read_csv(data_dir / f, keep_default_na=False, na_values=[""])
        prof = json.loads((ROOT / "profiles" / f"{profile}.json").read_text())
        return cls(r("D1_supplier_master.csv"), r("D2_material_master.csv"), r("D3_contracts_sla.csv"),
                   r("D4_purchase_orders_shipments.csv"), r("D5_stock_snapshot_2026-10-20.csv"),
                   r("D6_incident_response_log.csv"), r("D7_outside_alert_feed.csv"),
                   r("D9_cold_chain_stability.csv"), r("D10_product_material_bom.csv"), prof)

    @property
    def snapshot(self) -> date:
        return date.fromisoformat(self.profile["snapshot_date"])

    def alert_for_agents(self, alert_id: str) -> dict:
        """The Lookout never sees the answer key (gt_* columns)."""
        row = self.alerts.loc[self.alerts.alert_id == alert_id].iloc[0].to_dict()
        return {k: v for k, v in row.items() if not k.startswith("gt_")}


def _d(x):
    if x is None or (isinstance(x, float) and math.isnan(x)) or x == "":
        return None
    return date.fromisoformat(str(x)[:10])


def _norm_name(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(pvt|ltd|co|fze|llc|inc|gmbh|s\.?p\.?a|limited|company|plant|pli|f\.z\.e)\b\.?", " ", s)
    return re.sub(r"[^a-z0-9]", "", s)


def name_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm_name(a), _norm_name(b)).ratio()


def haversine_km(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    a = 0.5 - math.cos((lat2 - lat1) * p) / 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * (1 - math.cos((lon2 - lon1) * p)) / 2
    return 12742 * math.asin(math.sqrt(a))


# --------------------------------------------------------------------------- layer 1
class Lookout:
    def __init__(self, store: Store):
        self.s = store; self.cfg = store.profile["lookout"]

    def _guard(self, text: str):
        """Prompt-injection guard: remove instruction-like sentences, keep the facts."""
        flags = []
        clean = text
        for pat in self.cfg["injection_patterns"]:
            if re.search(pat, clean, flags=re.I):
                flags.append(pat)
                clean = re.sub(r"[^.;:-]*(" + pat + r")[^.;]*", " [instruction removed] ", clean, flags=re.I)
        return clean, flags

    def _suppliers_by_material(self, mids):
        return set(self.s.sup.loc[self.s.sup.materials_supplied.apply(lambda m: any(x in m.split(";") for x in mids)), "supplier_id"])

    def _materials_in(self, text):
        t = text.lower()
        return [m for m, kws in self.cfg["material_keywords"].items() if any(k in t for k in kws)]

    def _named_suppliers(self, text):
        t = text.lower(); out = set()
        for r in self.s.sup.itertuples():
            first_two = " ".join(r.supplier_name.lower().split()[:2]).rstrip(".,")
            if first_two in t:
                out.add(r.supplier_id)
        return out

    def run(self, alert: dict) -> dict:
        cfg = self.cfg; sup = self.s.sup
        raw = f"{alert.get('headline', '')}. {alert.get('body', '')}"
        text, inj = self._guard(raw)
        tl = text.lower()
        res = dict(alert_id=alert["alert_id"], published_at=alert["published_at"], source=alert["source_type"],
                   headline=alert["headline"], decision="NO_ALERT", scenario=None, confidence=None,
                   suppliers=[], materials=[], reasons=[], injection_flags=inj, cleaned_text=text)
        if inj:
            res["reasons"].append("Instruction-like text found and removed; the message is treated as information only.")
        if any(k in tl for k in cfg["out_of_scope_keywords"]):
            res.update(decision="BLOCKED"); res["reasons"].append("Out of scope (price negotiation / ERP edit): passed to people, no action.")
            res["suppliers"] = sorted(self._named_suppliers(raw)); return res
        neg = [k for k in cfg["negative_keywords"] if k in tl]
        src = alert["source_type"]

        if src == "PortWatch":
            v, b = float(alert["metric_value"]), float(alert["baseline_value"])
            ratio = v / b
            choke = next((c for c, kws in cfg["chokepoint_keywords"].items() if any(k in str(alert["location"]).lower() for k in kws)), None)
            res["reasons"].append(f"{alert['location']}: {v:.0f} transits vs normal {b:.0f} ({ratio:.0%}).")
            if choke and (ratio < cfg["portwatch_alert_ratio"] or (ratio < cfg["portwatch_probable_ratio"] and any(k in tl for k in ["suspension", "typhoon", "closure"]))):
                conf = "confirmed" if ratio < cfg["portwatch_alert_ratio"] else "probable"
                ids = sup.loc[sup.chokepoint_exposure == choke, "supplier_id"].tolist()
                res.update(decision="ALERT", scenario="S1", confidence=conf, suppliers=ids)
                res["reasons"].append(f"Below the {cfg['portwatch_alert_ratio']:.0%} rule" if conf == "confirmed" else "Between 50% and 60% of normal, with a closure/suspension reported")
            else:
                res["reasons"].append("Within normal range - no alert.")
            return res

        if src == "Weather":
            metric = alert["metric_name"]; v = float(alert["metric_value"])
            thr = cfg["weather_thresholds"].get(metric)
            res["reasons"].append(f"{alert['location']}: {metric} = {v:g}" + (f" (threshold {thr})" if thr else " (no alert threshold for this metric)"))
            if thr is not None and v >= thr:
                lat, lon = float(alert["lat"]), float(alert["lon"])
                near = sup[[haversine_km(lat, lon, a, b) <= cfg["weather_radius_km"] for a, b in zip(sup.site_lat, sup.site_lon)]]
                res.update(decision="ALERT", scenario="S4", confidence="confirmed", suppliers=near.supplier_id.tolist())
                res["reasons"].append(f"{len(near)} supplier sites within {cfg['weather_radius_km']} km.")
            return res

        if src in ("FDA", "CDSCO"):
            ent = alert.get("entity_name_in_feed"); ent = "" if ent is None or (isinstance(ent, float) and math.isnan(ent)) else str(ent)
            mids = self._materials_in(text)
            if ent:
                scores = sup.supplier_name.apply(lambda n: name_similarity(ent, n))
                i = scores.idxmax(); best = float(scores[i]); sid = sup.supplier_id[i]
                if best >= cfg["name_match_confirmed"]:
                    res.update(decision="ALERT", scenario="S2", confidence="confirmed", suppliers=[sid])
                    res["reasons"].append(f"'{ent}' matches {sid} ({sup.supplier_name[i]}), score {best:.2f}.")
                elif best >= cfg["name_match_possible"]:
                    res.update(decision="CHECK", scenario="S2", confidence="possible", suppliers=[sid])
                    res["reasons"].append(f"'{ent}' is only a partial match to {sid} ({sup.supplier_name[i]}), score {best:.2f}: a person must confirm before any action.")
                else:
                    res["reasons"].append(f"'{ent}' does not match any of our suppliers (best score {best:.2f}).")
            elif "shortage" in tl and mids and not neg:
                ids = sorted(self._suppliers_by_material(mids))
                res.update(decision="ALERT", scenario="S6", confidence="confirmed", suppliers=ids, materials=mids)
                res["reasons"].append("Shortage listing for a molecule we buy.")
            elif neg:
                res["reasons"].append(f"Negative/resolved wording: {neg}.")
            res["materials"] = res["materials"] or mids
            return res

        # News, trade notices, forwarder and supplier e-mails
        if neg:
            res["reasons"].append(f"Negative/resolved wording: {neg} - no alert."); return res
        scen = next((sc for sc, kws in cfg["scenario_keywords"].items() if any(k in tl for k in kws)), None)
        named = self._named_suppliers(raw)
        choke_hit = next((c for c, kws in cfg["chokepoint_keywords"].items() if any(k in tl for k in kws)), None)
        if scen is None and choke_hit:
            scen = "S1"; res["reasons"].append(f"Mentions a port or chokepoint on our {choke_hit} route.")
        if scen is None and named and "delay" in tl:
            ex = sup.loc[sup.supplier_id.isin(named), "chokepoint_exposure"].tolist()
            scen = "S1" if "Hormuz" in ex else None
            if scen: res["reasons"].append("Delay notice from a supplier on the Hormuz route.")
        if scen is None:
            res["reasons"].append("No disruption keywords for our routes, regions or materials."); return res
        mids = self._materials_in(text)
        ids = set()
        choke = next((c for c, kws in cfg["chokepoint_keywords"].items() if any(k in tl for k in kws)), None)
        if choke:
            ids |= set(sup.loc[sup.chokepoint_exposure == choke, "supplier_id"])
            ports = [p for p in sup.origin_port.unique() if str(p).lower() in tl]
            if ports:
                ids &= set(sup.loc[sup.origin_port.isin(ports), "supplier_id"])
        regions = [v for k, v in cfg["region_keywords"].items() if re.search(r"\b" + re.escape(k) + r"\b", tl)]
        if regions and not choke:
            reg = set(sup.loc[sup.apply(lambda r: any(g in r.site_city or g == r.country for g in regions), axis=1), "supplier_id"])
            ids = ids | reg if not ids else ids & reg
        if mids:
            by_mat = self._suppliers_by_material(mids)
            ids = (ids & by_mat) if ids else by_mat
        if named:
            ids = (ids & named) or named
        res.update(decision="ALERT" if ids else "CHECK", scenario=scen, confidence="confirmed" if src in ("Forwarder", "Supplier e-mail") else "probable",
                   suppliers=sorted(ids), materials=mids)
        res["reasons"].append(f"Keywords point to {scen} ({self.s.profile['scenario_names'][scen]})" + (f"; chokepoint {choke}" if choke else "") + (f"; materials {mids}" if mids else ""))
        if not ids:
            res["reasons"].append("Could not link to a specific supplier - a person should check.")
        return res


# --------------------------------------------------------------------------- layer 2
class Analyst:
    def __init__(self, store: Store):
        self.s = store; self.cfg = store.profile["analyst"]

    def _hist_delay(self, choke):
        since = self.cfg["historical_delay_by_chokepoint_since"].get(choke)
        po = self.s.po
        m = (po.chokepoint == choke) & (po.status == "Delivered")
        if since:
            m &= po.order_date > since
        d = pd.to_numeric(po.loc[m, "delay_days"], errors="coerce")
        return float(d.mean()) if len(d) else 0.0

    def run(self, look: dict) -> dict:
        s, cfg, snap = self.s, self.cfg, self.s.snapshot
        affected = set(look["suppliers"]); scen = look["scenario"]
        po = s.po
        open_po = po[po.supplier_id.isin(affected) & po.status.isin(["In transit", "Open (not shipped)"])].copy()
        open_po["promised"] = open_po.promised_date.map(_d)
        open_po["days_overdue"] = [(snap - p).days if p and p < snap else 0 for p in open_po.promised]
        # which plant x material rows are exposed
        rows = set(zip(open_po.plant, open_po.material_id))
        excursion = None
        if scen == "S2":   # regulatory hold: every material the supplier provides is exposed
            for sid in affected:
                for m in s.sup.loc[s.sup.supplier_id == sid, "materials_supplied"].iloc[0].split(";"):
                    for p in s.stock.loc[s.stock.material_id == m, "plant"]:
                        rows.add((p, m))
        if scen == "S5":
            rows = {(p, m) for (p, m) in rows if m in set(look.get("materials") or [])} or \
                   {(p, m) for m in (look.get("materials") or []) for p in s.stock.loc[s.stock.material_id == m, "plant"]}
            mt = re.search(r"(\d+(?:\.\d+)?)\s*C for (\d+(?:\.\d+)?)\s*(h|min)", look["headline"] + " " + look.get("cleaned_text", ""))
            if mt and look.get("materials"):
                temp = float(mt.group(1)); hrs = float(mt.group(2)) / (60 if mt.group(3) == "min" else 1)
                st = s.stab.loc[s.stab.material_id == look["materials"][0]]
                if len(st):
                    st = st.iloc[0]
                    within = temp <= st.max_excursion_temp_c and hrs <= st.max_cumulative_hours
                    excursion = dict(material_id=st.material_id, temp_c=temp, hours=hrs, limit_temp_c=float(st.max_excursion_temp_c),
                                     limit_hours=float(st.max_cumulative_hours), within_budget=bool(within), rule=st.disposition_rule,
                                     reference=st.stability_reference)
        choke = s.sup.loc[s.sup.supplier_id.isin(affected), "chokepoint_exposure"].mode()
        choke = choke.iloc[0] if len(choke) else None
        hist = self._hist_delay(choke) if scen == "S1" and choke else 0.0
        w = cfg["weights"]; sig = cfg["signal_points"].get(look.get("confidence") or "probable", 0.6)
        pressure = max(0.0, min(1.0, (cfg["gscpi_assumed"] + 1) / 4))
        out = []
        for plant, mid in sorted(rows):
            st = s.stock[(s.stock.plant == plant) & (s.stock.material_id == mid)]
            if st.empty: continue
            st = st.iloc[0]; m = s.mat.loc[s.mat.material_id == mid].iloc[0]
            dos = float(st.days_of_stock); runout = snap + timedelta(days=dos)
            mine = open_po[(open_po.plant == plant) & (open_po.material_id == mid)]
            # confirmed replenishment = in-transit order from an UNAFFECTED supplier arriving before run-out
            other = po[(po.plant == plant) & (po.material_id == mid) & (po.status == "In transit") & ~po.supplier_id.isin(affected)]
            confirmed = [e for e in other.eta_current.map(_d) if e and e <= runout]
            cover_gap = 0.0 if confirmed else max(0.0, 1 - dos / float(st.safety_stock_days))
            if scen == "S1":
                pred = max([*mine.days_overdue.tolist(), hist]) if len(mine) else hist
            elif scen == "S2":
                pred = cfg["delay_cap_days"]          # supply blocked until the regulatory issue is resolved
            else:
                pred = 0.0
            delay = min(1.0, pred / cfg["delay_cap_days"])
            crit = min(1.0, cfg["abc_points"][m.abc_criticality] + (cfg["essential_bonus"] if m.essential_medicine_link == "Y" else 0))
            n = int(m.n_approved_suppliers); conc = cfg["concentration_points"]["1" if n == 1 else "2" if n == 2 else "3+"]
            factors = dict(cover_gap=cover_gap, delay=delay, criticality=crit, concentration=conc, signal=sig, pressure=pressure)
            score = round(100 * sum(w[k] * v for k, v in factors.items()), 1)
            band = "Green" if score < cfg["tier_bands"]["green_below"] else "Red" if score > cfg["tier_bands"]["red_above"] else "Amber"
            prods = s.bom.loc[(s.bom.material_id == mid) & s.bom.plants.str.contains(plant), ["product_name", "nlem_essential"]].drop_duplicates()
            out.append(dict(plant=plant, material_id=mid, material=m.material_name, abc=m.abc_criticality, essential=m.essential_medicine_link,
                            usable=float(st.usable_qty), quarantine=float(st.quarantine_qty), uom=st.uom, daily_use=float(st.avg_daily_use),
                            days_of_stock=dos, safety_stock_days=int(st.safety_stock_days), runout_date=runout.isoformat(),
                            in_transit_affected=float(mine[mine.status == "In transit"].qty_ordered.sum()),
                            confirmed_replenishment=bool(confirmed), orders=mine[["po_id", "supplier_id", "status", "qty_ordered", "po_value_inr", "promised_date", "eta_current", "days_overdue"]].to_dict("records"),
                            predicted_delay_days=round(pred, 1), factors={k: round(v, 3) for k, v in factors.items()}, risk_score=score, score_band=band,
                            products=prods.to_dict("records")))
        out.sort(key=lambda r: -r["risk_score"])
        return dict(exposures=out, open_orders=len(open_po), in_transit=int((open_po.status == "In transit").sum()),
                    overdue=int((open_po.days_overdue > 0).sum()), historical_delay_days=round(hist, 1), excursion=excursion,
                    case_score=out[0]["risk_score"] if out else 0, case_band=out[0]["score_band"] if out else "Green")


# --------------------------------------------------------------------------- layer 3
STOP = set("the a an of to and or for in on at by with from is are was were be this that it as our we any no not per into via after".split())


def _tokens(t):
    return [w for w in re.findall(r"[a-z0-9\-]+", str(t).lower()) if w not in STOP and len(w) > 2]


def _cos(a, b):
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    num = sum(ca[k] * cb[k] for k in ca)
    den = math.sqrt(sum(v * v for v in ca.values())) * math.sqrt(sum(v * v for v in cb.values()))
    return num / den if den else 0.0


class Librarian:
    def __init__(self, store: Store):
        self.s = store; self.cfg = store.profile["librarian"]

    def contract_answers(self, sid: str, scen: str, orders: list) -> dict:
        s, snap = self.s, self.s.snapshot
        c = s.ctr.loc[s.ctr.supplier_id == sid]
        if c.empty:
            return dict(supplier_id=sid, error="No contract found - the Librarian will not guess. Ask the category team.")
        c = c.iloc[0]; cid = c.contract_id
        warn = None
        if _d(c.end_date) and _d(c.end_date) < snap:
            warn = f"Contract {cid} ended on {c.end_date}; its terms may no longer apply - check whether it was renewed."
        fm_col = self.cfg["fm_category_by_scenario"].get(scen)
        if scen == "S5":
            return dict(supplier_id=sid, contract_id=cid, warning=warn,
                        excused="Not a supplier delay - this is a temperature excursion in transit. Under the " + str(c.incoterm) +
                                " incoterm the risk passed to us when the goods were handed to the carrier, so any loss is claimed from the carrier or insurer, not the supplier.",
                        damages="No supplier damages apply.", damages_inr=0,
                        buy_elsewhere="Only if QA rejects the lot - then a replacement order is needed.",
                        quotes=[dict(ref=f"{cid}, Clause 16.1", text=c.cl_16_quality.split(" 16.2")[0])])
        if fm_col:
            excused = c[fm_col] == "Y"
            q1 = ("Yes - the event is listed as force majeure" if excused else "No - the event is not in the force-majeure list") + \
                 f" (provided the supplier gave notice within {int(c.fm_notice_days)} days)." if excused else \
                 ("No - the event is not in the force-majeure list, so the delay is not excused.")
            q1_quote = ("Clause 14.1", c.cl_14_force_majeure.split(" 14.2")[0])
        else:
            excused = False
            q1 = "No - a quality or regulatory failure at the supplier's site is not force majeure; the supplier must notify us within 48 hours."
            q1_quote = ("Clause 16.1", c.cl_16_quality.split(" 16.2")[0])
        late = [o for o in orders if o["supplier_id"] == sid and o["days_overdue"] > 0]
        if excused:
            q2, ld = "No - the delay is excused, so liquidated damages do not apply.", 0.0
        elif late:
            ld = 0.0; parts = []
            for o in late:
                weeks = o["days_overdue"] // 7
                pct = min(weeks * float(c.ld_pct_per_week), float(c.ld_cap_pct))
                amt = pct * float(o["po_value_inr"]); ld += amt
                parts.append(f"{o['po_id']}: {weeks} full weeks x {float(c.ld_pct_per_week):.1%} = {pct:.1%} of Rs {float(o['po_value_inr']):,.0f} = Rs {amt:,.0f}")
            q2 = f"Yes - about Rs {ld:,.0f} so far (cap {float(c.ld_cap_pct):.0%} of order value). " + "; ".join(parts)
        else:
            q2, ld = "Not yet - no order from this supplier is past its promised date.", 0.0
        q2_quote = ("Clause 12", c.cl_12_liquidated_damages)
        if str(c.right_to_buy_elsewhere).startswith("Y"):
            trig = int(float(c.alt_sourcing_trigger_days)); worst = max([o["days_overdue"] for o in orders if o["supplier_id"] == sid] + [0])
            q3 = f"Yes - now (an order is {worst} days late, trigger is {trig} days)." if worst >= trig else f"Yes, once an order is {trig} days late."
        else:
            q3 = "No - this contract is exclusive; buying elsewhere needs the supplier's written consent."
        return dict(supplier_id=sid, contract_id=cid, warning=warn, excused=q1, damages=q2, damages_inr=round(ld), buy_elsewhere=q3,
                    quotes=[dict(ref=f"{cid}, {q1_quote[0]}", text=q1_quote[1]), dict(ref=f"{cid}, {q2_quote[0]}", text=q2_quote[1]),
                            dict(ref=f"{cid}, Clause 15", text=c.cl_15_alternate_sourcing)])

    def past_cases(self, look: dict, ana: dict) -> list:
        s = self.s; inc = s.inc
        mids = {e["material_id"] for e in ana["exposures"][:3]} | set(look.get("materials") or [])
        chokes = set(s.sup.loc[s.sup.supplier_id.isin(look["suppliers"]), "chokepoint_exposure"])
        q = _tokens(look["headline"] + " " + look.get("cleaned_text", "") + " " + " ".join(e["material"] for e in ana["exposures"][:3]))
        cand = inc[inc.scenario == look["scenario"]] if look["scenario"] else inc
        scored = []
        for r in cand.itertuples():
            sc = _cos(q, _tokens(f"{r.description} {r.action_taken} {r.lesson_learned}"))
            sc += 0.3 * (r.chokepoint in chokes) + 0.2 * (r.material_id in mids) + 0.1 * (r.supplier_id in look["suppliers"])
            scored.append((round(sc, 3), r.detected_date, r))
        scored.sort(key=lambda x: (-x[0], x[1]), reverse=False)
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [dict(incident_id=r.incident_id, date=r.detected_date, similarity=sc, supplier_id=r.supplier_id, material_id=r.material_id,
                     what_happened=r.description, action=r.action_taken, extra_cost_inr=int(r.extra_cost_inr), tier=r.approval_tier,
                     lesson=r.lesson_learned) for sc, _, r in scored[: self.cfg["past_cases"]]]

    def run(self, look: dict, ana: dict) -> dict:
        orders = [o for e in ana["exposures"] for o in e["orders"]]
        sids = sorted({o["supplier_id"] for o in orders} or set(look["suppliers"]))
        return dict(contracts=[self.contract_answers(sid, look["scenario"], orders) for sid in sids],
                    past_cases=self.past_cases(look, ana))


# --------------------------------------------------------------------------- supervisor
class DeskManager:
    def __init__(self, store: Store):
        self.s = store
        self.lookout, self.analyst, self.librarian = Lookout(store), Analyst(store), Librarian(store)

    def run_case(self, alert_id: str) -> dict:
        log = []
        stamp = lambda layer, msg: log.append(dict(time=datetime.now().strftime("%H:%M:%S"), layer=layer, event=msg))
        alert = self.s.alert_for_agents(alert_id)
        stamp("Desk Manager", f"Case opened for {alert_id} (answer key hidden from agents)")
        look = self.lookout.run(alert)
        stamp("Lookout", f"{look['decision']} - scenario {look['scenario']} - {len(look['suppliers'])} suppliers")
        case = dict(alert=alert, lookout=look, analyst=None, librarian=None, log=log, stopped_reason=None)
        full = self.s.profile["full_flow_scenarios"]
        if look["decision"] != "ALERT":
            case["stopped_reason"] = {"NO_ALERT": "No disruption - case closed.", "CHECK": "Uncertain match - passed to a person to confirm.",
                                      "BLOCKED": "Out of scope - passed to people."}[look["decision"]]
        elif look["scenario"] not in full:
            case["stopped_reason"] = f"{look['scenario']} is alert-only in the prototype - alert sent to the buyer, case stops after layer 1."
        if case["stopped_reason"]:
            stamp("Desk Manager", case["stopped_reason"]); return case
        case["analyst"] = self.analyst.run(look)
        a = case["analyst"]
        stamp("Analyst", f"{len(a['exposures'])} stock positions exposed; top risk score {a['case_score']} ({a['case_band']})")
        if not a["exposures"]:
            case["stopped_reason"] = "No open orders or stock linked to this alert - case closed."; stamp("Desk Manager", case["stopped_reason"]); return case
        case["librarian"] = self.librarian.run(look, a)
        stamp("Librarian", f"{len(case['librarian']['contracts'])} contracts read; {len(case['librarian']['past_cases'])} similar past cases")
        stamp("Desk Manager", "Layers 1-3 complete - ready for the Advisor (layer 4, next milestone)")
        return case

    def scorecard(self) -> pd.DataFrame:
        rows = []
        for aid in self.s.alerts.alert_id:
            look = self.lookout.run(self.s.alert_for_agents(aid))
            gt = self.s.alerts.loc[self.s.alerts.alert_id == aid].iloc[0]
            g = set(filter(None, str(gt.gt_matched_supplier_ids).split(";"))) if gt.gt_matched_supplier_ids else set()
            p = set(look["suppliers"]) if look["decision"] == "ALERT" else set()
            rows.append(dict(alert_id=aid, source=look["source"], headline=look["headline"][:90], decision=look["decision"],
                             predicted_scenario=look["scenario"] if look["decision"] == "ALERT" else "", gt_genuine=gt.gt_is_genuine_disruption,
                             gt_scenario=gt.gt_scenario, supplier_jaccard=round(len(p & g) / len(p | g), 2) if (p | g) else 1.0,
                             injection_caught=bool(look["injection_flags"]) if gt.contains_prompt_injection == "Y" else None))
        return pd.DataFrame(rows)


def kpis(sc: pd.DataFrame) -> dict:
    alert = sc.decision == "ALERT"; gen = sc.gt_genuine == "Y"
    tp = int((alert & gen).sum()); fp = int((alert & ~gen).sum()); fn = int((~alert & gen).sum())
    scen_ok = int(((sc.predicted_scenario == sc.gt_scenario) & alert & gen).sum())
    inj = sc.injection_caught.dropna()
    return dict(alerts_tested=len(sc), true_alerts=tp, false_alarms=fp, missed=fn,
                precision=round(tp / (tp + fp), 2) if tp + fp else None, recall=round(tp / (tp + fn), 2) if tp + fn else None,
                scenario_accuracy=round(scen_ok / tp, 2) if tp else None,
                supplier_match=round(sc.loc[alert & gen, "supplier_jaccard"].mean(), 2) if tp else None,
                injections_caught=f"{int(inj.sum())}/{len(inj)}")
