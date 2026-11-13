"""Streamlit front end for layers 1-3.  Run:  streamlit run app.py"""
import pandas as pd
import streamlit as st
from agents import Store, DeskManager, kpis

st.set_page_config(page_title="Disruption Agent - Layers 1-3", layout="wide")


@st.cache_resource
def load():
    s = Store.load()
    return s, DeskManager(s)


store, dm = load()
SN = store.profile["scenario_names"]

st.title("Supply chain disruption agent: layers 1-3")
st.caption(f"Team 10, IIM Udaipur. Synthetic pharma data, snapshot {store.snapshot:%d %b %Y}. Industry profile: {store.profile['industry']}.")

tab_case, tab_score, tab_about = st.tabs(["Run a case", "Lookout scorecard (80 alerts)", "How it works"])

with tab_case:
    al = store.alerts.sort_values("published_at", ascending=False)
    labels = {r.alert_id: f"{r.alert_id} | {r.published_at} | {r.source_type} | {r.headline[:80]}" for r in al.itertuples()}
    quick = st.radio("Quick picks", ["ALR-073 Hormuz (S1)", "ALR-056 FDA import alert (S2)", "ALR-064 Insulin excursion (S5)",
                                     "ALR-077 Hidden instruction", "ALR-049 Name look-alike", "Choose any alert"], horizontal=True)
    if quick == "Choose any alert":
        aid = st.selectbox("Alert", list(labels), format_func=lambda k: labels[k])
    else:
        aid = quick.split()[0]
    case = dm.run_case(aid)
    L, A, B = case["lookout"], case["analyst"], case["librarian"]

    st.subheader("Layer 1 - Lookout")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Decision", L["decision"])
    c2.metric("Scenario", f"{L['scenario']}" if L["scenario"] else "-", SN.get(L["scenario"], "") if L["scenario"] else None, delta_color="off")
    c3.metric("Confidence", L["confidence"] or "-")
    c4.metric("Suppliers linked", len(L["suppliers"]))
    st.markdown(f"**Alert:** {L['headline']}  \n_Source: {L['source']}, published {L['published_at']}_")
    for r in L["reasons"]:
        st.write("- " + r)
    if L["injection_flags"]:
        st.warning("Guardrail: instruction-like text was removed. Cleaned text passed on: " + L["cleaned_text"])
    if L["suppliers"]:
        st.dataframe(store.sup[store.sup.supplier_id.isin(L["suppliers"])][["supplier_id", "supplier_name", "country", "route", "chokepoint_exposure", "otif_rate"]],
                     hide_index=True, width="stretch")
    if case["stopped_reason"]:
        st.info("Desk Manager: " + case["stopped_reason"])

    if A:
        st.subheader("Layer 2 - Analyst")
        top = A["exposures"][0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Open orders hit", A["open_orders"], f"{A['overdue']} overdue", delta_color="inverse")
        c2.metric("Stock positions exposed", len(A["exposures"]))
        c3.metric("Highest risk score", f"{top['risk_score']:.0f}", top["score_band"], delta_color="off")
        c4.metric("Earliest run-out", min(e["runout_date"] for e in A["exposures"]))
        if A["excursion"]:
            x = A["excursion"]
            msg = (f"Excursion {x['temp_c']} C for {x['hours']} h vs limit {x['limit_temp_c']} C / {x['limit_hours']} h ({x['reference']}). ")
            (st.success if x["within_budget"] else st.error)(msg + ("Within the stability limit: release subject to QA review." if x["within_budget"] else "Outside the limit: lot to be rejected."))
        df = pd.DataFrame([{**{k: e[k] for k in ["plant", "material", "abc", "essential", "days_of_stock", "safety_stock_days", "runout_date", "in_transit_affected", "confirmed_replenishment", "predicted_delay_days", "risk_score", "score_band"]},
                            "products": ", ".join(p["product_name"] for p in e["products"])} for e in A["exposures"]])
        st.dataframe(df, hide_index=True, width="stretch")
        pick = st.selectbox("Explain the score for", range(len(A["exposures"])), format_func=lambda i: f"{A['exposures'][i]['plant']} - {A['exposures'][i]['material']}")
        e = A["exposures"][pick]; w = store.profile["analyst"]["weights"]
        st.dataframe(pd.DataFrame([dict(factor=k, value=v, weight=w[k], points=round(100 * w[k] * v, 1)) for k, v in e["factors"].items()]),
                     hide_index=True)
        if e["orders"]:
            st.caption("Orders behind this position")
            st.dataframe(pd.DataFrame(e["orders"]), hide_index=True, width="stretch")

    if B:
        st.subheader("Layer 3 - Librarian")
        for c in B["contracts"]:
            with st.expander(f"{c.get('contract_id', '?')} - {c['supplier_id']}", expanded=True):
                if c.get("error"):
                    st.error(c["error"]); continue
                if c.get("warning"):
                    st.warning(c["warning"])
                st.markdown(f"**1. Is the delay excused?** {c['excused']}  \n**2. Can we claim damages?** {c['damages']}  \n**3. Can we buy elsewhere?** {c['buy_elsewhere']}")
                for q in c["quotes"]:
                    st.caption(f"{q['ref']}: \"{q['text']}\"")
        st.markdown("**Most similar past cases**")
        st.dataframe(pd.DataFrame(B["past_cases"])[["incident_id", "date", "similarity", "what_happened", "action", "extra_cost_inr", "lesson"]],
                     hide_index=True, width="stretch")

    with st.expander("Audit log"):
        st.dataframe(pd.DataFrame(case["log"]), hide_index=True, width="stretch")

with tab_score:
    sc = dm.scorecard(); k = kpis(sc)
    st.caption("The Lookout reads all 80 synthetic alerts with the answer key hidden; its decisions are then compared with the key. "
               "The rules were tuned on these same alerts, so this is an optimistic first check, not a final test.")
    c = st.columns(6)
    c[0].metric("Precision", k["precision"]); c[1].metric("Recall", k["recall"]); c[2].metric("False alarms", k["false_alarms"])
    c[3].metric("Scenario accuracy", k["scenario_accuracy"]); c[4].metric("Supplier match", k["supplier_match"]); c[5].metric("Injections caught", k["injections_caught"])
    st.dataframe(sc, hide_index=True, width="stretch")

with tab_about:
    st.markdown("""
**Layer 1, Lookout:** rule-based checks for each source: PortWatch below 50% of normal, weather thresholds within 200 km of a supplier, FDA/CDSCO name matching (0.85 = confirmed, 0.5-0.85 = a person checks), and keyword and route matching for news and e-mails. Instruction-like text is removed first.

**Layer 2, Analyst:** links the alert to open orders (D4) and stock (D5). Days of stock = usable ÷ daily use. Risk score = 100 × (0.30 cover gap + 0.20 delay + 0.20 criticality + 0.15 concentration + 0.10 signal + 0.05 pressure). Cover gap = 1 − days of stock ÷ safety-stock days, or 0 if an unaffected delivery arrives before the run-out date.

**Layer 3, Librarian:** reads the contract terms (D3), answers the three questions with the clause quoted, computes damages, and ranks past incidents (D6) by text similarity plus same route, material and supplier.

**Desk Manager:** runs the layers in order, stops alert-only scenarios (S3, S4, S6) after layer 1, and keeps an audit log.

Every industry-specific setting is in `profiles/pharma.json`. A new industry needs a new profile, not new code.
""")
