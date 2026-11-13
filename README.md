# Disruption agent: working model for layers 1–3

Team 10 (Shivendra Yadav, Devang, Athak Sahni), PGP-GSCM, IIM Udaipur. Industry project.

The first three layers of our multi-agent system, running on our synthetic pharma data:

| Layer | Agent | What it does |
|---|---|---|
| 1 | Lookout | Reads an outside alert (PortWatch, weather, FDA, CDSCO, news, forwarder or e-mail). Decides whether it is a disruption, which scenario it is, and which suppliers it affects. Removes instruction-like text first. |
| 2 | Analyst | Links the alert to open orders (D4) and stock (D5): days of stock, run-out date, products affected and a 0–100 risk score. |
| 3 | Librarian | Reads the contracts (D3): is the delay excused, can we claim damages, can we buy elsewhere. Quotes each clause and finds the 3 most similar past incidents (D6). |
| – | Desk Manager | Runs the layers in order and stops alert-only scenarios (S3, S4, S6) after layer 1. Keeps an audit log. |

Everything specific to pharma (keywords, thresholds, weights) lives in `profiles/pharma.json`. Another industry needs a new profile, not new code.

## Run it on your laptop (5 minutes)

```bash
# Python 3.10+ needed
cd sc_agent
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_case.py ALR-073       # Hormuz case in the terminal (JSON)
python run_case.py --scorecard   # Lookout KPIs on all 80 labelled alerts
streamlit run app.py             # opens the app at http://localhost:8501
```

## Put it online for free (Streamlit Community Cloud)

1. Create a **public GitHub repository**, e.g. `team10-disruption-agent`, and upload the whole `sc_agent` folder contents: `app.py`, `agents.py`, `run_case.py`, `requirements.txt`, `profiles/` and `data/`.
2. Go to **share.streamlit.io** and sign in with GitHub.
3. Click **Create app**, choose **Deploy a public app from GitHub**, pick the repository and branch `main`, and set the main file path to `app.py`.
4. Click **Deploy**. The first build takes about 2–3 minutes. You get a link like `https://team10-disruption-agent.streamlit.app` to share with the professor.
5. If nobody opens a free app for several days, it goes to sleep. Open the link about 10 minutes before a demo and click "wake up" if asked.

Other free options: **Hugging Face Spaces** (choose the Streamlit SDK and upload the same files), or **Google Colab** for running `run_case.py` in a notebook.

## Demo script (about 5 minutes)

1. **ALR-073, Hormuz (S1).** Lookout: 2 transits vs 87, so S1 with 5 Gulf suppliers. Analyst: 8 orders at sea, all overdue; methanol at Ahmedabad has 11 days left (score 67, Amber); HDPE bottles score highest at 74 (Red) because they pack essential medicines. Librarian: CTR-011 is excused (war/blockade is force majeure); CTR-013 is not, so damages of about ₹1.07 lakh; past cases say a reroute adds 8–13 days and SUP-023 is the fastest fix.
2. **ALR-056, FDA import alert (S2).** Exact name match to SUP-021. Ceftriaxone has 10 days of usable stock (581 kg in quarantine), a single source and an essential medicine, so the score is 92 (Red).
3. **ALR-064, insulin at 9.6 °C for 3.5 h (S5).** Within the stability limit (25 °C, 24 h), so the lot is released after QA review. The Librarian notes that under CIP the transport risk is ours, so any claim goes to the carrier or insurer.
4. **ALR-077, hidden instruction.** "IGNORE ALL PREVIOUS INSTRUCTIONS…" is removed, and the delay notice is still used.
5. **ALR-049, name look-alike.** "Vyomika Pharma" is only a partial match, so a person must check, and SUP-001 is not blocked.
6. **Scorecard tab.** Precision 1.0, recall 1.0, 3/3 injections caught. **Caveat:** the rules were tuned on these same synthetic alerts, so this is a first check, not a final test.

## Known limits (be upfront about these)

- The rules and data are synthetic. Live feeds (`integration/connectors.py`) are not wired in yet; the demo replays D7.
- The delay estimate is a simple rule (how late the order already is, or the average delay on that route since the event). The XGBoost model replaces it in Milestone 2.
- Some synthetic contracts have already expired on the snapshot date. The Librarian flags this; the data fix is on our list.
- No LLM is used yet: the logic is fully traceable. An LLM can be added later to write the case summary.
