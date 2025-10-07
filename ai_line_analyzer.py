# ai_line_analyzer.py
# All-in-one: odds ingestion -> SQLite -> consensus EV -> (optional) AI training -> AI EV -> reports
# iOS-friendly (GitHub Codespaces). No external scraping libs. Extend feature hooks as you go.

import os, sys, csv, math, json, sqlite3, datetime as dt
from collections import defaultdict
from typing import List, Dict, Any, Tuple

import requests
import numpy as np

# Try to import sklearn; if not available, we run without ML
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ====== USER CONFIG ===========================================================
API_KEY = "199208b47ace784eaebf10d81cb551f8"  # Replace later if you regenerate
REGION = "us"
MARKETS = ["h2h", "spreads", "totals"]  # extend later (first-half, etc., when your API tier allows)
OH_BOOKS = {
    "DraftKings", "FanDuel", "BetMGM", "Caesars", "BetRivers", "Bet365",
    "ESPN", "ESPN Bet", "Fanatics", "Hard Rock", "betJACK", "PointsBet"
}
BOOK_NORMALIZE = {
    "espn": "ESPN Bet","espn bet": "ESPN Bet","fanatics": "Fanatics",
    "draftkings":"DraftKings","fanduel":"FanDuel","betmgm":"BetMGM","caesars":"Caesars",
    "bet365":"Bet365","betrivers":"BetRivers","hard rock":"Hard Rock","betjack":"betJACK",
    "pointsbet":"PointsBet",
}
DB_PATH = "odds_ai.db"
DATE_STR = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
BASE = "https://api.the-odds-api.com/v4/sports"

# Output files
RAW_CSV = f"odds_raw_{DATE_STR}.csv"
EDGES_CSV = f"edges_ranked_{DATE_STR}.csv"
BEST_MD  = f"best_pick_{DATE_STR}.md"

# ====== UTILITIES =============================================================
def amer_to_decimal(american_odds: float) -> float:
    if american_odds is None: return None
    return 1.0 + (american_odds / 100.0) if american_odds > 0 else 1.0 + (100.0 / abs(american_odds))

def implied_from_decimal(o: float) -> float:
    return 0.0 if not o or o <= 0 else 1.0 / o

def vig_free_two_outcome(p1, p2):
    s = p1 + p2
    if s <= 0: return p1, p2
    return p1 / s, p2 / s

def vig_free_three_outcome(p1, p2, p3):
    s = p1 + p2 + p3
    if s <= 0: return p1, p2, p3
    return p1 / s, p2 / s, p3 / s

def ev_from_p_and_decimal(p: float, o: float) -> float:
    return p * (o - 1.0) - (1.0 - p)

def normalize_book_name(name: str) -> str:
    if not name: return ""
    key = name.lower().strip()
    for k, v in BOOK_NORMALIZE.items():
        if k in key: return v
    return name

def is_oh_book(name: str) -> bool:
    norm = normalize_book_name(name)
    return any(b.lower() == norm.lower() or b.lower() in norm.lower() for b in OH_BOOKS)

# ====== DATA ACCESS (ODDS API) ===============================================
def get_sports() -> List[Dict[str,Any]]:
    url = f"{BASE}?apiKey={API_KEY}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    js = r.json()
    if isinstance(js, dict) and js.get("message"):
        raise RuntimeError(js["message"])
    return js

def get_odds(sport_key: str, markets=MARKETS, region=REGION) -> List[Dict[str,Any]]:
    url = f"{BASE}/{sport_key}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": region,
        "markets": ",".join(markets),
        "oddsFormat": "american"
    }
    r = requests.get(url, params=params, timeout=45)
    if r.status_code != 200:
        print(f"[warn] odds fetch error for {sport_key}: {r.text[:200]}")
        return []
    return r.json()

# ====== STORAGE (SQLite) ======================================================
def ensure_schema(conn: sqlite3.Connection):
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS odds_snapshot (
        ts TEXT, sport_key TEXT, sport_title TEXT, event_id TEXT, commence_time TEXT,
        market TEXT, selection TEXT, point TEXT, book TEXT, price_american REAL,
        price_decimal REAL, implied_p_raw REAL
    );""")
    c.execute("""
    CREATE TABLE IF NOT EXISTS event_index (
        sport_key TEXT, event_id TEXT, home_team TEXT, away_team TEXT, commence_time TEXT,
        PRIMARY KEY (sport_key, event_id)
    );""")
    c.execute("""
    CREATE TABLE IF NOT EXISTS results (
        sport_key TEXT, event_id TEXT, winner TEXT,  -- 'home' or 'away'
        home_team TEXT, away_team TEXT,
        close_time TEXT,  -- optional
        UNIQUE(sport_key, event_id)
    );""")
    conn.commit()

def insert_raw_rows(conn, rows: List[List[Any]]):
    c = conn.cursor()
    c.executemany("""
    INSERT INTO odds_snapshot
    (ts,sport_key,sport_title,event_id,commence_time,market,selection,point,book,price_american,price_decimal,implied_p_raw)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
    conn.commit()

def upsert_event_index(conn, sport_key, event_id, home, away, commence):
    c = conn.cursor()
    c.execute("""
    INSERT INTO event_index (sport_key,event_id,home_team,away_team,commence_time)
    VALUES (?,?,?,?,?)
    ON CONFLICT(sport_key,event_id) DO UPDATE SET
      home_team=excluded.home_team,
      away_team=excluded.away_team,
      commence_time=excluded.commence_time
    """, (sport_key, event_id, home, away, commence))
    conn.commit()

# ====== FEATURE HOOKS (deep stats go here) ====================================
def feature_hooks_example(context: Dict[str,Any]) -> Dict[str,float]:
    """
    Extend this to add your own features:
      - injuries, rest days, back-to-back, travel distance, weather, pace, goalie, etc.
    For now, returns an empty dict (no additional features).
    The 'context' gives you:
      {
        'sport_key','event_id','home','away','commence_time',
        'market_key','point_key','sel','best_decimal','avg_implieds', ...
      }
    """
    return {}

# ====== INGEST + CONSENSUS EV =================================================
def ingest_today_and_compute():
    conn = sqlite3.connect(DB_PATH)
    ensure_schema(conn)

    sports = []
    try:
        sports = get_sports()
    except Exception as e:
        print("API error on get_sports:", e)
        sys.exit(1)

    target_sports = [s for s in sports if s.get("active")]
    raw_rows = []
    edges_rows = []

    for s in target_sports:
        sport_key = s.get("key")
        sport_title = s.get("title", sport_key)
        if not sport_key: continue

        data = get_odds(sport_key)
        if not data: continue

        for event in data:
            eid = event.get("id")
            commence = event.get("commence_time")
            home = event.get("home_team")
            away = event.get("away_team")
            upsert_event_index(conn, sport_key, eid, home, away, commence)

            bookmakers = event.get("bookmakers", [])
            # Filter to OH books + normalize names
            bm_filtered = []
            for b in bookmakers:
                bname = normalize_book_name(b.get("title",""))
                if is_oh_book(bname):
                    b["title"] = bname
                    bm_filtered.append(b)
            if not bm_filtered: continue

            market_snapshots = defaultdict(list)  # (mkt, point) -> list of dicts

            for bm in bm_filtered:
                bname = bm["title"]
                for mk in bm.get("markets", []):
                    mkt_key = mk.get("key")
                    if mkt_key not in MARKETS: continue
                    for outc in mk.get("outcomes", []):
                        sel = outc.get("name")
                        price = outc.get("price")
                        point = outc.get("point")
                        dec = amer_to_decimal(price)
                        if dec is None: continue
                        imp = implied_from_decimal(dec)

                        raw_rows.append([DATE_STR,sport_key,sport_title,eid,commence,mkt_key,sel,point,bname,price,dec,imp])
                        market_snapshots[(mkt_key, str(point))].append({"book": bname, "sel": sel, "dec": dec, "imp": imp})

            # consensus + EV
            for (mkt_key, point_key), snaps in market_snapshots.items():
                if not snaps: continue
                by_sel = defaultdict(list)
                for r in snaps: by_sel[r["sel"]].append(r)
                sels = list(by_sel.keys())
                if len(sels) < 2: continue

                avg_imp = {}
                for sel in sels:
                    imps = [x["imp"] for x in by_sel[sel] if x["imp"]>0]
                    if imps: avg_imp[sel] = sum(imps)/len(imps)
                if len(avg_imp) < 2: continue

                # vig removal
                if len(sels) == 2:
                    s1,s2 = sels[0], sels[1]
                    p1,p2 = vig_free_two_outcome(avg_imp.get(s1,0.0), avg_imp.get(s2,0.0))
                    fair_p = {s1:p1, s2:p2}
                elif len(sels) == 3:
                    s1,s2,s3 = sels[0], sels[1], sels[2]
                    p1,p2,p3 = vig_free_three_outcome(avg_imp.get(s1,0.0), avg_imp.get(s2,0.0), avg_imp.get(s3,0.0))
                    fair_p = {s1:p1, s2:p2, s3:p3}
                else:
                    tot = sum(avg_imp.values())
                    fair_p = {k:(v/tot if tot>0 else 0.0) for k,v in avg_imp.items()}

                # compute EV vs best OH price; also gather feature hook context
                for sel in sels:
                    if sel not in fair_p: continue
                    best_quote = max(by_sel[sel], key=lambda r: r["dec"])
                    o = best_quote["dec"]
                    p = fair_p[sel]
                    book_implied_p = 1.0/o if o>0 else 0.0
                    edge = p - book_implied_p
                    ev = ev_from_p_and_decimal(p, o)

                    # Feature hook context (can be used later by ML feature builder)
                    context = {
                        "sport_key":sport_key,"event_id":eid,"home":home,"away":away,
                        "commence_time":commence,"market_key":mkt_key,"point_key":point_key,
                        "sel":sel,"best_decimal":o,"avg_implieds":avg_imp
                    }
                    extra_feats = feature_hooks_example(context)  # returns dict

                    edges_rows.append([
                        DATE_STR, sport_key, s.get("title",""), eid, commence,
                        mkt_key, point_key, sel, best_quote["book"], o, p, book_implied_p, ev, edge,
                        json.dumps(extra_feats)  # store for later ML
                    ])

    # persist raw odds
    with sqlite3.connect(DB_PATH) as conn:
        insert_raw_rows(conn, raw_rows)

    # write CSVs for consensus
    with open(RAW_CSV,"w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts","sport_key","sport_title","event_id","commence_time","market","selection","point","book","price_american","price_decimal","implied_p_raw"])
        w.writerows(raw_rows)

    # Sort by consensus EV
    edges_sorted = sorted(edges_rows, key=lambda r: r[-3], reverse=True)  # EV column index -3
    with open(EDGES_CSV,"w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts","sport_key","sport_title","event_id","commence_time","market","point","selection","book_best_price","decimal_odds","fair_p_consensus","book_implied_p","EV_consensus","edge_consensus","extra_features_json"])
        w.writerows(edges_sorted)

    return edges_sorted

# ====== RESULTS IMPORT & ML TRAINING ==========================================
RESULTS_TEMPLATE = """# Save as results_import.csv and run: python ai_line_analyzer.py --import-results
# sport_key,event_id,winner,home_team,away_team,close_time
# winner is 'home' or 'away'
# example:
# americanfootball_nfl,1234-home-at-5678-away,home,Chicago Bears,Detroit Lions,2025-10-07T23:20:00Z
"""

def write_results_template():
    with open("results_import.csv","w") as f:
        f.write(RESULTS_TEMPLATE)
    print("Wrote results_import.csv template.")

def import_results_csv(path="results_import.csv"):
    if not os.path.exists(path):
        print("No results_import.csv found. Creating template...")
        write_results_template()
        return
    conn = sqlite3.connect(DB_PATH); ensure_schema(conn)
    c = conn.cursor()
    with open(path,"r") as f:
        reader = csv.DictReader(row for row in f if not row.startswith("#"))
        rows = []
        for r in reader:
            rows.append((r["sport_key"], r["event_id"], r["winner"], r["home_team"], r["away_team"], r.get("close_time","")))
    c.executemany("""
    INSERT OR REPLACE INTO results (sport_key,event_id,winner,home_team,away_team,close_time)
    VALUES (?,?,?,?,?,?)
    """, rows)
    conn.commit()
    print(f"Imported {len(rows)} results rows into SQLite.")

def build_training_table() -> Tuple[np.ndarray, np.ndarray, List[Dict[str,Any]]]:
    """
    Build simple training matrix from last odds snapshot per event (moneyline only).
    Label y = 1 if 'home' winner and selection==home ML, or 'away' winner and selection==away ML.
    Uses:
      - consensus fair_p,
      - book_implied_p (best price),
      - price decimal (value),
      - basic home/away indicator,
      - (optional) extra feature hooks if saved in JSON.
    """
    # Load latest consensus edges CSV for features (simplest path)
    if not os.path.exists(EDGES_CSV):
        print("No edges CSV from this run; run main ingest first.")
        return np.zeros((0,)), np.zeros((0,)), []

    # Only use H2H market for training a win/loss model
    rows = []
    with open(EDGES_CSV,"r") as f:
        rd = csv.DictReader(f)
        for r in rd:
            if r["market"] != "h2h": continue
            rows.append(r)

    # Join with results to get labels
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    X, y, meta = [], [], []
    for r in rows:
        sport_key = r["sport_key"]; eid = r["event_id"]
        c.execute("SELECT winner, home_team, away_team FROM results WHERE sport_key=? AND event_id=?", (sport_key, eid))
        got = c.fetchone()
        if not got: continue
        winner, home_team, away_team = got
        sel = r["selection"]
        # determine if selection maps to home or away
        if sel == home_team: sel_side = "home"
        elif sel == away_team: sel_side = "away"
        else:
            # Try simple contains match fallback
            sel_low = sel.lower()
            if home_team and home_team.lower() in sel_low: sel_side="home"
            elif away_team and away_team.lower() in sel_low: sel_side="away"
            else: continue

        label = 1 if sel_side == winner else 0

        fair_p = float(r["fair_p_consensus"])
        book_p = float(r["book_implied_p"])
        dec = float(r["decimal_odds"])
        ev = float(r["EV_consensus"])
        edge = float(r["edge_consensus"])
        # extra feature hooks
        extras = {}
        try:
            extras = json.loads(r.get("extra_features_json","{}")) or {}
        except Exception:
            extras = {}

        # Basic side indicator (home=1, away=0), can add sport one-hot & time to start
        side = 1.0 if sel_side=="home" else 0.0
        feat = [fair_p, book_p, dec, ev, edge, side]

        # Append any numeric extra features
        for k,v in sorted(extras.items()):
            try:
                feat.append(float(v))
            except Exception:
                pass

        X.append(feat)
        y.append(label)
        meta.append({
            "sport_key": sport_key, "event_id": eid, "home": home_team, "away": away_team, "sel_side": sel_side
        })

    return np.array(X, dtype=float), np.array(y, dtype=int), meta

def train_model(X: np.ndarray, y: np.ndarray):
    if not SKLEARN_AVAILABLE:
        print("scikit-learn not installed; skipping ML training.")
        return None
    if len(X) < 50 or len(set(y)) < 2:
        print("Not enough labeled samples to train (need >=50 and both classes).")
        return None
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(max_iter=200))
    ])
    pipe.fit(X, y)
    print(f"Model trained on {len(y)} samples.")
    return pipe

def predict_today_with_model(model, edges_rows):
    if (model is None) or (not SKLEARN_AVAILABLE):
        return []  # nothing to do
    preds = []
    for r in edges_rows:
        # r fields aligned to ingest_today_and_compute
        # [ts, sport_key, sport_title, event_id, commence, market, point, sel, book, o, p, book_p, ev, edge, extras_json]
        if r[5] != "h2h":  # model only for H2H in this starter
            continue
        try:
            fair_p = float(r[10]); book_p = float(r[11]); dec = float(r[9]); ev = float(r[12]); edge = float(r[13])
        except Exception:
            continue
        # Side indicator (home=1 if selection equals event_index.home_team)
        # Fetch home/away
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT home_team, away_team FROM event_index WHERE sport_key=? AND event_id=?", (r[1], r[3]))
        got = c.fetchone()
        side = 0.0
        if got:
            home, away = got
            if r[7] == home: side = 1.0
            else: side = 0.0

        feat = [fair_p, book_p, dec, ev, edge, side]
        # extras
        try:
            extras = json.loads(r[14]) or {}
            for k,v in sorted(extras.items()):
                feat.append(float(v))
        except Exception:
            pass

        X = np.array(feat, dtype=float).reshape(1,-1)
        p_home_side_wins = model.predict_proba(X)[0][1]  # prob(label==1)
        # If selection mapped to 'home' during training definition, this is directly p_model
        p_model = float(p_home_side_wins)
        # Re-map if our current selection corresponds to 'away' side:
        # We encoded side separately; our model targets "selected side wins" with side included,
        # so p_model is aligned with the selected side.

        # Model EV
        model_ev = ev_from_p_and_decimal(p_model, float(dec))
        model_edge = p_model - (1.0/float(dec))

        preds.append({
            "sport_key": r[1], "event_id": r[3], "market": r[5], "point": r[6], "selection": r[7],
            "book": r[8], "decimal_odds": float(dec),
            "p_consensus": fair_p, "EV_consensus": ev,
            "p_model": p_model, "EV_model": model_ev, "edge_model": model_edge
        })
    return preds

# ====== REPORTING =============================================================
def write_best_md(consensus_edges_sorted, model_preds):
    lines = []
    lines.append(f"# Best Picks — {DATE_STR}\n")

    if consensus_edges_sorted:
        best_ev = max(consensus_edges_sorted, key=lambda r: r[12])  # EV_consensus column
        best_edge = max(consensus_edges_sorted, key=lambda r: r[13])  # edge_consensus
        lines.append("## Consensus (vig-free market) signals\n")
        lines.append(f"- **Best EV**: {best_ev[1]} | {best_ev[2]} | {best_ev[7]} @ {best_ev[8]} (o={best_ev[9]:.2f}) — EV={best_ev[12]:.4f}\n")
        lines.append(f"- **Largest Edge**: {best_edge[1]} | {best_edge[2]} | {best_edge[7]} @ {best_edge[8]} (o={best_edge[9]:.2f}) — edge={best_edge[13]:.4f}\n")

    if model_preds:
        best_model_ev = max(model_preds, key=lambda x: x["EV_model"])
        best_model_edge = max(model_preds, key=lambda x: x["edge_model"])
        lines.append("\n## AI model signals (trained on your imported results)\n")
        lines.append(f"- **Best Model-EV**: {best_model_ev['sport_key']} | {best_model_ev['selection']} @ {best_model_ev['book']} (o={best_model_ev['decimal_odds']:.2f}) — EV_model={best_model_ev['EV_model']:.4f}, p_model={best_model_ev['p_model']:.3f}\n")
        lines.append(f"- **Largest Model Edge**: {best_model_edge['sport_key']} | {best_model_edge['selection']} @ {best_model_edge['book']} — edge_model={best_model_edge['edge_model']:.4f}, p_model={best_model_edge['p_model']:.3f}\n")

    with open(BEST_MD,"w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {BEST_MD}")

# ====== MAIN ==================================================================
def main():
    if not API_KEY or len(API_KEY) < 20:
        print("Put a valid API key into API_KEY.")
        sys.exit(1)

    # Step 1: ingest + consensus EV
    consensus_edges_sorted = ingest_today_and_compute()
    print(f"Wrote CSVs: {RAW_CSV}, {EDGES_CSV}")

    # Step 2: (optional) ML — load labels from results table and train
    X, y, meta = build_training_table()
    model = train_model(X, y) if (len(X) and SKLEARN_AVAILABLE) else None

    # Step 3: Predict with AI model for today's markets (H2H only in this starter)
    model_preds = predict_today_with_model(model, consensus_edges_sorted) if model else []

    # Step 4: Write summary
    write_best_md(consensus_edges_sorted, model_preds)

if __name__ == "__main__":
    # CLI convenience:
    #   python ai_line_analyzer.py --import-results    # import results_import.csv
    #   python ai_line_analyzer.py                     # normal run
    if len(sys.argv) > 1 and sys.argv[1] == "--import-results":
        import_results_csv()
    else:
        main()
