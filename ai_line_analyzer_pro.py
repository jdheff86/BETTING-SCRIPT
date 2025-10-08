# ai_line_analyzer_pro.py
# Deep odds analyzer: consensus EV, AI model, locks, arbitrage, middles, Kelly, feature hooks.
# Now supports exact local date (--date YYYY-MM-DD) and multi-day windows (--window-days N).

import os, sys, csv, math, json, sqlite3, datetime as dt
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from zoneinfo import ZoneInfo
import argparse

import requests
import numpy as np

# ------- OPTIONAL ML -------
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ===== USER CONFIG ============================================================
API_KEY = "199208b47ace784eaebf10d81cb551f8"   # your key (regenerate later for safety)
REGION = "us"
MARKETS = ["h2h", "spreads", "totals"]
LOCKS_TOP_N = 5
DB_PATH = "odds_ai.db"
TIMEZONE = "America/New_York"
WINDOW_DAYS = 1   # default: only consider events starting today; override via --window-days
DATE_STR = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
BASE = "https://api.the-odds-api.com/v4/sports"

# Ohio book list (fuzzy matched)
OH_BOOKS = {
    "DraftKings","FanDuel","BetMGM","Caesars","BetRivers","Bet365",
    "ESPN Bet","Fanatics","Hard Rock","betJACK","PointsBet"
}
BOOK_NORMALIZE = {
    "espn":"ESPN Bet","espn bet":"ESPN Bet","fanatics":"Fanatics",
    "draftkings":"DraftKings","fanduel":"FanDuel","betmgm":"BetMGM",
    "caesars":"Caesars","bet365":"Bet365","betrivers":"BetRivers",
    "hard rock":"Hard Rock","betjack":"betJACK","pointsbet":"PointsBet",
}

# Output files
RAW_CSV   = f"odds_raw_{DATE_STR}.csv"
EDGES_CSV = f"edges_ranked_{DATE_STR}.csv"
LOCKS_CSV = f"locks_{DATE_STR}.csv"
ARBS_CSV  = f"arbs_{DATE_STR}.csv"
MIDS_CSV  = f"middles_{DATE_STR}.csv"
BEST_MD   = f"best_pick_{DATE_STR}.md"

# ---- constants: key numbers (rough guides; refine per sport) ----
NFL_KEY_SPREADS = {3:0.15, 7:0.08, 6:0.05, 10:0.04, 4:0.03}
NFL_KEY_TOTALS  = {41:0.05, 44:0.05, 47:0.05, 51:0.04}

# ===== UTILITIES ==============================================================
def amer_to_decimal(american_odds: float) -> float:
    if american_odds is None: return None
    return 1.0 + (american_odds/100.0) if american_odds>0 else 1.0 + (100.0/abs(american_odds))

def implied_from_decimal(o: float) -> float:
    return 0.0 if not o or o<=0 else 1.0/o

def vig_free_two(p1, p2):
    s = p1+p2
    if s<=0: return p1,p2
    return p1/s, p2/s

def vig_free_three(p1,p2,p3):
    s = p1+p2+p3
    if s<=0: return p1,p2,p3
    return p1/s, p2/s, p3/s

def ev_from_p_o(p,o):  # EV per $1 stake
    return p*(o-1.0) - (1.0-p)

def kelly_fraction(p,o):  # o is decimal; b = o-1
    b = max(o-1.0, 0.0)
    q = 1.0 - p
    if b<=0: return 0.0
    f = (b*p - q)/b
    return max(0.0, f)

def normalize_book(name: str) -> str:
    if not name: return ""
    key = name.lower().strip()
    for k,v in BOOK_NORMALIZE.items():
        if k in key: return v
    return name

def is_oh_book(name: str) -> bool:
    norm = normalize_book(name)
    return any(b.lower()==norm.lower() or b.lower() in norm.lower() for b in OH_BOOKS)

def load_json(path, default):
    try:
        if os.path.exists(path):
            with open(path,"r") as f: return json.load(f)
    except Exception:
        pass
    return default

# Optional external data you can provide (simple JSONs you can edit)
INJURIES = load_json("injuries.json", {})        # {"Team": {"out": 2, "doubtful": 1, "questionable": 3}}
TEAM_RATINGS = load_json("team_ratings.json", {})# {"Team": {"elo": 1500, "off": 1.2, "def": 0.9}}
VENUES = load_json("venues.json", {})            # {"Team": {"lat":..., "lon":..., "alt":..., "indoor": true}}

# ===== API ACCESS =============================================================
def get_sports():
    r = requests.get(f"{BASE}?apiKey={API_KEY}", timeout=30)
    r.raise_for_status()
    js = r.json()
    if isinstance(js, dict) and js.get("message"):
        raise RuntimeError(js["message"])
    return js

def get_odds(sport_key, markets=MARKETS, region=REGION):
    params = {"apiKey":API_KEY, "regions":region, "markets":",".join(markets), "oddsFormat":"american"}
    r = requests.get(f"{BASE}/{sport_key}/odds", params=params, timeout=45)
    if r.status_code!=200:
        print(f"[warn] odds error {sport_key}: {r.text[:180]}")
        return []
    return r.json()

# ===== WEATHER (outdoor only; Open-Meteo free endpoint) =======================
def get_weather(lat, lon, iso_time):
    try:
        date = iso_time.split("T")[0]
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "hourly":"temperature_2m,precipitation,wind_speed_10m",
            "start_date": date, "end_date": date
        }
        r = requests.get(url, params=params, timeout=20)
        if r.status_code!=200: return {}
        js = r.json()
        hrs = js.get("hourly",{})
        times = hrs.get("time",[])
        if not times: return {}
        target = iso_time[:13]  # "YYYY-MM-DDTHH"
        idx = 0
        for i,t in enumerate(times):
            if t.startswith(target): idx=i; break
        return {
            "wx_temp_c": hrs.get("temperature_2m",[None])[idx] if idx<len(times) else None,
            "wx_precip_mm": hrs.get("precipitation",[None])[idx] if idx<len(times) else None,
            "wx_wind_mps": hrs.get("wind_speed_10m",[None])[idx] if idx<len(times) else None,
        }
    except Exception:
        return {}

# ===== STORAGE ================================================================
def ensure_schema(conn):
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS odds_snapshot(
      ts TEXT, sport_key TEXT, sport_title TEXT, event_id TEXT, commence_time TEXT,
      market TEXT, selection TEXT, point TEXT, book TEXT, price_american REAL,
      price_decimal REAL, implied_p_raw REAL
    )""")
    c.execute("""
    CREATE TABLE IF NOT EXISTS event_index(
      sport_key TEXT, event_id TEXT, home_team TEXT, away_team TEXT, commence_time TEXT,
      PRIMARY KEY(sport_key,event_id)
    )""")
    c.execute("""
    CREATE TABLE IF NOT EXISTS results(
      sport_key TEXT, event_id TEXT, winner TEXT, home_team TEXT, away_team TEXT, close_time TEXT,
      UNIQUE(sport_key,event_id)
    )""")
    conn.commit()

def insert_raw(conn, rows):
    c = conn.cursor()
    c.executemany("""INSERT INTO odds_snapshot
      (ts,sport_key,sport_title,event_id,commence_time,market,selection,point,book,price_american,price_decimal,implied_p_raw)
      VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
    conn.commit()

def upsert_event(conn, sport_key, eid, home, away, commence):
    c = conn.cursor()
    c.execute("""INSERT INTO event_index(sport_key,event_id,home_team,away_team,commence_time)
                 VALUES(?,?,?,?,?)
                 ON CONFLICT(sport_key,event_id) DO UPDATE SET
                   home_team=excluded.home_team, away_team=excluded.away_team, commence_time=excluded.commence_time
              """, (sport_key,eid,home,away,commence))
    conn.commit()

# ===== TIME HELPERS (robust to ...000Z etc.) =================================
def utc_to_local(iso_utc: str, tz: ZoneInfo):
    try:
        s = iso_utc.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"  # handle 'Z' and '...000Z'
        utc = dt.datetime.fromisoformat(s)
        return utc.astimezone(tz)
    except Exception:
        return None

# ===== FEATURE ENGINEERING ====================================================
def team_injury_score(team: str) -> float:
    d = INJURIES.get(team, {})
    out = float(d.get("out",0))
    doubtful = float(d.get("doubtful",0))*0.5
    questionable = float(d.get("questionable",0))*0.25
    return out + doubtful + questionable

def team_rating(team: str, key: str, default: float = 0.0) -> float:
    return float(TEAM_RATINGS.get(team, {}).get(key, default))

def venue_info(team: str) -> Dict[str,Any]:
    return VENUES.get(team, {})

def travel_km(home_team: str, away_team: str) -> float:
    import math
    h = venue_info(home_team); a = venue_info(away_team)
    if not h or not a: return 0.0
    lat1,lon1 = float(h.get("lat",0)), float(h.get("lon",0))
    lat2,lon2 = float(a.get("lat",0)), float(a.get("lon",0))
    R=6371.0
    dlat=math.radians(lat2-lat1); dlon=math.radians(lon2-lon1)
    x = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2)
    return 2*R*math.asin(min(1.0, math.sqrt(x)))

def is_indoor(home_team: str) -> bool:
    return bool(venue_info(home_team).get("indoor", False))

def altitude_m(home_team: str) -> float:
    return float(venue_info(home_team).get("alt", 0.0))

def rest_day_features(commence_iso: str, last_home_iso: str = None, last_away_iso: str = None) -> Dict[str,float]:
    # Placeholder; returns zeros but keys exist so your model schema is stable.
    return {"rest_home_days":0.0, "rest_away_days":0.0, "tz_shift_hours":0.0}

def key_number_weight(sport_key: str, market: str, point_val: float) -> float:
    if "americanfootball_nfl" in sport_key and market=="spreads":
        n = abs(round(float(point_val)))
        return NFL_KEY_SPREADS.get(n, 0.0)
    if "americanfootball_nfl" in sport_key and market=="totals":
        n = round(float(point_val))
        return NFL_KEY_TOTALS.get(n, 0.0)
    return 0.0

def feature_bundle(ctx: Dict[str,Any]) -> Dict[str,float]:
    # ctx: sport_key,event_id,home,away,commence_time,market_key,point_key,sel,best_decimal,avg_implieds
    home, away = ctx["home"], ctx["away"]
    f = {}
    # Team power + injuries
    f["elo_home"] = team_rating(home, "elo", 0.0)
    f["elo_away"] = team_rating(away, "elo", 0.0)
    f["inj_home"] = team_injury_score(home)
    f["inj_away"] = team_injury_score(away)

    # Travel / altitude / indoor
    f["travel_km"] = travel_km(home, away)
    f["altitude_m"] = altitude_m(home)
    f["indoor"] = 1.0 if is_indoor(home) else 0.0

    # Weather if outdoor
    if not f["indoor"] and f["travel_km"]>=0:
        vi = venue_info(home)
        if vi.get("lat") is not None:
            wx = get_weather(vi["lat"], vi["lon"], ctx["commence_time"])
            for k,v in wx.items():
                f[k] = float(v) if v is not None else 0.0

    # Key number emphasis (spreads/totals)
    try:
        if ctx["point_key"] not in ("None", None):
            f["keynum_weight"] = key_number_weight(ctx["sport_key"], ctx["market_key"], float(ctx["point_key"]))
        else:
            f["keynum_weight"] = 0.0
    except Exception:
        f["keynum_weight"] = 0.0

    # Rest/timezone placeholders
    f.update(rest_day_features(ctx["commence_time"]))
    return f

# ===== INGEST & CONSENSUS =====================================================
def run_ingest_and_consensus(target_date: dt.date = None, window_days: int = None):
    conn = sqlite3.connect(DB_PATH); ensure_schema(conn)

    tz = ZoneInfo(TIMEZONE)
    if target_date is None:
        target_date = dt.datetime.now(tz).date()
    if window_days is None:
        window_days = WINDOW_DAYS

    # exact start-of-day → end-of-window (local time)
    start_local = dt.datetime.combine(target_date, dt.time(0,0,0), tzinfo=tz)
    end_local   = start_local + dt.timedelta(days=window_days)

    try:
        sports = get_sports()
    except Exception as e:
        print("get_sports error:", e); sys.exit(1)
    target = [s for s in sports if s.get("active")]

    raw_rows = []
    edges_rows = []  # consensus rows with features
    h2h_by_event = defaultdict(lambda: defaultdict(list))  # for arbs/middles per event/market

    for s in target:
        skey = s.get("key"); stitle = s.get("title", skey)
        if not skey: continue
        events = get_odds(skey)
        if not events: continue

        for ev in events:
            eid = ev.get("id"); commence = ev.get("commence_time")
            commence_local = utc_to_local(commence, tz)
            # keep only events in [start_local, end_local)
            if not commence_local or not (start_local <= commence_local < end_local):
                continue

            home = ev.get("home_team"); away = ev.get("away_team")
            upsert_event(sqlite3.connect(DB_PATH), skey, eid, home, away, commence)

            # filter OH books
            bm = []
            for b in ev.get("bookmakers", []):
                t = normalize_book(b.get("title",""))
                if is_oh_book(t):
                    b["title"] = t; bm.append(b)
            if not bm: continue

            market_snapshots = defaultdict(list)  # (market, point) -> list rows

            for b in bm:
                bname = b["title"]
                for mk in b.get("markets", []):
                    mkey = mk.get("key")
                    if mkey not in MARKETS: continue
                    for oc in mk.get("outcomes", []):
                        sel = oc.get("name")
                        price = oc.get("price")
                        point = oc.get("point")
                        dec = amer_to_decimal(price)
                        if dec is None: continue
                        imp = implied_from_decimal(dec)

                        raw_rows.append([DATE_STR,skey,stitle,eid,commence,mkey,sel,point,bname,price,dec,imp])
                        market_snapshots[(mkey, str(point))].append({"book":bname,"sel":sel,"dec":dec,"imp":imp})
                        h2h_by_event[(skey,eid)][(mkey,str(point))].append({"book":bname,"sel":sel,"dec":dec,"imp":imp,"commence_local":commence_local.isoformat()})

            # consensus per (market,point)
            for (mkey,pkey), snaps in market_snapshots.items():
                by_sel = defaultdict(list)
                for r in snaps: by_sel[r["sel"]].append(r)
                sels = list(by_sel.keys())
                if len(sels)<2: continue

                avg_imp = {}
                for sel in sels:
                    imps = [x["imp"] for x in by_sel[sel] if x["imp"]>0]
                    if imps: avg_imp[sel] = sum(imps)/len(imps)
                if len(avg_imp)<2: continue

                # vig removal
                if len(sels)==2:
                    s1,s2 = sels[0], sels[1]
                    p1,p2 = vig_free_two(avg_imp.get(s1,0.0), avg_imp.get(s2,0.0))
                    fair = {s1:p1, s2:p2}
                elif len(sels)==3:
                    s1,s2,s3 = sels[0], sels[1], sels[2]
                    p1,p2,p3 = vig_free_three(avg_imp.get(s1,0.0), avg_imp.get(s2,0.0), avg_imp.get(s3,0.0))
                    fair = {s1:p1, s2:p2, s3:p3}
                else:
                    tot = sum(avg_imp.values()); fair = {k:(v/tot if tot>0 else 0.0) for k,v in avg_imp.items()}

                # EV/edge vs best price + features
                for sel in sels:
                    if sel not in fair: continue
                    best = max(by_sel[sel], key=lambda r: r["dec"])
                    o = best["dec"]; p = fair[sel]
                    book_p = 1.0/o if o>0 else 0.0
                    edge = p - book_p
                    ev  = ev_from_p_o(p,o)
                    kly = kelly_fraction(p,o)

                    # build features
                    ctx = {
                        "sport_key":skey,"event_id":eid,"home":home,"away":away,"commence_time":commence,
                        "market_key":mkey,"point_key":pkey,"sel":sel,"best_decimal":o,"avg_implieds":avg_imp
                    }
                    feats = feature_bundle(ctx)

                    edges_rows.append([
                        DATE_STR, skey, stitle, eid, commence, mkey, pkey, sel, best["book"], o,
                        p, book_p, ev, edge, kly, json.dumps(feats), commence_local.isoformat()
                    ])

    # persist + CSV
    with sqlite3.connect(DB_PATH) as conn:
        insert_raw(conn, raw_rows)

    with open(RAW_CSV,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["ts","sport_key","sport_title","event_id","commence_time","market","selection","point","book","price_american","price_decimal","implied_p_raw"]); w.writerows(raw_rows)

    edges_sorted = sorted(edges_rows, key=lambda r: r[12], reverse=True) # by EV
    with open(EDGES_CSV,"w",newline="") as f:
        w=csv.writer(f); w.writerow([
            "ts","sport_key","sport_title","event_id","commence_time_utc","market","point","selection","book_best_price",
            "decimal_odds","fair_p_consensus","book_implied_p","EV_consensus","edge_consensus","kelly_f","extra_features_json","commence_local"
        ])
        w.writerows(edges_sorted)

    return edges_sorted, h2h_by_event

# ===== RESULTS & MODEL ========================================================
RESULTS_TEMPLATE = """# Save as results_import.csv and run: python ai_line_analyzer_pro.py --import-results
# sport_key,event_id,winner,home_team,away_team,close_time
"""

def write_results_template():
    with open("results_import.csv","w") as f: f.write(RESULTS_TEMPLATE)
    print("Wrote results_import.csv template.")

def import_results_csv(path="results_import.csv"):
    if not os.path.exists(path):
        write_results_template(); return
    conn = sqlite3.connect(DB_PATH); ensure_schema(conn)
    c = conn.cursor()
    rows=[]
    with open(path,"r") as f:
        reader = csv.DictReader(r for r in f if not r.startswith("#"))
        for r in reader:
            rows.append((r["sport_key"], r["event_id"], r["winner"], r["home_team"], r["away_team"], r.get("close_time","")))
    c.executemany("""INSERT OR REPLACE INTO results(sport_key,event_id,winner,home_team,away_team,close_time) VALUES(?,?,?,?,?,?)""", rows)
    conn.commit(); print(f"Imported {len(rows)} results rows.")

def build_training_matrix(edges_csv=EDGES_CSV) -> Tuple[np.ndarray,np.ndarray,List[Dict[str,Any]]]:
    if not os.path.exists(edges_csv): return np.zeros((0,)), np.zeros((0,)), []
    rows=[]
    with open(edges_csv,"r") as f:
        rd = csv.DictReader(f)
        for r in rd:
            if r["market"]!="h2h": continue
            rows.append(r)

    conn = sqlite3.connect(DB_PATH); c=conn.cursor()
    X,y,meta = [],[],[]
    for r in rows:
        sk, eid = r["sport_key"], r["event_id"]
        c.execute("SELECT winner,home_team,away_team FROM results WHERE sport_key=? AND event_id=?", (sk,eid))
        got = c.fetchone()
        if not got: continue
        winner, home, away = got
        sel = r["selection"]
        sel_side = "home" if sel==home else ("away" if sel==away else None)
        if sel_side is None:
            s = sel.lower()
            if home and home.lower() in s: sel_side="home"
            elif away and away.lower() in s: sel_side="away"
            else: continue
        label = 1 if sel_side==winner else 0

        fair_p = float(r["fair_p_consensus"]); book_p = float(r["book_implied_p"])
        dec = float(r["decimal_odds"]); ev = float(r["EV_consensus"]); edge = float(r["edge_consensus"])
        kly = float(r.get("kelly_f",0.0))
        side = 1.0 if sel_side=="home" else 0.0

        extras={}
        try: extras=json.loads(r.get("extra_features_json","{}")) or {}
        except Exception: extras={}

        feat = [fair_p, book_p, dec, ev, edge, kly, side]
        for k,v in sorted(extras.items()):
            try: feat.append(float(v))
            except Exception: pass

        X.append(feat); y.append(label)
        meta.append({"sport_key":sk,"event_id":eid,"home":home,"away":away,"sel_side":sel_side})
    return np.array(X,float), np.array(y,int), meta

def train_model(X,y):
    if not SKLEARN_AVAILABLE or len(X)<100 or len(set(y))<2:
        print("Model skipped (need sklearn + >=100 labeled samples & both classes).")
        return None
    pipe = Pipeline([("scaler",StandardScaler()),("clf",LogisticRegression(max_iter=400))])
    pipe.fit(X,y); print(f"Model trained on {len(y)} samples."); return pipe

def predict_today(model, edges_rows):
    if model is None: return []
    preds=[]
    for r in edges_rows:
        if r[5]!="h2h": continue
        fair_p=float(r[10]); book_p=float(r[11]); dec=float(r[9]); ev=float(r[12]); edge=float(r[13]); kly=float(r[14])
        # side from event_index
        conn = sqlite3.connect(DB_PATH); c=conn.cursor()
        c.execute("SELECT home_team,away_team FROM event_index WHERE sport_key=? AND event_id=?", (r[1], r[3]))
        got=c.fetchone(); side=0.0
        if got:
            home,away = got
            side = 1.0 if r[7]==home else 0.0
        extras={}
        try: extras=json.loads(r[15]) or {}
        except Exception: extras={}
        feat=[fair_p,book_p,dec,ev,edge,kly,side]
        for k,v in sorted(extras.items()):
            try: feat.append(float(v))
            except Exception:
            pass
        X=np.array(feat).reshape(1,-1)
        p = float(model.predict_proba(X)[0][1])
        model_ev = ev_from_p_o(p, dec)
        model_edge = p - (1.0/dec)
        preds.append({
            "sport_key":r[1], "event_id":r[3], "selection":r[7], "book":r[8], "decimal_odds":dec,
            "p_consensus":fair_p, "EV_consensus":ev, "p_model":p, "EV_model":model_ev, "edge_model":model_edge,
            "commence_local": r[16]
        })
    return preds

# ===== ARBITRAGE & MIDDLES ====================================================
def find_two_way_arbs(h2h_by_event) -> List[Dict[str,Any]]:
    arbs=[]
    for (skey,eid), m in h2h_by_event.items():
        if ("h2h","None") not in m: continue
        quotes = m[("h2h","None")]
        by_sel = defaultdict(list)
        for q in quotes: by_sel[q["sel"]].append(q)
        if len(by_sel)!=2: continue
        sels = list(by_sel.keys())
        best = {sel: max(by_sel[sel], key=lambda r:r["dec"]) for sel in sels}
        invsum = sum(1.0/best[sel]["dec"] for sel in sels if best[sel]["dec"]>0)
        if invsum < 1.0:
            arbs.append({
                "sport_key":skey, "event_id":eid,
                "sel_a":sels[0], "book_a":best[sels[0]]["book"], "odds_a":best[sels[0]]["dec"],
                "sel_b":sels[1], "book_b":best[sels[1]]["book"], "odds_b":best[sels[1]]["dec"],
                "margin":1.0 - invsum
            })
    return arbs

def find_spread_middles(h2h_by_event) -> List[Dict[str,Any]]:
    mids=[]
    # look for spreads with different points across books: Fav -X at Book1 and Dog +Y at Book2 with Y>X
    for (skey,eid), m in h2h_by_event.items():
        spreads = [(k,v) for k,v in m.items() if k[0]=="spreads" and k[1] not in (None,"None")]
        if not spreads: continue
        by_side = defaultdict(list)
        for (mkey,pkey), quotes in spreads:
            try: pt=float(pkey)
            except Exception: continue
            for q in quotes:
                by_side[q["sel"]].append((pt, q["dec"], q["book"]))
        teams = list(by_side.keys())
        if len(teams)<2: continue
        t1,t2 = teams[0], teams[1]
        for pt1,od1,b1 in by_side[t1]:
            for pt2,od2,b2 in by_side[t2]:
                if pt1<0 and pt2>0 and pt2>abs(pt1):
                    mids.append({"sport_key":skey,"event_id":eid,"fav":t1,"fav_book":b1,"fav_pt":pt1,"fav_odds":od1,
                                 "dog":t2,"dog_book":b2,"dog_pt":pt2,"dog_odds":od2,"middle_pts":pt2-abs(pt1)})
    return mids

# ===== REPORTING ==============================================================
def write_csv(path, header, rows):
    with open(path,"w",newline="") as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows)

def write_report(cons_rows, model_preds, locks_cons, locks_model, arbs, mids):
    lines=[]
    lines.append(f"# Best Picks — {DATE_STR}\n")

    if cons_rows:
        best_ev   = max(cons_rows, key=lambda r:r[12])
        best_edge = max(cons_rows, key=lambda r:r[13])
        lines.append("## Value Signals (vig-free market)\n")
        lines.append(f"- **Best EV**: {best_ev[1]} | {best_ev[2]} | {best_ev[7]} @ {best_ev[8]} "
                     f"(o={float(best_ev[9]):.2f}, start={best_ev[16]}) — EV={float(best_ev[12]):.4f}, Kelly={float(best_ev[14]):.3f}")
        lines.append(f"- **Largest Edge**: {best_edge[1]} | {best_edge[2]} | {best_edge[7]} @ {best_edge[8]} "
                     f"(o={float(best_edge[9]):.2f}, start={best_edge[16]}) — edge={float(best_edge[13]):.4f}\n")

    if model_preds:
        best_mev = max(model_preds, key=lambda x:x["EV_model"])
        best_med = max(model_preds, key=lambda x:x["edge_model"])
        lines.append("## AI Model Signals\n")
        lines.append(f"- **Best Model-EV**: {best_mev['sport_key']} | {best_mev['selection']} @ {best_mev['book']} "
                     f"(o={best_mev['decimal_odds']:.2f}, start={best_mev['commence_local']}) — EV_model={best_mev['EV_model']:.4f}, p_model={best_mev['p_model']:.3f}")
        lines.append(f"- **Largest Model Edge**: {best_med['sport_key']} | {best_med['selection']} "
                     f"(start={best_med['commence_local']}) — edge_model={best_med['edge_model']:.4f}, p_model={best_med['p_model']:.3f}\n")

    lines.append("## Daily Locks — Consensus (Top 5 highest win prob)\n")
    for r in locks_cons:
        lines.append(f"- {r[1]} | {r[2]} | **{r[7]}** @ {r[8]} "
                     f"(o={float(r[9]):.2f}, p={float(r[10]):.3f}, EV={float(r[12]):.4f}, start={r[16]})")
    if model_preds:
        lines.append("\n## Daily Locks — AI Model (Top 5 highest win prob)\n")
        for m in locks_model:
            lines.append(f"- {m['sport_key']} | **{m['selection']}** @ {m['book']} "
                         f"(o={m['decimal_odds']:.2f}, p_model={m['p_model']:.3f}, EV_model={m['EV_model']:.4f}, start={m['commence_local']})")

    if arbs:
        lines.append("\n## Arbitrage Opportunities (two-way)\n")
        for a in arbs[:10]:
            lines.append(f"- {a['sport_key']} | {a['sel_a']} @ {a['book_a']} ({a['odds_a']:.2f}) vs "
                         f"{a['sel_b']} @ {a['book_b']} ({a['odds_b']:.2f}) — margin={a['margin']:.4f}")

    if mids:
        lines.append("\n## Spread Middles (positive window)\n")
        for m in mids[:10]:
            lines.append(f"- {m['sport_key']} | {m['fav']} {m['fav_pt']} @ {m['fav_book']} / "
                         f"{m['dog']} +{m['dog_pt']} @ {m['dog_book']} — middle={m['middle_pts']:.1f} pts")

    with open(BEST_MD,"w") as f: f.write("\n".join(lines))
    print(f"Wrote {BEST_MD}")

# ===== MAIN ===================================================================
def main():
    if not API_KEY or len(API_KEY)<20:
        print("Put a valid API key in API_KEY"); sys.exit(1)

    # CLI: allow exact date and multi-day windows
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="YYYY-MM-DD in America/New_York (default: today)", default=None)
    parser.add_argument("--window-days", type=int, help="How many local days to include starting at --date (default from WINDOW_DAYS)", default=None)
    args = parser.parse_args()

    tz = ZoneInfo(TIMEZONE)
    if args.date:
        try:
            target_date = dt.date.fromisoformat(args.date)
        except Exception:
            print("Invalid --date (use YYYY-MM-DD)."); sys.exit(1)
    else:
        target_date = dt.datetime.now(tz).date()

    window_days = args.window_days if args.window_days is not None else WINDOW_DAYS

    # 1) ingest + consensus for the requested slice
    cons_rows, h2h = run_ingest_and_consensus(target_date=target_date, window_days=window_days)

    # 2) Locks (Consensus & Model)
    cons_h2h = [r for r in cons_rows if r[5]=="h2h"]
    locks_cons = sorted(cons_h2h, key=lambda r:r[10], reverse=True)[:LOCKS_TOP_N]

    # 3) Train model (optional)
    X,y,_ = build_training_matrix(); model = train_model(X,y)
    model_preds = predict_today(model, cons_rows) if model else []
    locks_model = sorted(model_preds, key=lambda m:m["p_model"], reverse=True)[:LOCKS_TOP_N] if model_preds else []

    # 4) Arbitrage & Middles
    arbs = find_two_way_arbs(h2h)
    mids = find_spread_middles(h2h)

    # 5) Save CSVs
    write_csv(LOCKS_CSV, ["type","sport_key","event_id","selection","book","decimal_odds","p_or_pmodel","EV","start_local"],
              [["consensus", r[1], r[3], r[7], r[8], r[9], r[10], r[12], r[16]] for r in locks_cons] +
              ([["model", m["sport_key"], m["event_id"], m["selection"], m["book"], m["decimal_odds"], m["p_model"], m["EV_model"], m["commence_local"]] for m in locks_model] if locks_model else [])
             )
    write_csv(ARBS_CSV, ["sport_key","event_id","sel_a","book_a","odds_a","sel_b","book_b","odds_b","margin"],
              [[a["sport_key"],a["event_id"],a["sel_a"],a["book_a"],a["odds_a"],a["sel_b"],a["book_b"],a["odds_b"],a["margin"]] for a in arbs])
    write_csv(MIDS_CSV, ["sport_key","event_id","fav","fav_book","fav_pt","fav_odds","dog","dog_book","dog_pt","dog_odds","middle_pts"],
              [[m["sport_key"],m["event_id"],m["fav"],m["fav_book"],m["fav_pt"],m["fav_odds"],m["dog"],m["dog_book"],m["dog_pt"],m["dog_odds"],m["middle_pts"]] for m in mids])

    # 6) Markdown summary
    write_report(cons_rows, model_preds, locks_cons, locks_model, arbs, mids)
    print(f"Wrote CSVs: {RAW_CSV}, {EDGES_CSV}, {LOCKS_CSV}, {ARBS_CSV}, {MIDS_CSV}")

if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="--import-results":
        import_results_csv()
    else:
        main()