import json, urllib.request, urllib.parse, time, sys
H={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130.0 Safari/537.36","Accept":"application/json","Referer":"https://kalshi.com/"}
B="https://api.elections.kalshi.com"
def get(p):
    r=urllib.request.Request(B+p,headers=H)
    try:
        with urllib.request.urlopen(r,timeout=20) as f: return f.status, json.loads(f.read())
    except urllib.error.HTTPError as e: return e.code, e.read()[:120].decode(errors="replace")
    except Exception as e: return -1, str(e)[:120]
# a) categories
for c in ["Politics","Sports","Culture","Crypto","Climate","Economics","Mentions","Financials","Science","Elections","Companies","World","Entertainment","Health","Weather"]:
    s,d=get(f"/v1/social/leaderboard?metric_name=projected_pnl&limit=5&time_period=monthly&category={urllib.parse.quote(c)}")
    top=[x["nickname"] for x in d.get("rank_list",[])][:3] if isinstance(d,dict) else d
    print("cat",c,s,top); time.sleep(0.5)
# b) global trades REST
s,d=get("/trade-api/v2/markets/trades?limit=1000")
if s==200:
    t=d["trades"]; print("global trades",len(t),"cursor" ,bool(d.get("cursor")), t[0]["created_time"], t[-1]["created_time"], list(t[0].keys()))
else: print("global trades",s,d)
time.sleep(0.5)
# c) profile/metrics for Limited top handle
for n in ["final.lamb3899","far.bike"]:
    for p in ["weekly","monthly","all_time"]:
        s,d=get(f"/v1/social/profile/metrics?nickname={n}&time_period={p}"); print("metrics",n,p,s,json.dumps(d)[:200]); time.sleep(0.5)
# d) depth of Full user feed
cur=None; pages=0; first=last=None
for i in range(40):
    q=f"/v1/social/trades?nickname=far.bike&page_size=100"+(f"&cursor={cur}" if cur else "")
    s,d=get(q)
    if s!=200 or not d.get("trades"): print("depth stop",s,str(d)[:100]); break
    pages+=1; first=first or d["trades"][0]["create_date"]; last=d["trades"][-1]["create_date"]; cur=d.get("cursor")
    if not cur: break
    time.sleep(0.5)
print("far.bike pages",pages,"first",first,"last",last)
# e) series endpoint for category mapping
s,d=get("/trade-api/v2/series?limit=5"); print("series",s,json.dumps(d)[:400])
s,d=get("/trade-api/v2/series?category=Economics&limit=3"); print("series econ",s,json.dumps(d)[:300])
