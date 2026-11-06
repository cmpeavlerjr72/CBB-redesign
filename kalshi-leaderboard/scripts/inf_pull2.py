import json,sys,time,urllib.request,os,calendar
from datetime import datetime
B="https://api.elections.kalshi.com/trade-api/v2"
OUT=sys.argv[1]
def get(u):
    for i in range(5):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":"research/1.0"}),timeout=30))
        except Exception as e:
            time.sleep(2+i)
    raise SystemExit("fail "+u)
ts=lambda s:calendar.timegm(datetime.strptime(s,"%Y-%m-%dT%H:%M:%SZ").timetuple())
# (ticker, close, window start offset sec before close, maxpages)
J=[("KXNFLGAME-26NOV01KCDEN-KC","2026-11-02T00:37:22Z",1800),
   ("KXMLBGAME-26OCT282000MILTB-MIL","2026-10-29T03:14:32Z",1800),
   ("KXNFLGAME-26NOV01MINDET-MIN","2026-11-01T21:07:35Z",1800)]
for t,close,w in J:
    c0=ts(close);tr=[];c=None;p=0
    while p<80:
        u=f"{B}/markets/trades?ticker={t}&limit=1000&min_ts={c0-w}&max_ts={c0}"+(f"&cursor={c}" if c else "")
        d=get(u);tr+=d["trades"];c=d.get("cursor");p+=1
        print(t,p,len(tr),flush=True)
        if not c or not d["trades"]:break
        time.sleep(0.15)
    json.dump(tr,open(os.path.join(OUT,t+".trades.json"),"w"))
