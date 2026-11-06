import json,sys,time,urllib.request,os
B="https://api.elections.kalshi.com/trade-api/v2"
OUT=sys.argv[1]
def get(u):
    for i in range(5):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":"research/1.0"}),timeout=30))
        except Exception as e:
            time.sleep(2+i)
    raise SystemExit("fail "+u)
ms=[]
for s in ("KXNFLGAME","KXMLBGAME"):
    d=get(f"{B}/markets?series_ticker={s}&status=settled&limit=100")
    ms+=d["markets"]
ms=[m for m in ms if m["close_time"]>="2026-10-25"]
ms.sort(key=lambda m:-float(m.get("volume_fp") or 0))
json.dump(ms[:30],open(os.path.join(OUT,"markets_top.json"),"w"))
for m in ms[:6]:
    print(m["ticker"],m["close_time"],m.get("volume_fp"),m["result"])
sel=[m["ticker"] for m in ms[:4]]
for t in sel:
    tr=[];c=None
    while True:
        u=f"{B}/markets/trades?ticker={t}&limit=1000"+(f"&cursor={c}" if c else "")
        d=get(u);tr+=d["trades"];c=d.get("cursor")
        if not c or not d["trades"]:break
        time.sleep(0.2)
    json.dump(tr,open(os.path.join(OUT,t+".trades.json"),"w"))
    print(t,len(tr))
