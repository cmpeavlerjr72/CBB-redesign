import json,sys,glob,collections,os
from datetime import datetime
out=[]
for f in sorted(glob.glob(sys.argv[1]+"/*.trades.json")):
    tr=json.load(open(f));
    if not tr: continue
    if len(out)==0: print(tr[0])
    cnt=[float(t["count_fp"]) for t in tr]
    n=len(tr);tot=sum(cnt)
    s=sorted(cnt,reverse=True);k=max(1,n//100)
    top1=sum(s[:k])/tot
    sizes=collections.Counter(cnt)
    rep=[(c,v) for c,v in sizes.most_common(6)]
    big=[c for c in cnt if c>=1000]
    yes=sum(float(t["count_fp"]) for t in tr if t["taker_side"]=="yes")
    ts=[datetime.fromisoformat(t["created_time"].replace("Z","+00:00")).timestamp() for t in tr]
    sec=collections.Counter(int(x) for x in ts)
    span=max(ts)-min(ts)
    ps=sorted(sec.values())
    # exact-size repeats among >=500 lots
    bigsz=collections.Counter(c for c in cnt if c>=500)
    rb=[(c,v) for c,v in bigsz.most_common(5) if v>1]
    ms=collections.Counter(round((x%1)*1000)//100 for x in ts)
    print(os.path.basename(f),dict(n=n,vol=round(tot),span_min=round(span/60,1),median=sorted(cnt)[n//2],mean=round(tot/n,1),top1pct_share=round(top1,3),n_ge1000=len(big),vol_ge1000=round(sum(big)/tot,3),yes_taker_share=round(yes/tot,3),maxprints_per_sec=ps[-1],p99_per_sec=ps[int(len(ps)*.99)],active_secs=len(sec),top_sizes=rep,repeat_big=rb))
