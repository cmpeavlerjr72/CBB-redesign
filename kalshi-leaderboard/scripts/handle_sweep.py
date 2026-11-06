import urllib.request,urllib.parse,json,time,csv,statistics,collections,re,os,sys
H={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36","Accept":"application/json","Referer":"https://kalshi.com/"}
B="https://api.elections.kalshi.com/v1/"
def g(path,**p):
    time.sleep(0.55)
    u=B+path+"?"+urllib.parse.urlencode(p)
    try:return json.load(urllib.request.urlopen(urllib.request.Request(u,headers=H),timeout=30))
    except urllib.error.HTTPError as e:return {"_err":e.code}
    except Exception as e:return {"_err":str(e)}
S="samples/kalshi/"
hs={}
for m,tp in [("projected_pnl","monthly"),("projected_pnl","all_time"),("volume","all_time")]:
    for x in g("social/leaderboard",metric_name=m,time_period=tp,limit=60)["rank_list"]:
        d=hs.setdefault(x["nickname"],{})
        d[f"rank_{m}_{tp}"]=x["rank"];d[f"val_{m}_{tp}"]=x["value"]
print(len(hs),flush=True)
def cat(s):
    if re.match(r'KX(NFL|NBA|MLB|NHL|NCAA|CFB|CBB|WNBA|MVE|ATP|WTA|SOCCER|EPL|UFC|F1|NASCAR|MMA|SERIE|LALIGA|BUND|UCL|MLS|PGA|GOLF|TENNIS|EERSTE|LIGUE|CRICKET|BOXING|NCAAF|NCAAB|CS2|LOL|DOTA|VAL)',s) and not s.startswith('KXMVECROSS'):return 'sports'
    if s.startswith('KXMVECROSS'):return 'sports/combos'
    if re.match(r'KX(PRES|SENATE|HOUSE|GOV|ELECT|TRUMP|BIDEN|POTUS|MAYOR|DEM|GOP|MIDTERM|SCOTUS|VOTE|NYC|CONGRESS|SHUTDOWN|TARIFF|IMPEACH|VANCE|NOM|PRIMARY|ELON|REP)',s):return 'politics'
    if re.match(r'KX(CPI|FED|GDP|JOBS|UNRATE|PAYROLL|INX|NASDAQ|RECESS|FOMC|JOBLESS|PCE|RATE|TNOTE|ECON|WTI|OIL|GAS|GOLD|EGG|MORTGAGE|USDJPY|EURUSD|AAA|TREAS)',s):return 'econ/finance'
    if re.match(r'KX(HIGH|LOW|RAIN|SNOW|TEMP|WEATHER|HURR|HEAT|CLIM)',s):return 'weather'
    if re.match(r'KX(BTC|ETH|SOL|XRP|DOGE|CRYPTO|SHIBA|BNB)',s):return 'crypto'
    if re.match(r'KX(MENTION|TWEET|SPOTIFY|TOPSONG|OSCAR|EMMY|GRAMMY|BILLBOARD|BOXOFFICE|RT|NETFLIX|MOVIE|SNL|TIME|APPLE|SONG|ALBUM|TAYLOR|MRBEAST|CULT|TV)',s):return 'culture/mentions'
    return 'other'
rows=[];mix={}
for n,(nick,d) in enumerate(hs.items()):
    prof=g("social/profile",nickname=nick);met=g("social/profile/metrics",nickname=nick,time_period="all_time")
    metw=g("social/profile/metrics",nickname=nick,time_period="monthly")
    tr=[];vs="";cur=None
    for pg in range(2):
        p=dict(nickname=nick,page_size=100)
        if cur:p["cursor"]=cur
        t=g("social/trades",**p)
        if "_err" in t:vs="err"+str(t["_err"]);break
        vs=t.get("visibility_state","");tr+=t.get("trades") or [];cur=t.get("cursor")
        if not cur or vs=="hidden":break
    json.dump({"profile":prof,"metrics_all":met,"metrics_monthly":metw,"trades":tr,"visibility_state":vs},open(S+"handles/"+re.sub(r'[^\w.-]','_',nick)+".json","w"))
    m=met.get("metrics",{}) if isinstance(met,dict) else {}
    mm=metw.get("metrics",{}) if isinstance(metw,dict) else {}
    has_pnl=("pnl" in m) or ("pnl" in mm)
    if vs=="hidden": cls="Limited" if has_pnl else "Minimum/hidden"
    elif vs=="visible": cls="Full" if tr else "Full(no fills)"
    else: cls=vs or "unknown"
    mine=[x for x in tr if x.get("maker_nickname")==nick or x.get("taker_nickname")==nick]
    sizes=[float(x["count_fp"]) for x in mine]
    pref=collections.Counter(x["ticker"].split("-")[0] for x in mine)
    cats=collections.Counter(cat(k) for k in [x["ticker"].split("-")[0] for x in mine])
    tk=sum(1 for x in mine if x.get("taker_nickname")==nick);mk=sum(1 for x in mine if x.get("maker_nickname")==nick)
    times=sorted(x["create_date"] for x in mine)
    r=dict(nickname=nick,**d,social_id=(prof.get("social_profile") or {}).get("social_id",""),followers=(prof.get("social_profile") or {}).get("follower_count",""),
      class_=cls,pnl_all=m.get("pnl",""),vol_all=m.get("volume",""),pnl_month=mm.get("pnl",""),n_fills=len(mine),
      earliest=times[0] if times else "",latest=times[-1] if times else "",n_tickers=len({x["ticker"] for x in mine}),n_series=len(pref),
      median_size=statistics.median(sizes) if sizes else "",max_size=max(sizes) if sizes else "",taker_share=round(tk/len(mine),3) if mine else "",maker_share=round(mk/len(mine),3) if mine else "",
      top_series="; ".join(f"{k}:{v/len(mine):.0%}" for k,v in pref.most_common(6)),cat_mix="; ".join(f"{k}:{v/len(mine):.0%}" for k,v in cats.most_common()))
    rows.append(r)
    if n%10==0:print(n,nick,cls,len(mine),flush=True)
keys=[]
for r in rows:
    for k in r:
        if k not in keys:keys.append(k)
w=csv.DictWriter(open(S+"top_handles_sweep.csv","w",newline="",encoding="utf8"),fieldnames=keys);w.writeheader();w.writerows(rows)
print("done",collections.Counter(r["class_"] for r in rows))
