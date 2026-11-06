import urllib.request,json,time,sys,collections
H={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36","Accept":"application/json","Referer":"https://kalshi.com/"}
def g(u):
    time.sleep(0.6);return json.load(urllib.request.urlopen(urllib.request.Request(u,headers=H),timeout=30))
B="https://api.elections.kalshi.com/v1/social/trades?page_size=100"
cur="";rows=[]
for i in range(10):
    d=g(B+("&cursor="+cur if cur else ""));rows+=d['trades'];cur=d.get('cursor')
    if not cur:break
json.dump(rows,open(sys.argv[1],'w'))
n=[(r['maker_nickname'],r['taker_nickname'],r['ticker'],r['create_date']) for r in rows if r['maker_nickname'] or r['taker_nickname']]
print(len(rows),rows[0]['create_date'],rows[-1]['create_date'],len(n));print(n[:12])
