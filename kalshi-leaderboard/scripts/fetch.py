import sys,urllib.request,time,os
H={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36","Accept":"application/json, text/html, */*","Referer":"https://kalshi.com/"}
url,out=sys.argv[1],sys.argv[2]
try:
    r=urllib.request.urlopen(urllib.request.Request(url,headers=H),timeout=30);b=r.read();c=r.status
except urllib.error.HTTPError as e: b=e.read();c=e.code
except Exception as e: b=str(e).encode();c="ERR"
open(out,"wb").write(b);print(c,len(b),url);print(b[:300].decode("utf8","replace"))
time.sleep(0.6)
