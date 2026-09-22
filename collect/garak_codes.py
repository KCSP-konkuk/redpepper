import json, requests
P={}
for line in open('/opt/agri-forecast/application-secret.properties'):
    if '=' in line and not line.strip().startswith('#'):
        k,_,v=line.strip().partition('='); P[k.strip()]=v.strip()
u=('http://www.garak.co.kr/homepage/publicdata/dataJsonOpen.do'
   f"?id=7043&passwd={P['garak.public-data.password']}"
   '&dataid=data22&pagesize=500&pageidx=1&portal.templet=false&date=20260918')
rows=requests.get(u,timeout=60).json()['resultData']
out=[f"{r['PUM_CD']}\t{r['PUM_NM']}\t{r.get('BURYU','')}" for r in rows if str(r['PUM_CD']).isdigit()]
open('/tmp/garak_codes.txt','w').write('\n'.join(out))
print('품목수',len(out))
print('\n'.join([o for o in out if any(k in o for k in ('고추','피망','파프리카','가지','토마토','오이'))]))
