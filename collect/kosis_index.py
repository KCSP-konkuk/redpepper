import requests, csv
P={}
for line in open('/opt/agri-forecast/application-secret.properties'):
    if '=' in line and not line.strip().startswith('#'):
        k,_,v=line.strip().partition('='); P[k.strip()]=v.strip()
key=P['kosis.api-key']
B='https://kosis.kr/openapi/Param/statisticsParameterData.do'
def get(**kw):
    p=dict(method='getList',apiKey=key,format='json',jsonVD='Y',prdSe='M',
           objL3='',objL4='',objL5='',objL6='',objL7='',objL8='',
           startPrdDe='200101',endPrdDe='202608',**kw)
    return requests.get(B,params=p,timeout=180).json()
rows=[]
cpi=get(orgId='101',tblId='DT_1J22112',itmId='T+',objL1='T10',objL2='A02A01717')
for x in cpi: rows.append(('CPI_풋고추',x['PRD_DE'],x['DT']))
PPI={'PPI_풋고추':'10112113AA','PPI_건고추':'10112114AA','PPI_고추가루':'30116101AA','PPI_고추장':'30116107AA'}
for nm,c in PPI.items():
    d=get(orgId='301',tblId='DT_404Y016',itmId='13103134764999+',objL1=f'13102134764ACC_CD.{c}',objL2='')
    for x in d: rows.append((nm,x['PRD_DE'],x['DT']))
with open('/tmp/pepper_index.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['kw','prd','val']); w.writerows(rows)
from collections import Counter
print(Counter(r[0] for r in rows), min(r[1] for r in rows), max(r[1] for r in rows))
