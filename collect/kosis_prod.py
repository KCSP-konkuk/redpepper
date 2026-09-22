import requests, csv
P={}
for line in open('/opt/agri-forecast/application-secret.properties'):
    if '=' in line and not line.strip().startswith('#'):
        k,_,v=line.strip().partition('='); P[k.strip()]=v.strip()
key=P['kosis.api-key']
r=requests.get('https://kosis.kr/openapi/Param/statisticsParameterData.do',
   params=dict(method='getList',apiKey=key,format='json',jsonVD='Y',
               orgId='101',tblId='DT_1ET0291',itmId='ALL',objL1='ALL',objL2='',
               objL3='',objL4='',objL5='',objL6='',objL7='',objL8='',
               prdSe='Y',startPrdDe='2000',endPrdDe='2026'),timeout=180)
d=r.json()
KEEP={'T10':'건고추_면적','T12':'건고추_단수','T14':'건고추_생산량',
      'T16':'풋고추_면적','T20':'풋고추_생산량'}
REG={'00':'전국','37':'경북','36':'전남','32':'강원','35':'전북'}
rows=[(REG[x['C1']],KEEP[x['ITM_ID']],x['PRD_DE'],x['DT'])
      for x in d if x.get('ITM_ID') in KEEP and x.get('C1') in REG]
with open('/tmp/pepper_prod.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['region','item','year','val']); w.writerows(rows)
print(len(rows), sorted({r[2] for r in rows}))
for r in rows[:6]: print(r)
