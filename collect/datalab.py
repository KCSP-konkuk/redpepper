"""네이버 데이터랩 검색량 — 단일 호출(기간 분할 금지). DB 안 건드리고 /tmp 로만 출력"""
import json, requests, datetime
P={}
for line in open('/opt/agri-forecast/application-secret.properties'):
    if '=' in line and not line.strip().startswith('#'):
        k,_,v=line.strip().partition('='); P[k.strip()]=v.strip()
today=datetime.date.today().strftime('%Y-%m-%d')
groups=[{'groupName':n,'keywords':k} for n,k in
        [('홍고추',['홍고추','붉은고추']),('고추',['고추']),
         ('고춧가루',['고춧가루','고추가루']),('김장',['김장'])]]
body={'startDate':'2016-01-01','endDate':today,'timeUnit':'date','keywordGroups':groups}
r=requests.post('https://openapi.naver.com/v1/datalab/search',
    headers={'X-Naver-Client-Id':P['naver.datalab.client-id'],
             'X-Naver-Client-Secret':P['naver.datalab.client-secret'],
             'Content-Type':'application/json'}, json=body, timeout=60)
print(r.status_code)
d=r.json()
rows=[]
for g in d['results']:
    for x in g['data']: rows.append((g['title'],x['period'],x['ratio']))
import csv
with open('/tmp/pepper_trend.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['kw','period','ratio']); w.writerows(rows)
print(len(rows), rows[:2], rows[-2:])
