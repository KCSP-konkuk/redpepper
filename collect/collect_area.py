"""홍고추 순별 반입량·출하지역 장기 수집 (soonDataType=area)"""
import sys, time, pandas as pd
from nongnet import session, fetch
PS={1:'상순',2:'중순',3:'하순'}
s=session(); rows={}
y,m,p=2026,9,3
while (y,m)>=(2001,1):
    ds=f'{y}년 {m:02d}월 {[5,15,25][p-1]:02d}일'
    for att in range(3):
        try:
            d=fetch(s,'24210','10',ds,'area').json().get('datalist',[]); break
        except Exception as e:
            print('retry',ds,e,file=sys.stderr); time.sleep(3); d=[]
    for it in d:
        k=f"{it['year']}{it['month']:02d}{PS[it['soonVal']]}"
        rows[k]=dict(DATE=k,sup=it.get('selectSoon'),sup_py=it.get('bfYear'),
                     sup_ny=it.get('yearAvg'),r1=it.get('rank1Name'),w1=it.get('rank1Weight'),
                     r2=it.get('rank2Name'),w2=it.get('rank2Weight'),wetc=it.get('etc'))
    for _ in range(9):
        p-=1
        if p==0: p=3; m-=1
        if m==0: m=12; y-=1
    time.sleep(0.8)
df=pd.DataFrame(sorted(rows.values(),key=lambda r:r['DATE']))
df.to_csv('area_long_홍고추.csv',index=False)
print(df.shape, df.DATE.min(), df.DATE.max(), '반입결측', df.sup.isna().sum(), file=sys.stderr)
