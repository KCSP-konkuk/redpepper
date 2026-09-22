"""교차 품목 순별 가격 수집 — 풋고추·청피망·(홍고추 검증용). 품목당 ~39회"""
import sys, time, pandas as pd
from nongnet import session, fetch
ITEMS={'홍고추':('24210','10'),'풋고추':('24201','10'),'청피망':('25101','10'),'홍피망':('25102','10')}
PS={1:'상순',2:'중순',3:'하순'}
def crawl(s,code,spec,dtype='price'):
    rows={}; 
    # 2026-09 에서 9순씩 거슬러 올라가며 2017-01 까지
    y,m,p=2026,9,3
    while (y,m) >= (2001,1):
        ds=f'{y}년 {m:02d}월 {[5,15,25][p-1]:02d}일'
        for att in range(3):
            try:
                r=s.get_json if False else fetch(s,code,spec,ds,dtype)
                d=r.json().get('datalist',[]); break
            except Exception as e:
                print('  retry',ds,e,file=sys.stderr); time.sleep(3); d=[]
        for it in d:
            key=f"{it['year']}{it['month']:02d}{PS[it['soonVal']]}"
            rows[key]=dict(DATE=key, val=it.get('selectSoon'), yearAvg=it.get('yearAvg'),
                           bfYear=it.get('bfYear'))
        # 9순 뒤로
        for _ in range(9):
            p-=1
            if p==0: p=3; m-=1
            if m==0: m=12; y-=1
        time.sleep(0.8)
    return pd.DataFrame(sorted(rows.values(), key=lambda r:r['DATE']))
if __name__=='__main__':
    s=session()
    for nm,(c,sp) in ITEMS.items():
        df=crawl(s,c,sp)
        df.to_csv(f'cross_long_{nm}.csv',index=False)
        print(nm, df.shape, df.DATE.min(), df.DATE.max(), '결측', df.val.isna().sum(), file=sys.stderr)
