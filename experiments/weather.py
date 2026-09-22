import os
_HERE=os.path.dirname(os.path.abspath(__file__))
DATA_DIR=os.path.join(_HERE,'..','data')+os.sep
D=DATA_DIR
SP=DATA_DIR
D2=DATA_DIR
"""ASOS 일별 → 순별 집계. 지점별 컬럼 세트 생성 (throwaway)"""
import numpy as np, pandas as pd

NAME={192:'진주',156:'광주',212:'홍천',100:'대관령',211:'인제',136:'안동'}
def per(d): return 0 if d<=10 else (1 if d<=20 else 2)

def soon_table():
    d=pd.read_csv(SP+'pepper_kma_daily.csv')
    d['tm']=d.tm.astype(str)
    d['RN']=d.RN.fillna(0)                       # 결측 = 무강수
    d['y']=d.tm.str[:4].astype(int); d['m']=d.tm.str[4:6].astype(int)
    d['p']=d.tm.str[6:8].astype(int).map(per)
    d['hot']=(d.TMAX>=33).astype(float)          # 고온장해
    d['rainy']=(d.RN>=1).astype(float)           # 강우일수(탄저병)
    d['heavy']=(d.RN>=30).astype(float)
    d['cold']=(d.TMIN<=5).astype(float)
    g=d.groupby(['stn','y','m','p']).agg(
        TA=('TA','mean'),TMAX=('TMAX','mean'),TMIN=('TMIN','mean'),TS=('TS','mean'),
        TG=('TG','mean'),HM=('HM','mean'),SS=('SS','mean'),SI=('SI','mean'),
        CA=('CA','mean'),WSMAX=('WSMAX','max'),
        RNsum=('RN','sum'),RNmax=('RN','max'),
        hot=('hot','sum'),rainy=('rainy','sum'),heavy=('heavy','sum'),cold=('cold','sum')).reset_index()
    g['DATE']=[f'{r.y}{r.m:02d}'+['상순','중순','하순'][r.p] for r in g.itertuples()]
    g['trange']=g.TMAX-g.TMIN
    return g

# 산지 계절 전환: 출하지역 1위 기준 (1~7·12월 진주 / 8월 홍천 / 9~10월 대관령(평창) / 11월 광주)
SEASON={1:192,2:192,3:192,4:192,5:192,6:192,7:192,8:212,9:100,10:100,11:156,12:192}
SI_DONOR={212:100, 211:100}                      # 일사 미관측 → 대관령 값으로 대체

def season_series(g, cols):
    out=[]
    si=g[g.stn==100][['DATE','SI']].rename(columns={'SI':'SI_don'})
    for (y,m,p),_ in g.groupby(['y','m','p']):
        stn=SEASON[m]; date=f'{y}{m:02d}'+['상순','중순','하순'][p]
        row=g[(g.stn==stn)&(g.DATE==date)]
        if row.empty: continue
        r=row.iloc[0][cols].to_dict(); r['DATE']=date
        if stn in SI_DONOR or pd.isna(r.get('SI',np.nan)):
            dv=si[si.DATE==date]
            if not dv.empty: r['SI']=dv.iloc[0].SI_don
        out.append(r)
    return pd.DataFrame(out)

def wide(g, stns, cols):
    """지점별 컬럼을 옆으로 붙임"""
    base=None
    for s in stns:
        t=g[g.stn==s][['DATE']+cols].rename(columns={c:f'{NAME[s]}_{c}' for c in cols})
        base=t if base is None else base.merge(t,on='DATE',how='outer')
    return base
