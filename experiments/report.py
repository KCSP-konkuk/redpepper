import os
_HERE=os.path.dirname(os.path.abspath(__file__))
DATA_DIR=os.path.join(_HERE,'..','data')+os.sep
D=DATA_DIR
SP=DATA_DIR
D2=DATA_DIR
"""홍고추 vs 배추 — 1순 앞, 논문 표준 지표 + Diebold-Mariano"""
from soon import *
D2=DATA_DIR
import json

PSn={'상순':0,'중순':1,'하순':2}
cfg=json.load(open(os.path.join(_HERE,'best_soon.json'))); BPp=cfg['params']; Kp=cfg['K']
YRS=(2021,2022,2023,2024,2025)          # 배추 CSV 가 2025 까지라 공통 구간

# --- 홍고추
dfp=load_long()
Xp=pd.concat([f_price(dfp),f_cross_l(dfp),f_trend(dfp),f_hol_soon(dfp)],axis=1)
rp_,(P1,A1,N1,Y1)=eval_soon(dfp,Xp,K=Kp,seeds=12,params=BPp,years=YRS,ret_pred=True)
rp6=eval_soon(dfp,Xp,K=Kp,seeds=12,params=BPp,years=YEARS)

# --- 배추 (자기 최적: 가격+기상+검색량)
def load_cb():
    c=pd.read_csv(D2+'hist_price.csv',encoding='utf-8-sig'); c['DATE']=c.DATE.astype(str)
    c=c.rename(columns={'평균가격':'price','전년':'py','평년':'ny'})[['DATE','price','py','ny']]
    for col in ('price','py','ny'): c[col]=pd.to_numeric(c[col],errors='coerce')
    c['year']=c.DATE.str[:4].astype(int); c['month']=c.DATE.str[4:6].astype(int)
    c['pidx']=c.DATE.str[6:].map(PSn)
    return c.sort_values('DATE').reset_index(drop=True)
def to_soon(dates):
    d=pd.to_datetime(dates)
    return d.dt.strftime('%Y%m')+d.dt.day.map(lambda x:'상순' if x<=10 else '중순' if x<=20 else '하순')
def lagf(df,src,cols,lags,pre=''):
    m=df[['DATE']].merge(src,on='DATE',how='left'); f=pd.DataFrame(index=df.index)
    for c in cols:
        v=pd.to_numeric(m[c],errors='coerce')
        for l in lags: f[f'{pre}{c}_l{l}']=v.shift(l)
        f[f'{pre}{c}_ma3']=v.shift(1).rolling(3).mean()
    return f
def wthr(df,fn,tag,lags=(1,3,6,9)):
    w=pd.read_csv(D2+fn,encoding='utf-8-sig'); w.columns=[c.strip() for c in w.columns]
    w['DATE']=w.DATE.astype(str)
    cols=[c for c in w.columns if c!='DATE' and not c.startswith('전년')]
    w=w[['DATE']+cols].rename(columns={c:f'{tag}_{c}' for c in cols})
    w=w.loc[:,~w.columns.duplicated()]
    return lagf(df,w,[f'{tag}_{c}' for c in cols],lags)
dfc=load_cb()
tr=pd.read_csv(SP+'trend_db.tsv',sep='\t',names=['kw','period','ratio']); tr['DATE']=to_soon(tr.period)
TR=tr.pivot_table(index='DATE',columns='kw',values='ratio',aggfunc='mean').reset_index()
Xc=pd.concat([f_price(dfc),wthr(dfc,'weather_haenam.csv','해남'),wthr(dfc,'weather_taebak.csv','태백'),
              lagf(dfc,TR,['배추'],(1,2,3,6,12),pre='sr_')],axis=1)
rc,(P2,A2,N2,Y2)=eval_soon(dfc,Xc,K=Kp,seeds=12,params=BPp,years=YRS,start=2018,ret_pred=True)

names=['R2','MAE','RMSE','MAPE','sMAPE','MdAPE','MASE','RMSSE','TheilU','DA']
t=pd.DataFrame({'홍고추 모델':{k:rp_[k] for k in names},'홍고추 나이브':panel(N1,A1,N1),
                '배추 모델':{k:rc[k] for k in names},'배추 나이브':panel(N2,A2,N2)}).T
print('=== 1순 앞 · 홀드아웃 2021~2025 (공통 구간) ===')
print(t.round(3).to_string())
print()
for nm,(P,A,N) in {'홍고추':(P1,A1,N1),'배추':(P2,A2,N2)}.items():
    s1,p1=dm(P,N,A,loss='abs'); s2,p2=dm(P,N,A,loss='sq')
    print(f'Diebold-Mariano {nm}: 절대손실 t={s1:+.2f} p={p1:.4f} / 제곱손실 t={s2:+.2f} p={p2:.4f}  (음수=모델 우위)')
print()
print('연도별 MASE')
for nm,r in [('홍고추',rp_),('배추',rc)]:
    print(f'  {nm}: '+"  ".join(f'{y} {m:.2f}' for y,m,_ in r['yr']))
print(f"\n홍고추 2026 포함(2021~2026): MASE {rp6['MASE']:.3f}  MAPE {rp6['MAPE']:.2f}%  R² {rp6['R2']:.3f}")
