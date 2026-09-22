import os
_HERE=os.path.dirname(os.path.abspath(__file__))
DATA_DIR=os.path.join(_HERE,'..','data')+os.sep
D=DATA_DIR
SP=DATA_DIR
D2=DATA_DIR
"""최종 비교 — 각 품목 최적 구성, 1순 앞, 2021~2025"""
from report import dfp, Xp, dfc, Xc, YRS
from soon import *
import json
cp=json.load(open(os.path.join(_HERE,'best_soon.json'))); cc=json.load(open(os.path.join(_HERE,'best_cabbage.json')))
rp_,(P1,A1,N1,Y1)=eval_soon(dfp,Xp,K=cp['K'],seeds=12,params=cp['params'],years=YRS,ret_pred=True)
rc,(P2,A2,N2,Y2)=eval_soon(dfc,Xc,K=cc['K'],seeds=12,params=cc['params'],years=YRS,start=2018,ret_pred=True)
names=['R2','MAE','RMSE','MAPE','sMAPE','MdAPE','MASE','RMSSE','DA']
t=pd.DataFrame({'홍고추 모델':{k:rp_[k] for k in names},'홍고추 나이브':{k:panel(N1,A1,N1)[k] for k in names},
                '배추 모델':{k:rc[k] for k in names},'배추 나이브':{k:panel(N2,A2,N2)[k] for k in names}}).T
print('=== 1순 앞 · 홀드아웃 2021~2025 · 각 품목 최적 구성 ===')
print(t.round(3).to_string()); print()
for nm,(P,A,N) in {'홍고추':(P1,A1,N1),'배추':(P2,A2,N2)}.items():
    s1,p1=dm(P,N,A,loss='abs'); s2,p2=dm(P,N,A,loss='sq')
    print(f'DM {nm}: 절대손실 t={s1:+.2f} p={p1:.4f} | 제곱손실 t={s2:+.2f} p={p2:.4f}')
print()
for nm,r in [('홍고추',rp_),('배추',rc)]:
    print(f'{nm} 연도별 MASE: '+"  ".join(f'{y} {m:.2f}' for y,m,_ in r['yr'])
          +'   MAPE: '+"  ".join(f'{mp:.1f}%' for _,_,mp in r['yr']))
r6=eval_soon(dfp,Xp,K=cp['K'],seeds=12,params=cp['params'],years=YEARS)
print(f"\n홍고추 2026 포함: MASE {r6['MASE']:.3f} MAPE {r6['MAPE']:.2f}% R² {r6['R2']:.3f} DA {r6['DA']:.1f}%")
