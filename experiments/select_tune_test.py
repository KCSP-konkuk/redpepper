import os
"""피쳐 선택 → 튜닝 → 최종 시험 (2026-09-23)

검증 2016~2020 에서만 고르고, 시험 2021~2025 는 마지막에 한 번만 본다. 2026 은 추가 미관측 검증.
연도 y 평가 = 2001 ~ y-1 로 학습해 y 를 예측(확장 윈도우).

1) 피쳐 선택 (검증, 기존 파라미터 BP·K=80 고정, 시드 4)
   기준 B = 가격 + 교차 + 검색량(순) + 명절 + d_last1_vs_p1
   - 빼기: 교차·검색량·명절·d_last1 각각. 빼도 검증 MASE 가 +TOL 이상 나빠지지 않으면 뺀다(단순한 쪽 우선)
   - 더하기: 기상·물가지수·재배면적·반입량·출하지역·검색량 일별. -TOL 이상 좋아질 때만 넣는다
2) 튜닝 (검증) — 선택 구성 S 와 현행 C0 를 같은 후보(K 5개 × 파라미터 41개)로 각각
3) 시험 2021~2025 (시드 12) + 2026
"""
import random, json, time
from daily_anchor import *
_HERE=os.path.dirname(os.path.abspath(__file__))

VAL=(2016,2017,2018,2019,2020); TEST=(2021,2022,2023,2024,2025); EXTRA=(2026,)
TOL=0.012; N_PARAM=int(os.environ.get('N_PARAM',40)); VSEEDS=4; TSEEDS=12
P1=df.price.shift(1)

def search_daily(df):
    t=pd.read_csv(DATA_DIR+'pepper_trend.csv'); t=t[t.kw=='고춧가루']
    s=pd.Series(t.ratio.values,index=pd.to_datetime(t.period)).sort_index()
    soon_mean=s.groupby(to_soon(pd.Series(s.index)).values).mean()
    prev=df.DATE.shift(1).map(soon_mean)
    rows=[]
    for t0 in df.start:
        b=s[s.index<t0]
        rows.append(dict(m3=b.tail(3).mean() if len(b)>=3 else np.nan,
                         m7=b.tail(7).mean() if len(b)>=7 else np.nan,
                         p14=b.iloc[-21:-7].mean() if len(b)>=21 else np.nan))
    r=pd.DataFrame(rows,index=df.index); f=pd.DataFrame(index=df.index)
    f['sd_m3_vs_prev']=r.m3/prev; f['sd_m7_vs_prev']=r.m7/prev; f['sd_m7_vs_p14']=r.m7/r.p14
    return f

G={'price':f_price(df),'cross':f_cross_l(df),'trend':f_trend(df),'hol':f_hol_soon(df),
   'dlast':DF[['d_last1_vs_p1']],
   'weather':f_weather(df),'index':f_index(df),'prod':f_prod(df),
   'supply':f_supply_l(df),'origin':f_origin_l(df),'sdaily':search_daily(df)}
def X_of(groups): return pd.concat([G[g] for g in groups],axis=1)
def mase(R): return np.abs(R.pred-R.act).mean()/np.abs(R.nv-R.act).mean()
def val(groups,params=BP,k=K,seeds=VSEEDS):
    return mase(run(X_of(groups),P1,P1,years=VAL,seeds=seeds,params=params,k=k))

RESUME=os.environ.get('RESUME')
t0=time.time()
if not RESUME:
  print('=== 1) 피쳐 선택 (검증 2016~2020) ===',flush=True)
  B=['price','cross','trend','hol','dlast']
  b=val(B); print(f'기준 {B}: {b:.3f}',flush=True)
  sel=list(B)
  for g in ('cross','trend','hol','dlast'):
      m=val([x for x in B if x!=g]); d=m-b
      keep=d>=TOL; print(f'  - {g:<8} {m:.3f} ({d:+.3f}) → {"유지" if keep else "제거"}',flush=True)
      if not keep: sel.remove(g)
  for g in ('weather','index','prod','supply','origin','sdaily'):
      m=val(B+[g]); d=m-b
      add=d<=-TOL; print(f'  + {g:<8} {m:.3f} ({d:+.3f}) → {"추가" if add else "제외"}',flush=True)
      if add: sel.append(g)
  sv=val(sel); print(f'선택 구성 {sel}: {sv:.3f}  ({time.time()-t0:.0f}s)',flush=True)

else:
  sel=['price','cross','trend','dlast']
print('\n=== 2) 튜닝 (검증 2016~2020) ===',flush=True)
random.seed(11)
CANDS=[dict(max_depth=random.choice([2,3,4,5,6]),n_estimators=random.choice([400,800,1200]),
            learning_rate=random.choice([0.02,0.03,0.05,0.08]),subsample=random.choice([0.6,0.8,1.0]),
            colsample_bytree=random.choice([0.5,0.7,0.9]),min_child_weight=random.choice([1,3,5,8,12]),
            reg_lambda=random.choice([0.5,1.0,3.0,8.0])) for _ in range(N_PARAM)]+[BP]
CFG={'C0 현행':['price','cross','trend','hol'],'S 선택':sel}
best={}
for nm,groups in CFG.items():
    if RESUME:
        k_,i_=dict(x.split('=') for x in RESUME.split(','))[nm].split(':')
        best[nm]=dict(groups=groups,K=int(k_),params=CANDS[int(i_)],val=float('nan'))
        print(f'[{nm}] 재개: K {k_}, 후보 #{i_} {CANDS[int(i_)]}',flush=True); continue
    X=X_of(groups); ks=[(k,val(groups,k=k)) for k in (25,40,60,80,X.shape[1])]
    K_=min(ks,key=lambda x:x[1])[0]
    res=sorted((val(groups,params=g,k=K_),i) for i,g in enumerate(CANDS))
    best[nm]=dict(groups=groups,K=K_,params=CANDS[res[0][1]],val=res[0][0])
    print(f'[{nm}] K '+' '.join(f'{k}:{m:.3f}' for k,m in ks)+f' → {K_} | 파라미터 top3 '
          +' '.join(f'{m:.3f}#{i}' for m,i in res[:3])+f'  ({time.time()-t0:.0f}s)',flush=True)
json.dump(best,open(os.path.join(_HERE,'best_split.json'),'w'),ensure_ascii=False,indent=1)
# 참고: 선택 구성에서 검색량만 뺀 것(2016 이전 공백 문제). 별도 튜닝 없이 S 설정 그대로
best['S−검색량']=dict(best['S 선택'],groups=[g for g in sel if g!='trend'])

def report(title,years):
    print(f'\n=== {title} (검증에서 고른 설정, 시드 {TSEEDS}) ===',flush=True)
    R={}
    for nm,b in best.items():
        R[nm]=run(X_of(b['groups']),P1,P1,years=years,seeds=TSEEDS,params=b['params'],k=b['K'])
        p=panel(R[nm].pred.values,R[nm].act.values,R[nm].nv.values)
        yr=R[nm].groupby('yr').apply(mase)
        print(f'{nm:<7} '+' '.join(f'{k} {p[k]:.3f}' if k in ('R2','MASE','RMSSE') else f'{k} {p[k]:,.1f}'
              for k in ('R2','MAE','RMSE','MAPE','sMAPE','MdAPE','MASE','RMSSE','DA'))
              +' | 연도별 '+' '.join(f'{v:.2f}' for v in yr),flush=True)
        n=panel(R[nm].nv.values,R[nm].act.values,R[nm].nv.values)
    print('나이브   '+' '.join(f'{k} {n[k]:.3f}' if k in ('R2','MASE','RMSSE') else f'{k} {n[k]:,.1f}'
          for k in ('R2','MAE','RMSE','MAPE','sMAPE','MdAPE')))
    for nm in R:
        a=dm(R[nm].pred.values,R[nm].nv.values,R[nm].act.values,loss='abs')
        s=dm(R[nm].pred.values,R[nm].nv.values,R[nm].act.values,loss='sq')
        print(f'DM {nm} vs 나이브: abs t={a[0]:+.2f} p={a[1]:.2e} | sq t={s[0]:+.2f} p={s[1]:.2e}')
    a=dm_pair(R['S 선택'],R['C0 현행'],'abs'); s=dm_pair(R['S 선택'],R['C0 현행'],'sq')
    print(f'DM 선택 vs 현행: abs t={a[0]:+.2f} p={a[1]:.4f} | sq t={s[0]:+.2f} p={s[1]:.4f}  n={a[2]}',flush=True)

report('3) 시험 2021~2025',TEST)
report('추가 미관측 2026',EXTRA)
print(f'\n총 {time.time()-t0:.0f}s')
