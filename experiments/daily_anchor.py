import os
_HERE=os.path.dirname(os.path.abspath(__file__))
DATA_DIR=os.path.join(_HERE,'..','data')+os.sep
"""일별 가격으로 앵커를 바꾸면 나아지는가 — 1순 앞, 홀드아웃 2021~2025(+2026)

S0  현행: 앵커 = 직전 순 평균, 피쳐 = report.py 최종 구성
S1  예측 대상 순을 한 날도 보지 않는 조건. 앵커만 '직전 순 안의 최신 일별'로 교체(+일별 파생 피쳐)
S2  순 중간 k일째 예측(서비스는 매일 돈다). 대상 순의 지난 날짜가 보이므로 **자기 나이브도 같은 정보로**

MASE 는 두 가지로 적는다.
  MASE_old = MAE / MAE(직전 순 평균)  — 구성 간 공통 분모, 절대 개선량
  MASE_own = MAE / MAE(같은 정보의 나이브) — 모델이 정보 이상으로 보태는 몫
"""
import sys, json
from soon import *
from scipy import stats

cfg=json.load(open(os.path.join(_HERE,'best_soon.json'))); BP=cfg['params']; K=cfg['K']
YRS=(2021,2022,2023,2024,2025)
SEEDS=int(os.environ.get('SEEDS',12))

df=load_long()
X0=pd.concat([f_price(df),f_cross_l(df),f_trend(df),f_hol_soon(df)],axis=1)

dly=pd.read_csv(DATA_DIR+'daily_홍고추.csv',parse_dates=['date']).dropna(subset=['상'])
dly['DATE']=to_soon(dly.date).values
start={d:pd.Timestamp(int(d[:4]),int(d[4:6]),{'상순':1,'중순':11,'하순':21}[d[6:]]) for d in df.DATE}
df['start']=df.DATE.map(start)

def before(t0,n):
    """t0(순 시작일) 이전 마지막 n 거래일 상 가격"""
    s=dly[dly.date<t0].tail(n)['상']
    return s

def daily_feats(df):
    f=pd.DataFrame(index=df.index); p1=df.price.shift(1)
    rows=[]
    for t0 in df.start:
        b=before(t0,10); rows.append(dict(
            d_last1=b.iloc[-1] if len(b) else np.nan,
            d_m3=b.tail(3).mean() if len(b) else np.nan,
            d_m5=b.tail(5).mean() if len(b) else np.nan,
            d_std5=b.tail(5).std() if len(b)>2 else np.nan,
            d_gap=(t0-dly[dly.date<t0].date.iloc[-1]).days if len(b) else np.nan))
    r=pd.DataFrame(rows,index=df.index)
    for c in ('d_last1','d_m3','d_m5'): f[c+'_vs_p1']=r[c]/p1
    f['d_last_vs_m5']=r.d_last1/r.d_m5; f['d_cv5']=r.d_std5/r.d_m5; f['d_gap']=r.d_gap
    return f,r

DF,DR=daily_feats(df)

def run(X,A,naive,years=YRS,seeds=SEEDS,params=BP,k=K,start=2001):
    """eval_soon 과 같되 앵커·나이브를 인자로. 반환: 예측·실측·나이브·기존나이브·연도"""
    X=X.replace([np.inf,-np.inf],np.nan); cols=list(X.columns); p=df.price; P1=p.shift(1)
    out=[]
    for y in years:
        base=p.notna()&A.notna()&P1.notna()&naive.notna()
        tr=(df.year<y)&(df.year>=start)&base; te=(df.year==y)&base
        if te.sum()==0: continue
        t=(p/A)[tr]; ok=(t.notna()&np.isfinite(t)).values
        m0=make_model('xgb',params,0); m0.fit(X.loc[tr,cols][ok],t[ok])
        sel=list(pd.Series(m0.feature_importances_,index=cols).sort_values(ascending=False).index[:k])
        pr=np.mean([make_model('xgb',params,s).fit(X.loc[tr,sel][ok],t[ok]).predict(X.loc[te,sel])
                    *A[te].values for s in range(seeds)],axis=0)
        out.append(pd.DataFrame(dict(pred=pr,act=p[te].values,nv=naive[te].values,
                                     nv0=P1[te].values,yr=y),index=df.index[te]))
    return pd.concat(out)

def summ(name,R):
    e=np.abs(R.pred-R.act); m=dict(
        MASE_old=e.mean()/np.abs(R.nv0-R.act).mean(),
        MASE_own=e.mean()/np.abs(R.nv-R.act).mean(),
        MAPE=(e/R.act).mean()*100, R2=1-((R.pred-R.act)**2).sum()/((R.act-R.act.mean())**2).sum(),
        DA=np.mean(np.sign(R.pred-R.nv0)==np.sign(R.act-R.nv0))*100,
        nvMAPE=(np.abs(R.nv-R.act)/R.act).mean()*100)
    yr=R.groupby('yr').apply(lambda g:np.abs(g.pred-g.act).mean()/np.abs(g.nv0-g.act).mean())
    print(f'{name:<30} MASE_old {m["MASE_old"]:.3f}  MASE_own {m["MASE_own"]:.3f}  MAPE {m["MAPE"]:5.2f}%'
          f'  R² {m["R2"]:.3f}  DA {m["DA"]:4.1f}%  (자기나이브 MAPE {m["nvMAPE"]:5.2f}%)  '
          +' '.join(f'{v:.2f}' for v in yr), flush=True)
    return m

def dm_pair(Ra,Rb,loss='abs'):
    """같은 대상 순에서 a 가 b 보다 나은가(음수 t = a 우세)"""
    j=Ra.join(Rb,rsuffix='_b',how='inner')
    return dm(j.pred.values,j.pred_b.values,j.act.values,loss=loss)+(len(j),)

def partial_feats(k):
    """순 시작 후 k일째 새벽 시점(시작일+k-1 까지 관측). 나이브 = 관측일은 실제가, 남은 날(일요일 제외)은
    최신가 유지로 채운 순평균. 대상 순 거래일 수는 미리 모르므로 달력의 비일요일 수로 가중한다."""
    rows=[]
    for i,t0 in enumerate(df.start):
        end=t0+pd.offsets.MonthEnd(0) if t0.day==21 else t0+pd.Timedelta(days=9)
        cut=t0+pd.Timedelta(days=k)                        # cut 이전(미만)만 관측
        obs=dly[(dly.date>=t0)&(dly.date<cut)]['상']
        last=obs.iloc[-1] if len(obs) else DR.d_last1.iloc[i]
        rem=pd.date_range(max(cut,t0),end); rem=int((rem.dayofweek!=6).sum())
        nv=(obs.sum()+rem*last)/(len(obs)+rem) if (len(obs)+rem) else last
        rows.append(dict(nv=nv,n_obs=len(obs),rem=rem,
                         po_mean=obs.mean() if len(obs) else np.nan,
                         po_last=obs.iloc[-1] if len(obs) else np.nan,
                         po_first=obs.iloc[0] if len(obs) else np.nan))
    r=pd.DataFrame(rows,index=df.index); P1=df.price.shift(1)
    f=pd.DataFrame(index=df.index)
    f['k_nobs']=r.n_obs; f['k_rem']=r.rem; f['k_frac']=r.n_obs/(r.n_obs+r.rem)
    f['k_mean_vs_p1']=r.po_mean/P1; f['k_last_vs_p1']=r.po_last/P1
    f['k_trend']=r.po_last/r.po_first; f['k_nv_vs_p1']=r.nv/P1
    return f,r.nv

def run_s2(ks=(0,3,5,7,9)):
    P1=df.price.shift(1); X1=pd.concat([X0,DF],axis=1); out={}
    print('\n--- S2 순 중간 k일째 (나이브 = 관측분 + 최신가 유지) ---')
    for k in ks:
        F,NV=partial_feats(k)
        Rn=pd.DataFrame(dict(pred=NV,act=df.price,nv=NV,nv0=P1,yr=df.year))
        Rn=Rn[Rn.yr.isin(YRS)].dropna(); summ(f'k={k} 나이브만',Rn)
        R=run(pd.concat([X1,F],axis=1),NV,NV); summ(f'k={k} 모델(앵커=나이브)',R)
        a=dm(R.pred.values,R.nv.values,R.act.values,loss='abs')
        print(f'      모델 vs 자기나이브 DM abs t={a[0]:+.2f} p={a[1]:.4f}',flush=True)
        out[k]=R
    return out

if __name__=='__main__' and os.environ.get('S2'):
    run_s2(); sys.exit()

if __name__=='__main__':
    P1=df.price.shift(1)
    print(f'daily coverage: {DR.d_last1.notna().sum()} / {len(df)} soons, seeds {SEEDS}\n')

    print('--- 나이브만(모델 없음) ---')
    for c in ('d_last1','d_m3','d_m5'):
        R=pd.DataFrame(dict(pred=DR[c],act=df.price,nv=DR[c],nv0=P1,yr=df.year))
        R=R[R.yr.isin(YRS)].dropna(); summ(f'naive {c}',R)

    print('\n--- S0 / S1 ---')
    R0=run(X0,P1,P1); summ('S0 현행(앵커 직전순)',R0)
    res={}
    for c in ('d_last1','d_m3','d_m5'):
        res[c]=run(X0,DR[c],P1); summ(f'S1 앵커 {c}',res[c])
    X1=pd.concat([X0,DF],axis=1)
    for c in ('p1','d_last1','d_m3','d_m5'):
        A=P1 if c=='p1' else DR[c]
        res[c+'+f']=run(X1,A,P1); summ(f'S1 앵커 {c} + 일별피쳐',res[c+'+f'])

    print('\n--- S1 vs S0 쌍대 DM (음수 = S1 우세) ---')
    for nm,R in res.items():
        a=dm_pair(R,R0,'abs'); s=dm_pair(R,R0,'sq')
        print(f'{nm:<14} abs t={a[0]:+.2f} p={a[1]:.4f} | sq t={s[0]:+.2f} p={s[1]:.4f}  n={a[2]}')
