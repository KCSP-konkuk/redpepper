import os
_HERE=os.path.dirname(os.path.abspath(__file__))
DATA_DIR=os.path.join(_HERE,'..','data')+os.sep
D=DATA_DIR
SP=DATA_DIR
D2=DATA_DIR
"""홍고추 '다음 달 평균' 예측 전용 실험 틀 (throwaway)
   - 나이브 = 전월 평균 (공정 기준)
   - 연도별 홀드아웃 2021~2026, 시드 평균
"""
import numpy as np, pandas as pd, xgboost as xgb


PSn={'상순':0,'중순':1,'하순':2}
YEARS=(2021,2022,2023,2024,2025,2026)
PARAMS=dict(n_estimators=400,max_depth=3,learning_rate=0.05,subsample=0.8,
            colsample_bytree=0.8,min_child_weight=3,reg_lambda=1.0,
            objective='reg:absoluteerror')

def to_soon(dates):
    d=pd.to_datetime(dates)
    return d.dt.strftime('%Y%m')+d.dt.day.map(lambda x:'상순' if x<=10 else '중순' if x<=20 else '하순')

def load():
    n=pd.read_csv(SP+'cross_홍고추.csv'); n['DATE']=n.DATE.astype(str)
    n=n[n.DATE<'202609하순']
    df=n.rename(columns={'val':'price','yearAvg':'ny','bfYear':'py'})[['DATE','price','ny','py']]
    for c in ('price','ny','py'): df[c]=pd.to_numeric(df[c],errors='coerce')
    o=pd.read_csv(D+'hist_supply_redpepper.csv',encoding='utf-8-sig'); o['DATE']=o.DATE.astype(str)
    df=df.merge(o[['DATE','총반입량','전년']].rename(columns={'총반입량':'sup','전년':'sup_py'}),on='DATE',how='left')
    g=pd.read_csv(D+'hist_origin_redpepper.csv',encoding='utf-8-sig'); g['DATE']=g.DATE.astype(str)
    df=df.merge(g[['DATE','1위산지','1위물량','2위산지','2위물량','기타물량']].rename(
        columns={'1위산지':'r1','1위물량':'w1','2위산지':'r2','2위물량':'w2','기타물량':'wetc'}),on='DATE',how='left')
    df['year']=df.DATE.str[:4].astype(int); df['month']=df.DATE.str[4:6].astype(int)
    df['pidx']=df.DATE.str[6:].map(PSn)
    return df.sort_values('DATE').reset_index(drop=True)

def monthly(df):
    m=df.groupby(['year','month']).price.mean().reset_index().rename(columns={'price':'mavg'})
    m['ym']=m.year*12+m.month; lut=dict(zip(m.ym,m.mavg))
    return (df.year*12+df.month+1).map(lut), (df.year*12+df.month-1).map(lut)

def anchor_of(df,kind):
    p=df.price
    return {'lag':p.shift(1),'ma3':p.shift(1).rolling(3).mean(),
            'ma2':p.shift(1).rolling(2).mean(),'prevm':monthly(df)[1]}[kind]

def evaluate(df,X,cols,anchor='lag',logt=False,seeds=8,years=YEARS,alpha=1.0,
             params=None,model='xgb',ret_pred=False):
    X=X.replace([np.inf,-np.inf],np.nan)
    tgt,prevm=monthly(df); A=anchor_of(df,anchor)
    acc={}; preds={}
    for y in years:
        tr=(df.year<y)&tgt.notna()&A.notna(); te=(df.year==y)&tgt.notna()&prevm.notna()&A.notna()
        if te.sum()==0: continue
        t=(tgt/A)[tr]
        if logt: t=np.log(t)
        ok=(t.notna()&np.isfinite(t)).values
        ps=[]
        for s in range(seeds):
            m=make_model(model,params,s)
            xt=X.loc[tr,cols][ok]; xe=X.loc[te,cols]
            if model!='xgb':
                med=xt.median()
                xt=xt.fillna(med).fillna(0); xe=xe.fillna(med).fillna(0)
            m.fit(xt,t[ok])
            q=m.predict(xe)
            ps.append((np.exp(q) if logt else q)*A[te].values)
        pred=np.mean(ps,axis=0); act=tgt[te].values; nv=prevm[te].values
        pred=alpha*pred+(1-alpha)*nv
        preds[y]=(df.DATE[te].values,pred,act,nv)
        acc.setdefault('m',[]).append(np.mean(np.abs(pred-act)))
        acc.setdefault('n',[]).append(np.mean(np.abs(nv-act)))
        acc.setdefault('mape',[]).append(np.mean(np.abs(pred-act)/act)*100)
        acc.setdefault('nmape',[]).append(np.mean(np.abs(nv-act)/act)*100)
        acc.setdefault('yr',[]).append((y,np.mean(np.abs(pred-act))/np.mean(np.abs(nv-act)),
                                        np.mean(np.abs(pred-act)/act)*100))
    r=dict(mase=np.mean(acc['m'])/np.mean(acc['n']),mape=np.mean(acc['mape']),
           nmape=np.mean(acc['nmape']),yr=acc['yr'])
    return (r,preds) if ret_pred else r

def make_model(kind,params,seed):
    pr=dict(PARAMS); pr.update(params or {})
    if kind=='xgb': return xgb.XGBRegressor(**pr,random_state=seed)
    if kind=='lgb':
        import lightgbm as lgb
        return lgb.LGBMRegressor(n_estimators=pr.get('n_estimators',400),max_depth=-1,
            learning_rate=pr.get('learning_rate',0.05),num_leaves=pr.get('num_leaves',15),
            subsample=0.8,colsample_bytree=0.8,min_child_samples=8,objective='mae',
            random_state=seed,verbose=-1)
    if kind=='rf':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=400,max_depth=8,min_samples_leaf=3,
                                     random_state=seed,n_jobs=-1)
    if kind=='ridge':
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(),Ridge(alpha=params.get('alpha',10) if params else 10))
    if kind=='huber':
        from sklearn.linear_model import HuberRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(),HuberRegressor(alpha=1e-3,max_iter=500))
    raise ValueError(kind)

def show(name,r):
    print(f"{name:<28} MASE {r['mase']:.3f}  MAPE {r['mape']:5.1f}%  (나이브 {r['nmape']:4.1f}%)  "
          +" ".join(f"{m:.2f}" for _,m,_ in r['yr']))

# ---------------- 피쳐 빌더들
def f_price(df):
    f=pd.DataFrame(index=df.index); p=df.price
    for l in (1,2,3,4,6,9,12,36): f[f'plag{l}']=p.shift(l)
    for w in (2,3,6,12): f[f'pma{w}']=p.shift(1).rolling(w).mean()
    f['pmom1']=p.shift(1)/p.shift(2)-1; f['pmom3']=p.shift(1)/p.shift(4)-1
    f['pstd3']=p.shift(1).rolling(3).std(); f['pstd6']=p.shift(1).rolling(6).std()
    f['pmin6']=p.shift(1).rolling(6).min(); f['pmax6']=p.shift(1).rolling(6).max()
    f['ny']=df.ny; f['py']=df.py
    f['lag_vs_ny']=p.shift(1)/df.ny; f['p_vs_ny']=p.shift(1)/df.ny.shift(1)
    f['ny_ratio']=df.ny/df.ny.shift(1); f['py_ratio']=df.py/df.py.shift(1)
    f['pyoy']=p.shift(1)/p.shift(37)
    # 다음 달 평년/전년(달력 기준) — 사전에 아는 값
    nx=df.year*12+df.month+1
    nym=df.assign(ym=df.year*12+df.month).groupby('ym').ny.mean()
    pym=df.assign(ym=df.year*12+df.month).groupby('ym').py.mean()
    f['ny_next']=nx.map(nym).values; f['py_next']=nx.map(pym).values
    f['ny_next_ratio']=f['ny_next']/p.shift(1)
    f['msin']=np.sin(2*np.pi*df.month/12); f['mcos']=np.cos(2*np.pi*df.month/12)
    kk=(df.month-1)*3+df.pidx
    f['ksin']=np.sin(2*np.pi*kk/36); f['kcos']=np.cos(2*np.pi*kk/36)
    f['pidx']=df.pidx; f['month']=df.month
    return f

def f_cross(df,items=('풋고추','청피망'),lags=(1,2,3,6)):
    f=pd.DataFrame(index=df.index)
    for nm in items:
        c=pd.read_csv(SP+f'cross_{nm}.csv'); c['DATE']=c.DATE.astype(str)
        m=df[['DATE']].merge(c,on='DATE',how='left')
        v=pd.to_numeric(m.val,errors='coerce'); ya=pd.to_numeric(m.yearAvg,errors='coerce')
        by=pd.to_numeric(m.bfYear,errors='coerce')
        for l in lags: f[f'{nm}_l{l}']=v.shift(l)
        f[f'{nm}_ma3']=v.shift(1).rolling(3).mean()
        f[f'{nm}_mom1']=v.shift(1)/v.shift(2)-1; f[f'{nm}_mom3']=v.shift(1)/v.shift(4)-1
        f[f'{nm}_vs_ny']=v.shift(1)/ya.shift(1); f[f'{nm}_ratio']=df.price.shift(1)/v.shift(1)
        f[f'{nm}_ratio_mom']=(df.price.shift(1)/v.shift(1))/(df.price.shift(2)/v.shift(2))-1
        f[f'{nm}_ny_dir']=ya/ya.shift(1); f[f'{nm}_yoy']=v.shift(1)/by.shift(1)
    return f

def f_trend(df,kws=('고춧가루',),lags=(1,2,3,6,12)):
    t=pd.read_csv(SP+'pepper_trend.csv'); t['DATE']=to_soon(t.period)
    piv=t.pivot_table(index='DATE',columns='kw',values='ratio',aggfunc='mean').reset_index()
    m=df[['DATE']].merge(piv,on='DATE',how='left'); f=pd.DataFrame(index=df.index)
    for kw in kws:
        v=pd.to_numeric(m[kw],errors='coerce')
        for l in lags: f[f'sr_{kw}_l{l}']=v.shift(l)
        f[f'sr_{kw}_ma3']=v.shift(1).rolling(3).mean()
        f[f'sr_{kw}_mom']=v.shift(1)/v.shift(2)-1
        f[f'sr_{kw}_yoy']=v.shift(1)/v.shift(37)
    return f

def f_supply(df):
    f=pd.DataFrame(index=df.index); v=pd.to_numeric(df.sup,errors='coerce')
    for l in (1,2,3,6): f[f'slag{l}']=v.shift(l)
    f['sma3']=v.shift(1).rolling(3).mean(); f['smom1']=v.shift(1)/v.shift(2)-1
    f['s_vs_py']=v.shift(1)/pd.to_numeric(df.sup_py,errors='coerce').shift(1)
    f['sup_py']=pd.to_numeric(df.sup_py,errors='coerce')
    f['ps_l1']=df.price.shift(1)/v.shift(1)
    return f

def f_origin(df):
    f=pd.DataFrame(index=df.index); tot=df.w1+df.w2+df.wetc
    f['hhi']=((df.w1/tot)**2+(df.w2/tot)**2).shift(1); f['share1']=(df.w1/tot).shift(1)
    f['w1_l1']=df.w1.shift(1); f['w1_mom']=df.w1.shift(1)/df.w1.shift(2)-1
    f['switch']=(df.r1!=df.r1.shift(1)).astype(float).shift(1)
    def reg(r):
        r=str(r)
        if '강원' in r: return 2
        if any(k in r for k in ('경기','충청','인천','서울')): return 1
        return 0
    f['r1code']=df.r1.map(reg).shift(1); f['r1chg']=f['r1code']-df.r1.map(reg).shift(2)
    return f

WCOLS=['TA','TMAX','TMIN','TS','TG','HM','SS','SI','CA','WSMAX','RNsum','RNmax',
       'hot','rainy','heavy','cold','trange']
def f_weather(df,lags=(1,3,6,9)):
    import weather as W
    g=W.soon_table(); ss=W.season_series(g,WCOLS)
    m=df[['DATE']].merge(ss,on='DATE',how='left'); f=pd.DataFrame(index=df.index)
    for c in WCOLS:
        v=pd.to_numeric(m[c],errors='coerce')
        for l in lags: f[f'w_{c}_l{l}']=v.shift(l)
    return f

def f_index(df,kws=('PPI_건고추','PPI_고추가루'),lags=(3,6,9)):
    ix=pd.read_csv(SP+'pepper_index.csv'); ix['ym']=ix.prd.astype(str)
    piv=ix.pivot_table(index='ym',columns='kw',values='val',aggfunc='first')
    m=df.assign(ym=df.DATE.str[:6])[['ym']].merge(piv,left_on='ym',right_index=True,how='left')
    f=pd.DataFrame(index=df.index)
    for kw in kws:
        v=pd.to_numeric(m[kw],errors='coerce').ffill()
        for l in lags: f[f'{kw}_l{l}']=v.shift(l)
        f[f'{kw}_mom']=v.shift(3)/v.shift(6)-1
    return f

def f_prod(df):
    pr=pd.read_csv(SP+'pepper_prod.csv'); pr['year']=pr.year.astype(int)
    pr['val']=pd.to_numeric(pr.val,errors='coerce')
    piv=pr.pivot_table(index='year',columns=['region','item'],values='val')
    piv.columns=[f'{r}_{i}' for r,i in piv.columns]
    f=pd.DataFrame(index=df.index)
    for c in ['전국_건고추_면적','전국_건고추_단수','전국_건고추_생산량']:
        s=piv[c]
        f[f'{c}_prev']=df.year.map(lambda y: s.get(y-1,np.nan)).values
        f[f'{c}_yoy']=df.year.map(lambda y: s.get(y-1,np.nan)/s.get(y-2,np.nan)).values
    return f
