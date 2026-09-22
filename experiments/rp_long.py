import os
_HERE=os.path.dirname(os.path.abspath(__file__))
DATA_DIR=os.path.join(_HERE,'..','data')+os.sep
D=DATA_DIR
SP=DATA_DIR
D2=DATA_DIR
"""홍고추 장기표본(2001~) 전용 — 모든 소스 결합"""
from rp import *
import os
PSn={'상순':0,'중순':1,'하순':2}
def load_long():
    n=pd.read_csv(SP+'cross_long_홍고추.csv'); n['DATE']=n.DATE.astype(str)
    n=n[n.DATE<'202609하순']
    df=n.rename(columns={'val':'price','yearAvg':'ny','bfYear':'py'})[['DATE','price','ny','py']]
    for c in ('price','ny','py'): df[c]=pd.to_numeric(df[c],errors='coerce')
    f=SP+'area_long_홍고추.csv'
    if os.path.exists(f):
        a=pd.read_csv(f); a['DATE']=a.DATE.astype(str)
        df=df.merge(a,on='DATE',how='left')
    df['year']=df.DATE.str[:4].astype(int); df['month']=df.DATE.str[4:6].astype(int)
    df['pidx']=df.DATE.str[6:].map(PSn)
    return df.sort_values('DATE').reset_index(drop=True)
def f_cross_l(df,items=('풋고추','청피망'),lags=(1,2,3,6)):
    f=pd.DataFrame(index=df.index)
    for nm in items:
        c=pd.read_csv(SP+f'cross_long_{nm}.csv'); c['DATE']=c.DATE.astype(str)
        m=df[['DATE']].merge(c,on='DATE',how='left')
        v=pd.to_numeric(m.val,errors='coerce'); ya=pd.to_numeric(m.yearAvg,errors='coerce')
        by=pd.to_numeric(m.bfYear,errors='coerce')
        for l in lags: f[f'{nm}_l{l}']=v.shift(l)
        f[f'{nm}_ma3']=v.shift(1).rolling(3).mean()
        f[f'{nm}_mom1']=v.shift(1)/v.shift(2)-1; f[f'{nm}_mom3']=v.shift(1)/v.shift(4)-1
        f[f'{nm}_vs_ny']=v.shift(1)/ya.shift(1); f[f'{nm}_ratio']=df.price.shift(1)/v.shift(1)
        f[f'{nm}_ny_dir']=ya/ya.shift(1); f[f'{nm}_yoy']=v.shift(1)/by.shift(1)
    return f
def f_supply_l(df):
    f=pd.DataFrame(index=df.index)
    if 'sup' not in df: return f
    v=pd.to_numeric(df.sup,errors='coerce')
    for l in (1,2,3,6): f[f'slag{l}']=v.shift(l)
    f['sma3']=v.shift(1).rolling(3).mean(); f['smom1']=v.shift(1)/v.shift(2)-1
    f['s_vs_py']=v.shift(1)/pd.to_numeric(df.sup_py,errors='coerce').shift(1)
    f['s_vs_ny']=v.shift(1)/pd.to_numeric(df.sup_ny,errors='coerce').shift(1)
    f['sup_py']=pd.to_numeric(df.sup_py,errors='coerce')
    f['sup_ny_next']=pd.to_numeric(df.sup_ny,errors='coerce')
    f['ps_l1']=df.price.shift(1)/v.shift(1)
    return f
def f_origin_l(df):
    f=pd.DataFrame(index=df.index)
    if 'w1' not in df: return f
    w1=pd.to_numeric(df.w1,errors='coerce'); w2=pd.to_numeric(df.w2,errors='coerce')
    we=pd.to_numeric(df.wetc,errors='coerce'); tot=w1+w2+we
    f['hhi']=((w1/tot)**2+(w2/tot)**2).shift(1); f['share1']=(w1/tot).shift(1)
    f['w1_l1']=w1.shift(1); f['w1_mom']=w1.shift(1)/w1.shift(2)-1
    f['switch']=(df.r1!=df.r1.shift(1)).astype(float).shift(1)
    def reg(r):
        r=str(r)
        if '강원' in r: return 2
        if any(k in r for k in ('경기','충청','인천','서울')): return 1
        return 0
    f['r1code']=df.r1.map(reg).shift(1); f['r1chg']=f['r1code']-df.r1.map(reg).shift(2)
    return f
def build(df,use=('price','cross','trend','weather','index','prod','supply','origin')):
    parts=[]
    if 'price' in use: parts.append(f_price(df))
    if 'cross' in use: parts.append(f_cross_l(df))
    if 'trend' in use: parts.append(f_trend(df))
    if 'weather' in use: parts.append(f_weather(df))
    if 'index' in use: parts.append(f_index(df))
    if 'prod' in use: parts.append(f_prod(df))
    if 'supply' in use: parts.append(f_supply_l(df))
    if 'origin' in use: parts.append(f_origin_l(df))
    return pd.concat(parts,axis=1).replace([np.inf,-np.inf],np.nan)
def eval_long(df,X,K=25,seeds=8,params=None,start=2001,years=YEARS,ret=False):
    tgt,prevm=monthly(df); A=anchor_of(df,'lag'); cols=list(X.columns); acc={}
    for y in years:
        tr=(df.year<y)&(df.year>=start)&tgt.notna()&A.notna()
        te=(df.year==y)&tgt.notna()&prevm.notna()&A.notna()
        t=(tgt/A)[tr]; ok=(t.notna()&np.isfinite(t)).values
        m0=make_model('xgb',params,0); m0.fit(X.loc[tr,cols][ok],t[ok])
        sel=list(pd.Series(m0.feature_importances_,index=cols).sort_values(ascending=False).index[:K])
        ps=[]
        for s in range(seeds):
            m=make_model('xgb',params,s); m.fit(X.loc[tr,sel][ok],t[ok])
            ps.append(m.predict(X.loc[te,sel])*A[te].values)
        pred=np.mean(ps,axis=0); act=tgt[te].values; nv=prevm[te].values
        acc.setdefault('m',[]).append(np.mean(np.abs(pred-act)))
        acc.setdefault('n',[]).append(np.mean(np.abs(nv-act)))
        acc.setdefault('mape',[]).append(np.mean(np.abs(pred-act)/act)*100)
        acc.setdefault('yr',[]).append((y,np.mean(np.abs(pred-act))/np.mean(np.abs(nv-act)),
                                        np.mean(np.abs(pred-act)/act)*100))
        acc.setdefault('sel',[]).append(sel)
    r=dict(mase=np.mean(acc['m'])/np.mean(acc['n']),mape=np.mean(acc['mape']),yr=acc['yr'],sel=acc['sel'])
    return r
def sh(n,r): print(f'{n:<26} MASE {r["mase"]:.3f}  MAPE {r["mape"]:5.1f}%  '+" ".join(f'{m:.2f}' for _,m,_ in r['yr']))
