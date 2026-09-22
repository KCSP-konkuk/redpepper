import os
_HERE=os.path.dirname(os.path.abspath(__file__))
DATA_DIR=os.path.join(_HERE,'..','data')+os.sep
D=DATA_DIR
SP=DATA_DIR
D2=DATA_DIR
"""홍고추 1순 앞 예측 — 장기표본 전용 틀 + 논문 지표 패널"""
from rp_long import *
from scipy import stats
SEOL={y:m for y,m in zip(range(2001,2027),
  ['0124','0212','0201','0122','0209','0129','0218','0207','0126','0214','0203','0123','0210',
   '0131','0219','0208','0128','0216','0205','0125','0212','0201','0122','0210','0129','0217'])}
CHU={y:m for y,m in zip(range(2001,2027),
  ['1001','0921','0911','0928','0918','1006','0925','0914','1003','0922','0912','0930','0919',
   '0908','0927','0915','1004','0924','0913','1001','0921','0910','0929','0917','1006','0925'])}
def f_hol_soon(df):
    """대상 순(다음 순)에 명절이 걸리는지 — 사전 인지 가능"""
    f=pd.DataFrame(index=df.index)
    k=(df.year*36+(df.month-1)*3+df.pidx)+1
    ny=k//36; nm=(k%36)//3+1; npi=(k%36)%3
    def hit(tbl):
        out=[]
        for y,m,p in zip(ny,nm,npi):
            d=tbl.get(int(y),'')
            if d and int(d[:2])==int(m):
                day=int(d[2:]); pp=0 if day<=10 else 1 if day<=20 else 2
                out.append(1 if pp==p else 0)
            else: out.append(0)
        return out
    f['tgt_seol']=hit(SEOL); f['tgt_chu']=hit(CHU)
    f['tgt_pre_chu']=pd.Series(hit(CHU)).shift(-1).fillna(0).values
    f['tgt_month']=nm; f['tgt_harvest']=np.isin(nm,[8,9,10]).astype(int)
    f['tgt_kimjang']=np.isin(nm,[11,12]).astype(int)
    return f
def eval_soon(df,X,K=25,seeds=8,params=None,years=YEARS,start=2001,ret_pred=False):
    X=X.replace([np.inf,-np.inf],np.nan); cols=list(X.columns)
    p=df.price; A=p.shift(1); acc={}; P=[];AC=[];NV=[];YR=[]
    for y in years:
        tr=(df.year<y)&(df.year>=start)&p.notna()&A.notna(); te=(df.year==y)&p.notna()&A.notna()
        if te.sum()==0: continue
        t=(p/A)[tr]; ok=(t.notna()&np.isfinite(t)).values
        m0=make_model('xgb',params,0); m0.fit(X.loc[tr,cols][ok],t[ok])
        sel=list(pd.Series(m0.feature_importances_,index=cols).sort_values(ascending=False).index[:K])
        ps=[make_model('xgb',params,s).fit(X.loc[tr,sel][ok],t[ok]).predict(X.loc[te,sel])*A[te].values
            for s in range(seeds)]
        pred=np.mean(ps,axis=0); act=p[te].values; nv=A[te].values
        P.append(pred); AC.append(act); NV.append(nv); YR+= [y]*len(pred)
        acc.setdefault('yr',[]).append((y,np.mean(np.abs(pred-act))/np.mean(np.abs(nv-act)),
                                        np.mean(np.abs(pred-act)/act)*100))
        acc.setdefault('sel',[]).append(sel)
    P=np.concatenate(P);AC=np.concatenate(AC);NV=np.concatenate(NV)
    m=panel(P,AC,NV); m['yr']=acc['yr']; m['sel']=acc['sel']
    return (m,(P,AC,NV,np.array(YR))) if ret_pred else m
def panel(pred,act,naive):
    e=pred-act; ae=np.abs(e); ne=np.abs(naive-act)
    return dict(R2=1-np.sum(e**2)/np.sum((act-act.mean())**2), MAE=ae.mean(),
                RMSE=np.sqrt((e**2).mean()), MAPE=np.mean(ae/act)*100,
                sMAPE=np.mean(2*ae/(np.abs(pred)+act))*100, MdAPE=np.median(ae/act)*100,
                MASE=ae.mean()/ne.mean(), RMSSE=np.sqrt((e**2).mean())/np.sqrt(((naive-act)**2).mean()),
                TheilU=np.sqrt((e**2).mean())/np.sqrt(((naive-act)**2).mean()),
                DA=np.mean(np.sign(pred-naive)==np.sign(act-naive))*100)
def dm(p1,p2,act,h=1,loss='abs'):
    d=(np.abs(p1-act)-np.abs(p2-act)) if loss=='abs' else ((p1-act)**2-(p2-act)**2)
    n=len(d); dbar=d.mean(); g0=d.var(ddof=0); var=g0/n
    s=dbar/np.sqrt(var); hln=s*np.sqrt((n+1-2*h+h*(h-1)/n)/n)
    return hln, 2*(1-stats.t.cdf(abs(hln),df=n-1))
def sh_s(n,r): print(f'{n:<28} MASE {r["MASE"]:.3f}  MAPE {r["MAPE"]:5.2f}%  R² {r["R2"]:.3f}  '
                     +" ".join(f'{m:.2f}' for _,m,_ in r['yr']))
