import os
_HERE=os.path.dirname(os.path.abspath(__file__))
DATA_DIR=os.path.join(_HERE,'..','data')+os.sep
D=DATA_DIR
SP=DATA_DIR
D2=DATA_DIR
from soon import *
import random, json
df=load_long()
X=pd.concat([f_price(df),f_cross_l(df),f_trend(df),f_hol_soon(df)],axis=1)
print('피쳐',X.shape)
print('--- K ---')
best=(9,None)
for K in (15,25,40,60,80,len(X.columns)):
    r=eval_soon(df,X,K=K,seeds=6)
    print(f'  top{K:<4} MASE {r["MASE"]:.3f} MAPE {r["MAPE"]:5.2f}% R² {r["R2"]:.3f}')
    if r['MASE']<best[0]: best=(r['MASE'],K)
K=best[1]; print('K =',K)
print('--- 파라미터 ---')
random.seed(7); res=[]
for _ in range(24):
    g=dict(max_depth=random.choice([2,3,4,5,6]),n_estimators=random.choice([400,800,1200]),
           learning_rate=random.choice([0.02,0.03,0.05,0.08]),subsample=random.choice([0.6,0.8,1.0]),
           colsample_bytree=random.choice([0.5,0.7,0.9]),min_child_weight=random.choice([1,3,5,8,12]),
           reg_lambda=random.choice([0.5,1.0,3.0,8.0]))
    r=eval_soon(df,X,K=K,seeds=4,params=g); res.append((r['MASE'],r['MAPE'],r['R2'],g))
res.sort(key=lambda x:x[0])
for ms,mp,r2,g in res[:4]: print(f'  MASE {ms:.3f} MAPE {mp:5.2f}% R² {r2:.3f} {g}')
bp=res[0][3]
r=eval_soon(df,X,K=K,seeds=12,params=bp); sh_s('최종(시드12)',r)
json.dump({'K':K,'params':bp},open(os.path.join(_HERE,'best_soon.json'),'w'))
from collections import Counter
c=Counter(x for s in r['sel'] for x in s); print('상위 피쳐:',[k for k,_ in c.most_common(18)])
