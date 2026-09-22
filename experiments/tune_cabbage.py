import os
_HERE=os.path.dirname(os.path.abspath(__file__))
DATA_DIR=os.path.join(_HERE,'..','data')+os.sep
D=DATA_DIR
SP=DATA_DIR
D2=DATA_DIR
from report import dfc, Xc, YRS
from soon import *
import random, json
print('배추 피쳐',Xc.shape)
best=(9,None)
for K in (15,25,40,60,80):
    r=eval_soon(dfc,Xc,K=K,seeds=6,years=YRS,start=2018)
    print(f'  top{K:<4} MASE {r["MASE"]:.3f} MAPE {r["MAPE"]:5.2f}% R² {r["R2"]:.3f}')
    if r['MASE']<best[0]: best=(r['MASE'],K)
K=best[1]; print('배추 K =',K)
random.seed(11); res=[]
for _ in range(20):
    g=dict(max_depth=random.choice([2,3,4,5]),n_estimators=random.choice([400,800,1200]),
           learning_rate=random.choice([0.02,0.03,0.05,0.08]),subsample=random.choice([0.6,0.8,1.0]),
           colsample_bytree=random.choice([0.5,0.7,0.9]),min_child_weight=random.choice([1,3,5,8]),
           reg_lambda=random.choice([0.5,1.0,3.0,8.0]))
    r=eval_soon(dfc,Xc,K=K,seeds=4,params=g,years=YRS,start=2018); res.append((r['MASE'],r['MAPE'],g))
res.sort(key=lambda x:x[0])
for ms,mp,g in res[:3]: print(f'  MASE {ms:.3f} MAPE {mp:5.2f}% {g}')
bp=res[0][2]
r=eval_soon(dfc,Xc,K=K,seeds=12,params=bp,years=YRS,start=2018)
print(f'배추 최종: MASE {r["MASE"]:.3f} MAPE {r["MAPE"]:.2f}% R² {r["R2"]:.3f}  '+" ".join(f'{m:.2f}' for _,m,_ in r['yr']))
json.dump({'K':K,'params':bp},open(os.path.join(_HERE,'best_cabbage.json'),'w'))
