import os
"""daily_anchor.py 최선 구성(앵커 직전순 + 일별피쳐)의 강건성 확인
  1) 2026 홀드아웃 — 하이퍼파라미터 선택에 안 쓰인 해
  2) 일별 피쳐 하나씩 빼기(ablation)
  3) S2 순 중간 k일째
"""
from daily_anchor import *

P1=df.price.shift(1); X1=pd.concat([X0,DF],axis=1)

print('--- 1) 2026 홀드아웃 ---',flush=True)
R0=run(X0,P1,P1,years=(2026,)); summ('S0 2026',R0)
R1=run(X1,P1,P1,years=(2026,)); summ('p1+f 2026',R1)
a=dm_pair(R1,R0,'abs'); print(f'   DM abs t={a[0]:+.2f} p={a[1]:.4f} n={a[2]}',flush=True)

print('\n--- 2) ablation (2021~2025, 하나씩 빼기) ---',flush=True)
base=run(X1,P1,P1); summ('p1+f 전체',base)
for c in DF.columns:
    R=run(X1.drop(columns=c),P1,P1); summ(f'  - {c}',R)
for c in DF.columns:
    R=run(pd.concat([X0,DF[[c]]],axis=1),P1,P1); summ(f'  only {c}',R)

run_s2()
