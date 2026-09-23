import os
"""수입량 피쳐 효과 — select_tune_test 와 같은 판정(검증 2016~2020 ±0.012), 통과하면 시험 2021~2025 1회

수입 통계는 발표가 한두 달 늦다. 순 t(y년 m월)를 예측할 때 쓸 수 있는 건 **m-2월까지**로 둔다.
월 값을 그 달의 세 순에 펴고, 순 기준 시차 대신 '예측 시점에서 2·3·12개월 전 달'로 만든다.
"""
import json
from daily_anchor import *
_HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(_HERE, 'select_tune_test.py')).read().split("RESUME=os.environ.get('RESUME')")[0]
     .split("from daily_anchor import *")[1])

imp = pd.read_csv(DATA_DIR + 'import_pepper_monthly.csv', dtype={'ym': str})
LAG = 2   # 발표 지연(개월)


def f_import(df, series):
    piv = imp.pivot_table(index='ym', columns='series', values='ton', aggfunc='sum')
    s = piv[series].astype(float)
    # 첫 실적 이전의 0 은 '수입 없음'이 아니라 자료 없음(합계·냉동 2003~, 건고추·고춧가루는 HS 개정 2012~)
    s[s.index < s.index[s > 0].min()] = np.nan
    ym = df.year * 12 + df.month - 1
    def at(k):  # 예측 대상 달 기준 k개월 전 값
        t = ym - k
        return (t // 12).astype(str) + (t % 12 + 1).map('{:02d}'.format)
    f = pd.DataFrame(index=df.index)
    v = lambda k: at(k).map(s)
    f[f'imp_{series}_l{LAG}'] = v(LAG)
    f[f'imp_{series}_l{LAG + 1}'] = v(LAG + 1)
    f[f'imp_{series}_ma3'] = (v(LAG) + v(LAG + 1) + v(LAG + 2)) / 3
    f[f'imp_{series}_yoy'] = v(LAG) / v(LAG + 12)
    f[f'imp_{series}_mom'] = v(LAG) / v(LAG + 1)
    return f


best = json.load(open(os.path.join(_HERE, 'best_split.json')))
S = best['S 선택']; BASE = ['price', 'cross', 'trend', 'dlast']
for sname in ('total', 'frozen', 'dried', 'powder'):
    G[f'imp_{sname}'] = f_import(df, sname)

b = val(BASE, params=S['params'], k=S['K'])
print(f'기준(채택 구성, 채택 설정) 검증 {b:.3f}', flush=True)
add = []
for sname in ('total', 'frozen', 'dried', 'powder'):
    m = val(BASE + [f'imp_{sname}'], params=S['params'], k=S['K']); d = m - b
    ok = d <= -TOL
    print(f'  + imp_{sname:<7} {m:.3f} ({d:+.3f}) → {"추가" if ok else "제외"}', flush=True)
    if ok: add.append(f'imp_{sname}')

if add:
    groups = BASE + add
    print(f'\n통과: {add} → 시험 2021~2025 (시드 12, 채택 설정 그대로)', flush=True)
    R0 = run(X_of(BASE), P1, P1, years=TEST, seeds=TSEEDS, params=S['params'], k=S['K'])
    R1 = run(X_of(groups), P1, P1, years=TEST, seeds=TSEEDS, params=S['params'], k=S['K'])
    for nm, R in (('채택', R0), ('+수입량', R1)):
        p = panel(R.pred.values, R.act.values, R.nv.values)
        print(f'{nm} MASE {p["MASE"]:.3f} MAPE {p["MAPE"]:.2f}% R² {p["R2"]:.3f} DA {p["DA"]:.1f}%')
    a = dm_pair(R1, R0, 'abs'); print(f'DM +수입량 vs 채택: t={a[0]:+.2f} p={a[1]:.4f}')
else:
    print('\n검증에서 통과한 수입량 계열 없음 → 시험 구간은 보지 않는다')
