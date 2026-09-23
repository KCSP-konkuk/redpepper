"""양파 1순 앞 — 홍고추 레시피(2001~ 장기표본, 비율 타깃, 학습마다 상위 K, 시드 평균) 적용 (진행 중)

판정은 redpepper select_tune_test 와 같다: 검증 2016~2020 에서만 고르고 시험 2021~2025 1회, 2026 추가.
나이브 = 직전 순 가격. MASE < 1 이어야 값어치가 있다.
"""
import os, sys, json, random, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser('~/redpepper/experiments'))
from rp import f_price, make_model, PSn          # noqa: E402
from soon import panel, dm                       # noqa: E402

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'onion')
VAL = (2016, 2017, 2018, 2019, 2020); TEST = (2021, 2022, 2023, 2024, 2025); EXTRA = (2026,)
VSEEDS, TSEEDS, TOL = 4, 12, 0.012
BP = json.load(open(os.path.expanduser('~/redpepper/experiments/best_split.json')))['S 선택']['params']


def load():
    d = pd.read_csv(os.path.join(HERE, 'soon_onion.csv')); d['DATE'] = d.DATE.astype(str)
    d = d[d.DATE < '202609하순']                                   # 진행 중인 순 제외
    a = pd.read_csv(os.path.join(HERE, 'area_onion.csv')); a['DATE'] = a.DATE.astype(str)
    df = d.rename(columns={'val': 'price', 'yearAvg': 'ny', 'bfYear': 'py'}).merge(a, on='DATE', how='left')
    for c in ('price', 'ny', 'py', 'sup', 'sup_py', 'sup_ny'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['year'] = df.DATE.str[:4].astype(int); df['month'] = df.DATE.str[4:6].astype(int)
    df['pidx'] = df.DATE.str[6:].map(PSn)
    return df.sort_values('DATE').reset_index(drop=True)


def f_supply(df):
    f = pd.DataFrame(index=df.index); v = df.sup
    for l in (1, 2, 3, 6): f[f'slag{l}'] = v.shift(l)
    f['sma3'] = v.shift(1).rolling(3).mean(); f['smom1'] = v.shift(1) / v.shift(2) - 1
    f['s_vs_py'] = v.shift(1) / df.sup_py.shift(1); f['s_vs_ny'] = v.shift(1) / df.sup_ny.shift(1)
    f['sup_py'] = df.sup_py; f['sup_ny_next'] = df.sup_ny            # 평년·전년은 미리 아는 값
    f['ps_l1'] = df.price.shift(1) / v.shift(1)
    return f


df = load()
P1 = df.price.shift(1)
G = {'price': f_price(df), 'supply': f_supply(df)}
EXTRA_GROUPS = {}


def X_of(groups):
    return pd.concat([G.get(g, EXTRA_GROUPS.get(g)) for g in groups], axis=1).replace([np.inf, -np.inf], np.nan)


def run(X, years, seeds, params, k, start=2001):
    cols = list(X.columns); p = df.price; out = []
    for y in years:
        base = p.notna() & P1.notna()
        tr = (df.year < y) & (df.year >= start) & base; te = (df.year == y) & base
        if te.sum() == 0: continue
        t = (p / P1)[tr]; ok = (t.notna() & np.isfinite(t)).values
        m0 = make_model('xgb', params, 0).fit(X.loc[tr, cols][ok], t[ok])
        sel = list(pd.Series(m0.feature_importances_, index=cols).sort_values(ascending=False).index[:k])
        pr = np.mean([make_model('xgb', params, s).fit(X.loc[tr, sel][ok], t[ok]).predict(X.loc[te, sel])
                      * P1[te].values for s in range(seeds)], axis=0)
        out.append(pd.DataFrame(dict(pred=pr, act=p[te].values, nv=P1[te].values, yr=y), index=df.index[te]))
    return pd.concat(out)


def mase(R): return np.abs(R.pred - R.act).mean() / np.abs(R.nv - R.act).mean()


def val(groups, params=None, k=60, start=2001):
    return mase(run(X_of(groups), VAL, VSEEDS, params or BP, k, start))


def tune(groups, start=2001, n=40):
    X = X_of(groups)
    ks = [(k, val(groups, k=k, start=start)) for k in (15, 25, 40, 60, X.shape[1])]
    K = min(ks, key=lambda x: x[1])[0]
    random.seed(11)
    cands = [dict(max_depth=random.choice([2, 3, 4, 5, 6]), n_estimators=random.choice([400, 800, 1200]),
                  learning_rate=random.choice([0.02, 0.03, 0.05, 0.08]), subsample=random.choice([0.6, 0.8, 1.0]),
                  colsample_bytree=random.choice([0.5, 0.7, 0.9]), min_child_weight=random.choice([1, 3, 5, 8, 12]),
                  reg_lambda=random.choice([0.5, 1.0, 3.0, 8.0])) for _ in range(n)] + [BP]
    res = sorted((val(groups, g, K, start), i) for i, g in enumerate(cands))
    print(f'  [{"+".join(groups)} start={start}] K ' + ' '.join(f'{k}:{m:.3f}' for k, m in ks) +
          f' → {K} | best {res[0][0]:.3f} #{res[0][1]}', flush=True)
    return dict(groups=groups, K=K, params=cands[res[0][1]], val=res[0][0], start=start)


def report(title, years, bests):
    print(f'\n=== {title} ===', flush=True)
    R = {}
    for b in bests:
        nm = '+'.join(b['groups']) + f' ({b["start"]}~)'
        R[nm] = run(X_of(b['groups']), years, TSEEDS, b['params'], b['K'], b['start'])
        p = panel(R[nm].pred.values, R[nm].act.values, R[nm].nv.values)
        a = dm(R[nm].pred.values, R[nm].nv.values, R[nm].act.values, loss='abs')
        yr = R[nm].groupby('yr').apply(mase)
        print(f'{nm:<28} MASE {p["MASE"]:.3f} MAPE {p["MAPE"]:5.2f}% R² {p["R2"]:.3f} DA {p["DA"]:4.1f}% '
              f'| DM vs 나이브 t={a[0]:+.2f} p={a[1]:.3f} | 연도별 ' + ' '.join(f'{v:.2f}' for v in yr), flush=True)
    n = panel(R[nm].nv.values, R[nm].act.values, R[nm].nv.values)
    print(f'{"나이브":<28} MAPE {n["MAPE"]:5.2f}% R² {n["R2"]:.3f}')
    return R


if __name__ == '__main__':
    t0 = time.time()
    print(f'rows {len(df)} {df.DATE.iloc[0]} ~ {df.DATE.iloc[-1]}, price NaN {df.price.isna().sum()}, sup NaN {df.sup.isna().sum()}')
    print('=== 튜닝 (검증 2016~2020) ===', flush=True)
    configs = [(['price'], 2001), (['price', 'supply'], 2001)]
    bests = [tune(g, s) for g, s in configs]
    json.dump(bests, open(os.path.join(HERE, 'best_onion.json'), 'w'), ensure_ascii=False, indent=1)
    report('시험 2021~2025', TEST, bests)
    report('추가 2026', EXTRA, bests)
    print(f'총 {time.time() - t0:.0f}s')
