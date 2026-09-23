"""nongnet_daily.py 의 JSONL → 일별 CSV + 검증.

  python parse_daily.py ../data/raw_daily_홍고추.jsonl ../data/daily_홍고추.csv

검증 3가지를 출력한다.
  1) 겹치는 날짜 값 불일치(앞뒤 응답이 같은 날을 다르게 주는지)
  2) 커버리지 공백: 조회일 사이에 창이 닿지 않은 구간 → 재요청할 조회일 목록
  3) 순별 CSV(cross_long_홍고추.csv, 상 등급)와 일별 상 순평균 대조
"""
import json, sys, datetime as dt
import pandas as pd

GRADES = ['특', '상', '보통', '하']


def num(x):
    x = x.replace(',', '').strip()
    try:
        v = float(x)
    except ValueError:
        return None
    return v if v > 0 else None   # 0 은 무거래 표기


def rows_of(rec):
    q = dt.date.fromisoformat(rec['date'])
    t = (rec.get('tables') or [[]])[0]
    if not t or t[0][0] != '날짜':
        return []
    head, out = t[0], []
    for r in t[1:]:
        if '/' not in r[0]:
            continue
        m, d = map(int, r[0].split('/'))
        y = q.year if (m, d) <= (q.month, q.day) else q.year - 1
        row = {'date': dt.date(y, m, d), 'query': q}
        for g, v in zip(head[1:], r[1:]):
            if g in GRADES:
                row[g] = num(v)
        out.append(row)
    return out


def soon_of(d):
    return f'{d.year}{d.month:02d}' + ('상순' if d.day <= 10 else '중순' if d.day <= 20 else '하순')


def main(src, dst):
    recs = [json.loads(l) for l in open(src, encoding='utf-8')]
    fails = [r['date'] for r in recs if r['status'] == 'fail']
    recs = sorted((r for r in recs if r['status'] != 'fail'), key=lambda r: r['date'])
    raw = pd.DataFrame([x for r in recs for x in rows_of(r)])
    print(f'records {len(recs)} (fail {len(fails)}: {fails[:10]}) / rows {len(raw)}')

    # 1) 겹침 불일치
    g = raw.groupby('date')
    bad = [(d, sub[GRADES].drop_duplicates().shape[0]) for d, sub in g
           if sub[[c for c in GRADES if c in sub]].drop_duplicates().shape[0] > 1]
    print(f'[1] overlapping dates {int((g.size() > 1).sum())}, inconsistent {len(bad)}: {bad[:10]}')

    # 2) 커버리지: 각 조회일 창의 가장 이른 날짜가 직전 조회일+1 이하인지
    qs = [dt.date.fromisoformat(r['date']) for r in recs]
    earliest = raw.groupby('query').date.min()
    gaps = []
    for prev, cur in zip(qs, qs[1:]):
        lo = earliest.get(cur)
        if lo is None or lo > prev + dt.timedelta(1):
            gaps.append((prev, cur, lo))
    print(f'[2] coverage gaps {len(gaps)}')
    for p, c, lo in gaps[:40]:
        print(f'    {p} → {c}  window starts {lo}')

    daily = g.first().drop(columns='query').reset_index()
    daily = daily[['date'] + [c for c in GRADES if c in daily]]
    daily.to_csv(dst, index=False)
    print(f'saved {dst}: {len(daily)} days {daily.date.min()} ~ {daily.date.max()}')

    # 3) 순별 CSV 대조
    daily['DATE'] = daily.date.map(soon_of)
    sm = daily.dropna(subset=['상']).groupby('DATE')['상'].mean()
    ref = pd.read_csv(sys.argv[3] if len(sys.argv) > 3 else '../data/cross_long_홍고추.csv',
                      index_col='DATE')['val']
    j = pd.concat([sm.rename('daily'), ref.rename('soon')], axis=1).dropna()
    rel = (j.daily / j.soon - 1).abs()
    print(f'[3] soons {len(j)}  |daily_mean/soon-1|: median {rel.median():.4f}  '
          f'p90 {rel.quantile(.9):.4f}  >5% {int((rel > .05).sum())}')
    print(j[rel > .05].assign(rel=rel[rel > .05]).head(15))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
