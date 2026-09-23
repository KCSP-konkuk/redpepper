"""양파(농넷 24400 / 1키로) 순별 가격·반입량 2001~ — 이어받기·중간 저장

  python collect_soon.py            # data/onion/ 에 soon_onion.csv · area_onion.csv

- 9순씩 거슬러 올라간다(1회 = 9순). 가격·반입량을 따로 받는다
- 받을 때마다 CSV 를 덮어써 중간에 멈춰도 남고, 다시 실행하면 이미 있는 순은 건너뛴다
- 반입량(area) 요청은 응답이 느리고 시간 초과가 잦다(2026-09-23 수십 번) → 간격을 길게 두고,
  연속 실패가 MAX_FAIL 이면 멈춘다(농넷 과다 요청 차단 이력)
"""
import os, sys, time
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'collect'))
from nongnet import session, fetch   # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'onion')
PS = {1: '상순', 2: '중순', 3: '하순'}
CODE, SPEC = '24400', '1'
GAP = {'price': 1.5, 'area': 4.0}
MAX_FAIL = 5


def load(fn):
    p = os.path.join(OUT, fn)
    return pd.read_csv(p).astype({'DATE': str}) if os.path.exists(p) else pd.DataFrame(columns=['DATE'])


def main():
    os.makedirs(OUT, exist_ok=True)
    s = session()
    for dtype, fn in (('price', 'soon_onion.csv'), ('area', 'area_onion.csv')):
        have = load(fn); rows = {r.DATE: r._asdict() for r in have.itertuples(index=False)}
        y, m, p, fail = 2026, 9, 3, 0
        while (y, m) >= (2001, 1):
            want = []
            yy, mm, pp = y, m, p
            for _ in range(9):
                want.append(f'{yy}{mm:02d}{PS[pp]}')
                pp -= 1
                if pp == 0: pp = 3; mm -= 1
                if mm == 0: mm = 12; yy -= 1
            if all(w in rows for w in want):
                y, m, p = yy, mm, pp; continue
            ds = f'{y}년 {m:02d}월 {[5, 15, 25][p - 1]:02d}일'
            time.sleep(GAP[dtype])
            try:
                d = fetch(s, CODE, SPEC, ds, dtype).json().get('datalist', []); fail = 0
            except Exception as e:
                fail += 1; print('fail', dtype, ds, e, flush=True)
                if fail >= MAX_FAIL:
                    print(f'stop: {dtype} {fail} consecutive failures — 나중에 다시 실행하면 이어 받는다'); break
                time.sleep(10); continue
            for it in d:
                k = f"{it['year']}{it['month']:02d}{PS[it['soonVal']]}"
                rows[k] = (dict(DATE=k, val=it.get('selectSoon'), yearAvg=it.get('yearAvg'), bfYear=it.get('bfYear'))
                           if dtype == 'price' else
                           dict(DATE=k, sup=it.get('selectSoon'), sup_py=it.get('bfYear'), sup_ny=it.get('yearAvg')))
            pd.DataFrame(sorted(rows.values(), key=lambda r: r['DATE'])).to_csv(os.path.join(OUT, fn), index=False)
            print(dtype, ds, len(rows), flush=True)
            y, m, p = yy, mm, pp
    print('done')


if __name__ == '__main__':
    main()
