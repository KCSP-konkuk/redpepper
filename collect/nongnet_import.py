"""농넷 수출입(수입) 월별 고추류 수입량 — getImexportUnitList.do

품목 12161(고추). HS 코드별 계열을 따로 받는다. 응답 1건 = 조회월 포함 최근 9개월.
최근부터 9개월씩 거슬러 올라가다 빈 창이 나오면 멈춘다. 요청 간격 2초.
단위: searchUnit=02 → 톤. selectDayCur = 당월, bfYear = 전년 누계(같은 달까지).

  python nongnet_import.py 2026-08 ../data/import_pepper_monthly.csv
"""
import sys, time
import pandas as pd, requests

BASE = 'https://www.nongnet.or.kr/front/M000000273/imexport/'
UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
ITEM = '12161'
HS = {'fresh': ['0709609000'],                       # 신선 고추류 — 생홍고추와 직접 경쟁
      'dried': ['0904210000'],                       # 건고추(부수지 않은 것)
      'powder': ['0904220000'],                      # 고춧가루(파쇄·분쇄)
      'total': ['0709609000', '0711905091', '0710807000', '0904220000', '0904210000']}
GAP, TRIES = 2.0, 3


def session():
    s = requests.Session()
    s.headers.update({'User-Agent': UA, 'Referer': BASE + 'trade.do'})
    s.get(BASE + 'trade.do', timeout=20)
    return s


def fetch(s, codes, y, m):
    data = [('imexportGugun', 'I'), ('nationGubun', ''), ('searchUnit', '02'), ('searchUnitType', 'weight'),
            ('searchDate', f'{y}년 {m:02d}월'), ('searchSymbol1', ITEM)] + [('searchSymbol2Arr', c) for c in codes]
    for i in range(TRIES):
        try:
            r = s.post(BASE + 'getImexportUnitList.do', data=data, timeout=20,
                       headers={'X-Requested-With': 'XMLHttpRequest'})
            r.raise_for_status()
            return r.json().get('datalist') or []
        except Exception as e:
            print(f'  retry {i + 1}: {e}', flush=True); time.sleep(5)
    raise RuntimeError(f'failed {codes} {y}-{m}')


def main(start, out):
    y, m = map(int, start.split('-'))
    s, rows, n = session(), [], 0
    for name, codes in HS.items():
        cy, cm = y, m
        while True:
            time.sleep(GAP)
            lst = fetch(s, codes, cy, cm); n += 1
            got = [x for x in lst if x.get('selectDayCur') is not None]
            print(f'{name} {cy}-{cm:02d}: {len(got)} months ({got[-1]["saledate"] if got else "-"})', flush=True)
            if not got:
                break
            rows += [dict(series=name, ym=x['saledate'], ton=x['selectDayCur'], ton_ytd=x['selectDayTot'],
                          ton_ytd_bfyear=x['bfYear']) for x in got]
            k = cy * 12 + cm - 1 - 9                        # 9개월 앞으로
            cy, cm = k // 12, k % 12 + 1
            if cy < 1995:
                break
    df = pd.DataFrame(rows).drop_duplicates(['series', 'ym']).sort_values(['series', 'ym'])
    df.to_csv(out, index=False)
    print(f'requests {n}, rows {len(df)}', df.groupby('series').ym.agg(['min', 'max', 'count']).to_string(), sep='\n')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
