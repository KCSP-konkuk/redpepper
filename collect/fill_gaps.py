"""nongnet_daily 결과의 커버리지 공백을 메운다.

조회일이 명절 휴장과 겹치면 빈 응답이 와서 직전 며칠이 비는 경우가 있다.
공백 (prev, cur) 마다 '창이 시작한 날(없으면 cur) 하루 전'을 조회해 같은 JSONL 에 덧붙인다.
그래도 비면 다음 라운드에서 하루씩 더 당긴다. 공백 구간을 넘어서면 멈춘다(진짜 연휴).

  python fill_gaps.py ../data/raw_daily_홍고추.jsonl
"""
import json, sys, time, datetime as dt
import nongnet_daily as nd
import parse_daily as pdl

ROUNDS = 6


def gaps(path):
    recs = sorted((json.loads(l) for l in open(path, encoding='utf-8')), key=lambda r: r['date'])
    tried = {r['date'] for r in recs}
    ok = [r for r in recs if r['status'] != 'fail']
    rows = {}
    for r in ok:
        for x in pdl.rows_of(r):
            rows.setdefault(r['date'], []).append(x['date'])
    qs = sorted({dt.date.fromisoformat(r['date']) for r in ok})
    # 창 하나는 가장 이른 행 ~ 조회일의 달력 구간 전체를 덮는다(그 사이 없는 날은 휴장)
    covered = set()
    for q, v in rows.items():
        q = dt.date.fromisoformat(q)
        covered |= {min(v) + dt.timedelta(n) for n in range((q - min(v)).days + 1)}
    need = []
    for prev, cur in zip(qs, qs[1:]):
        # (prev, cur) 사이에서 어떤 창에도 없고 아직 조회도 안 한 가장 늦은 날.
        # 휴장일이면 빈 응답이 오고, 다음 라운드에서 그 전날이 뽑힌다.
        left = [prev + dt.timedelta(n) for n in range(1, (cur - prev).days)
                if prev + dt.timedelta(n) not in covered
                and (prev + dt.timedelta(n)).isoformat() not in tried]
        if left:
            need.append(left[-1])
    return need


def main(path):
    for rnd in range(1, ROUNDS + 1):
        need = gaps(path)
        print(f'round {rnd}: {len(need)} dates {[str(d) for d in need[:20]]}', flush=True)
        if not need:
            break
        s, last = nd.new_session(), 0.0
        with open(path, 'a', encoding='utf-8') as out:
            for day in need:
                rec = {'date': day.isoformat()}
                for attempt in range(1, nd.ATTEMPTS + 1):
                    time.sleep(max(0.0, nd.GAP - (time.time() - last)))
                    last = time.time()
                    try:
                        r = nd.fetch(s, day)
                        p = nd.Tables(); p.feed(r.text)
                        tables = [t for t in p.tables if t]
                        rec.update(bytes=len(r.content), tables=tables, status='ok')
                        break
                    except Exception as e:
                        rec.update(status='fail', error=f'{type(e).__name__}: {e}'[:200])
                        time.sleep(nd.RETRY_WAIT)
                out.write(json.dumps(rec, ensure_ascii=False) + '\n'); out.flush()


if __name__ == '__main__':
    main(sys.argv[1])
