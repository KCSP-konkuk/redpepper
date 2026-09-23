"""농넷 가락 일별 가격 백필 (홍고추 24210 / 10키로상자).

POST garak.do 를 날짜마다 1회 호출한다(NongnetService 와 같은 폼). 표준 라이브러리 + requests 만 쓴다.
응답에서 <table> 전부를 셀 텍스트로 뽑아 JSONL 에 한 줄씩 남긴다(파싱은 따로).
응답 1건에 조회일 포함 최근 8거래일이 온다(달력으로 8일 이상). 그래서 조회일을 STEP(7)일 간격으로
잡아도 빈틈이 없고, 겹치는 날로 앞뒤 응답의 일관성을 검증할 수 있다. 행에 연도가 없으므로 파싱 때
조회일 기준으로 붙인다. 이미 받은 날짜(status=ok/empty)는 건너뛰므로 중단 후 재실행하면 이어 받는다.

  python nongnet_daily.py 2001-01-01 2026-09-22 out.jsonl
"""
import json, sys, time, datetime as dt
from html.parser import HTMLParser
import requests

URL = 'https://www.nongnet.or.kr/front/M000000258/marketInfo/garak.do'
UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
ITEM = ('홍고추', '24210', '10키로상자', '10')
STEP = 7           # 조회일 간격(일). 8거래일 창보다 작아야 한다
GAP = 1.5          # 요청 사이 최소 간격(초)
TIMEOUT = 20
ATTEMPTS = 3
RETRY_WAIT = 5
MAX_CONSEC_FAIL = 10   # 연속 실패가 이만큼이면 멈춘다(장애·차단 시 계속 두드리지 않게)


class Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables, self._row, self._cell, self._depth = [], None, None, 0

    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            self._depth += 1
            if self._depth == 1:
                self.tables.append([])
        elif self._depth and tag == 'tr':
            self._row = []
        elif self._depth and tag in ('td', 'th') and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag == 'table' and self._depth:
            self._depth -= 1
        elif tag in ('td', 'th') and self._cell is not None:
            self._row.append(' '.join(''.join(self._cell).split()))
            self._cell = None
        elif tag == 'tr' and self._row is not None:
            if self._row and self._depth:
                self.tables[-1].append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def new_session():
    s = requests.Session()
    s.headers.update({'User-Agent': UA, 'Referer': URL,
                      'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                      'Accept-Language': 'ko-KR,ko;q=0.9'})
    s.get(URL, timeout=TIMEOUT)
    return s


def fetch(s, day):
    d = day.strftime('%Y년 %m월 %d일')
    name, cd, trd_name, trd = ITEM
    form = {'searchSymbol1': 'garak', 'searchName1': '가락', 'menuType': 'garak',
            'searchDate': d, 'bestDate': d,
            'searchName2': name, 'searchSymbol2': cd, 'searchScd': cd,
            'searchName3': trd_name, 'searchSymbol3': trd, 'searchTrd': trd}
    r = s.post(URL, data=form, timeout=TIMEOUT)
    r.raise_for_status()
    return r


def done_dates(path):
    out = set()
    try:
        with open(path, encoding='utf-8') as f:
            for line in f:
                rec = json.loads(line)
                if rec['status'] in ('ok', 'empty'):
                    out.add(rec['date'])
    except FileNotFoundError:
        pass
    return out


def main(start, end, path):
    start, end = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    have = done_dates(path)
    todo = [end - dt.timedelta(n) for n in range(0, (end - start).days + 1, STEP)][::-1]
    if todo[0] > start:
        todo.insert(0, start)
    todo = [d for d in todo if d.isoformat() not in have]
    print(f'{len(todo)} dates to fetch ({len(have)} already done)', flush=True)

    s, consec_fail, last = new_session(), 0, 0.0
    with open(path, 'a', encoding='utf-8') as out:
        for i, day in enumerate(todo, 1):
            rec = {'date': day.isoformat()}
            for attempt in range(1, ATTEMPTS + 1):
                time.sleep(max(0.0, GAP - (time.time() - last)))
                last = time.time()
                try:
                    r = fetch(s, day)
                    p = Tables(); p.feed(r.text)
                    tables = [t for t in p.tables if t]
                    rec.update(bytes=len(r.content), tables=tables,
                               status='ok' if tables else 'empty')
                    break
                except Exception as e:
                    rec.update(status='fail', error=f'{type(e).__name__}: {e}'[:200])
                    if attempt < ATTEMPTS:
                        time.sleep(RETRY_WAIT)
                        try:
                            s = new_session()
                        except Exception:
                            pass
            out.write(json.dumps(rec, ensure_ascii=False) + '\n'); out.flush()
            consec_fail = consec_fail + 1 if rec['status'] == 'fail' else 0
            if i % 100 == 0 or rec['status'] == 'fail':
                print(f'[{i}/{len(todo)}] {rec["date"]} {rec["status"]} {rec.get("error", "")}', flush=True)
            if consec_fail >= MAX_CONSEC_FAIL:
                print(f'stop: {consec_fail} consecutive failures', flush=True)
                sys.exit(2)
    print('finished', flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:4])
