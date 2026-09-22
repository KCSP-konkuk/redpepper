"""농넷 순별 가격/반입량 수집 (공개, 인증 불필요). 요청은 품목당 ~33회 + 0.8초 간격."""
import time, requests, pandas as pd
BASE='https://www.nongnet.or.kr'
SOON=BASE+'/front/M000000258/marketInfo/getGarakSoonList.do'
def session():
    s=requests.Session()
    s.headers.update({'User-Agent':'Mozilla/5.0','X-Requested-With':'XMLHttpRequest',
                      'Referer':BASE+'/front/M000000258/marketInfo/garak.do'})
    s.get(BASE+'/front/M000000258/marketInfo/garak.do',timeout=30)
    return s
def fetch(s,item,spec,datestr,dtype='price'):
    p={'soonDataType':dtype,'searchSoonGrade':'1','searchSoonGradeNm':'상','searchUnitCd':'1',
       'searchDate':datestr,'searchSymbol1':'garak','searchSymbol2':item,'searchSymbol3':spec}
    r=s.get(SOON,params=p,timeout=30)
    return r
if __name__=='__main__':
    s=session(); r=fetch(s,'24201','10','2024년 06월 15일')
    print(r.status_code, len(r.content)); print(r.text[:900])
