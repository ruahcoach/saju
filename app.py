# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, date, time, timedelta, timezone
import re, math, calendar as cal_mod, os
from urllib.parse import urlencode
from urllib.request import urlopen
import xml.etree.ElementTree as ET
import streamlit as st
from zoneinfo import ZoneInfo
try:
    from korean_lunar_calendar import KoreanLunarCalendar
    HAS_LUNAR = True
except Exception:
    HAS_LUNAR = False

from korea_tz_history import wall_to_true_solar_time   # ← 이 줄 추가

def get_kasi_key():
    try:
        val = st.secrets.get('KASI_KEY')
        if val: return val
    except Exception: pass
    return os.getenv('KASI_KEY')

LOCAL_TZ = ZoneInfo('Asia/Seoul')
DEFAULT_LONGITUDE = 126.9780  # 서울 기본값

city_options = {
    "서울": 126.9780,
    "부산": 129.0756,
    "대구": 128.6014,
    "인천": 126.7052,
    "광주": 126.8526,
    "대전": 127.3845,
    "울산": 129.3114,
    "제주": 126.5312,
}
def to_solar_time(dt_local, longitude=DEFAULT_LONGITUDE):
    """역사적 표준시 + 썸머타임 + 균시차 완전 반영"""
    result = wall_to_true_solar_time(dt_local, longitude, apply_eot=True)
    if result.tzinfo is None:
        result = result.replace(tzinfo=LOCAL_TZ)
    return result
    
CHEONGAN = ['갑','을','병','정','무','기','경','신','임','계']
JIJI = ['자','축','인','묘','진','사','오','미','신','유','술','해']
HANJA_GAN = ['甲','乙','丙','丁','戊','己','庚','辛','壬','癸']
HANJA_JI = ['子','丑','寅','卯','辰','巳','午','未','申','酉','戌','亥']
MONTH_JI = ['인','묘','진','사','오','미','신','유','술','해','자','축']
JIE_TO_MONTH_JI = {'입춘':'인','경칩':'묘','청명':'진','입하':'사','망종':'오','소서':'미','입추':'신','백로':'유','한로':'술','입동':'해','대설':'자','소한':'축','(전년)대설':'자'}
MONTH_TO_2TERMS = {'인':('입춘','우수'),'묘':('경칩','춘분'),'진':('청명','곡우'),'사':('입하','소만'),'오':('망종','하지'),'미':('소서','대서'),'신':('입추','처서'),'유':('백로','추분'),'술':('한로','상강'),'해':('입동','소설'),'자':('대설','동지'),'축':('소한','대한')}
GAN_BG = {'갑':'#2ecc71','을':'#2ecc71','병':'#e74c3c','정':'#e74c3c','무':'#f1c40f','기':'#f1c40f','경':'#ffffff','신':'#ffffff','임':'#000000','계':'#000000'}
BR_BG = {'해':'#000000','자':'#000000','인':'#2ecc71','묘':'#2ecc71','사':'#e74c3c','오':'#e74c3c','신':'#ffffff','유':'#ffffff','진':'#f1c40f','술':'#f1c40f','축':'#f1c40f','미':'#f1c40f'}
def gan_fg(gan): bg=GAN_BG.get(gan,'#fff'); return '#000000' if bg in ('#ffffff','#f1c40f') else '#ffffff'
def br_fg(ji): bg=BR_BG.get(ji,'#fff'); return '#000000' if bg in ('#ffffff','#f1c40f') else '#ffffff'
STEM_ELEM = {'갑':'목','을':'목','병':'화','정':'화','무':'토','기':'토','경':'금','신':'금','임':'수','계':'수'}
STEM_YY = {'갑':'양','을':'음','병':'양','정':'음','무':'양','기':'음','경':'양','신':'음','임':'양','계':'음'}
BRANCH_MAIN = {'자':'계','축':'기','인':'갑','묘':'을','진':'무','사':'병','오':'정','미':'기','신':'경','유':'신','술':'무','해':'임'}
ELEM_PRODUCE = {'목':'화','화':'토','토':'금','금':'수','수':'목'}
ELEM_CONTROL = {'목':'토','화':'금','토':'수','금':'목','수':'화'}
ELEM_OVER_ME = {v:k for k,v in ELEM_CONTROL.items()}
ELEM_PROD_ME = {v:k for k,v in ELEM_PRODUCE.items()}
SAMHAP = {'화':{'인','오','술'},'목':{'해','묘','미'},'수':{'신','자','진'},'금':{'사','유','축'}}
MONTH_SAMHAP = {'인':'화','오':'화','술':'화','해':'목','묘':'목','미':'목','신':'수','자':'수','진':'수','사':'금','유':'금','축':'금'}
BRANCH_HIDDEN = {'자':['임','계'],'축':['계','신','기'],'인':['무','병','갑'],'묘':['갑','을'],'진':['을','계','무'],'사':['무','경','병'],'오':['병','기','정'],'미':['정','을','기'],'신':['무','임','경'],'유':['경','신'],'술':['신','정','무'],'해':['무','갑','임']}
NOTEARTH = {'갑','을','병','정','경','신','임','계'}
def stems_of_element(elem): return {'목':['갑','을'],'화':['병','정'],'토':['무','기'],'금':['경','신'],'수':['임','계']}[elem]
def stem_with_polarity(elem, parity): a,b=stems_of_element(elem); return a if parity=='양' else b
def is_yang_stem(gan): return gan in ['갑','병','무','경','임']
def ten_god_for_stem(day_stem, other_stem):
    d_e,d_p = STEM_ELEM[day_stem],STEM_YY[day_stem]
    o_e,o_p = STEM_ELEM[other_stem],STEM_YY[other_stem]
    if o_e==d_e: return '비견' if o_p==d_p else '겁재'
    if o_e==ELEM_PRODUCE[d_e]: return '식신' if o_p==d_p else '상관'
    if o_e==ELEM_CONTROL[d_e]: return '편재' if o_p==d_p else '정재'
    if o_e==ELEM_OVER_ME[d_e]: return '편관' if o_p==d_p else '정관'
    if o_e==ELEM_PROD_ME[d_e]: return '편인' if o_p==d_p else '정인'
    return '미정'
def ten_god_for_branch(day_stem, branch): return ten_god_for_stem(day_stem, BRANCH_MAIN[branch])
def six_for_stem(ds,s): return ten_god_for_stem(ds,s)
def six_for_branch(ds,b): return ten_god_for_branch(ds,b)
def all_hidden_stems(branches):
    s=set()
    for b in branches: s.update(BRANCH_HIDDEN.get(b,[]))
    return s
def is_first_half_by_terms(dt_solar, first_term_dt, mid_term_dt): return first_term_dt <= dt_solar < mid_term_dt

JIE_DEGREES = {'입춘':315,'경칩':345,'청명':15,'입하':45,'망종':75,'소서':105,'입추':135,'백로':165,'한로':195,'입동':225,'대설':255,'소한':285}
JIE_ORDER = ['입춘','경칩','청명','입하','망종','소서','입추','백로','한로','입동','대설','소한']
JIE24_DEGREES = {'입춘':315,'우수':330,'경칩':345,'춘분':0,'청명':15,'곡우':30,'입하':45,'소만':60,'망종':75,'하지':90,'소서':105,'대서':120,'입추':135,'처서':150,'백로':165,'추분':180,'한로':195,'상강':210,'입동':225,'소설':240,'대설':255,'동지':270,'소한':285,'대한':300}
JIE24_ORDER = ['입춘','우수','경칩','춘분','청명','곡우','입하','소만','망종','하지','소서','대서','입추','처서','백로','추분','한로','상강','입동','소설','대설','동지','소한','대한']
SIDU_START = {('갑','기'):'갑',('을','경'):'병',('병','신'):'무',('정','임'):'경',('무','계'):'임'}
def month_start_gan_idx(year_gan_idx): return ((year_gan_idx % 5) * 2 + 2) % 10
K_ANCHOR = 49

def jdn_0h_utc(y,m,d):
    if m<=2: y-=1; m+=12
    A=y//100; B=2-A+A//4
    return int(365.25*(y+4716))+int(30.6001*(m+1))+d+B-1524

def jd_from_utc(dt_utc):
    y=dt_utc.year; m=dt_utc.month
    d=dt_utc.day+(dt_utc.hour+dt_utc.minute/60+dt_utc.second/3600)/24
    if m<=2: y-=1; m+=12
    A=y//100; B=2-A+A//4
    return int(365.25*(y+4716))+int(30.6001*(m+1))+d+B-1524.5

def norm360(x): return x%360.0
def wrap180(x): return (x+180.0)%360.0-180.0

def delta_t_seconds(year):
    y = year
    if 2005 <= y <= 2050:
        t = y - 2000
        return 62.92 + 0.32217*t + 0.005589*t*t
    elif 1986 <= y < 2005:
        t = y - 2000
        return 63.86 + 0.3345*t - 0.060374*t*t \
               + 0.0017275*t**3 + 0.000651814*t**4 \
               + 0.00002373599*t**5
    else:
        t = (y - 2000)/100
        return 62.92 + 32.217*t + 55.89*t*t

def equation_of_time_minutes(dt_utc):
    doy = dt_utc.timetuple().tm_yday
    B = math.radians((360/365) * (doy - 81))
    return 9.87*math.sin(2*B) - 7.53*math.cos(B) - 1.5*math.sin(B)

# ── ephem (VSOP87) 우선, 폴백: 간이 Meeus ──
try:
    import ephem as _ephem
    _HAS_EPHEM = True
except ImportError:
    _HAS_EPHEM = False

def solar_longitude_deg(dt_utc):
    if _HAS_EPHEM:
        d = _ephem.Date(dt_utc)
        s = _ephem.Sun(d)
        eq = _ephem.Equatorial(s.ra, s.dec, epoch=d)
        ec = _ephem.Ecliptic(eq)
        return math.degrees(float(ec.lon)) % 360

    # 폴백: 간이 Meeus 공식
    dt_tt = dt_utc + timedelta(seconds=delta_t_seconds(dt_utc.year))
    JD = jd_from_utc(dt_tt)
    T = (JD - 2451545.0) / 36525.0
    L0 = norm360(280.46646 + 36000.76983*T + 0.0003032*T*T)
    M  = norm360(357.52911 + 35999.05029*T - 0.0001537*T*T)
    Mr = math.radians(M)
    C = ((1.914602 - 0.004817*T - 0.000014*T*T) * math.sin(Mr)
         + (0.019993 - 0.000101*T) * math.sin(2*Mr)
         + 0.000289 * math.sin(3*Mr))
    theta = L0 + C
    Omega = 125.04 - 1934.136*T
    lam = theta - 0.00569 - 0.00478 * math.sin(math.radians(Omega))
    return norm360(lam)

def find_longitude_time_local(year, target_deg, approx_dt_local):
    a=(approx_dt_local-timedelta(days=7)).astimezone(timezone.utc)
    b=(approx_dt_local+timedelta(days=7)).astimezone(timezone.utc)
    def f(dt_utc): return wrap180(solar_longitude_deg(dt_utc)-target_deg)
    scan,step=a,timedelta(hours=6); fa=f(scan); found=False
    while scan<b:
        scan2=scan+step; fb=f(scan2)
        if fa==0 or fb==0 or (fa<0 and fb>0) or (fa>0 and fb<0): a,b=scan,scan2; found=True; break
        scan,fa=scan2,fb
    if not found:
        a=(approx_dt_local-timedelta(days=1)).astimezone(timezone.utc)
        b=(approx_dt_local+timedelta(days=1)).astimezone(timezone.utc)
    for _ in range(100):
        mid=a+(b-a)/2; fm=f(mid); fa=f(a)
        if fm==0: a=b=mid; break
        if (fa<=0 and fm>=0) or (fa>=0 and fm<=0): b=mid
        else: a=mid
    mid_local = (a+(b-a)/2).astimezone(LOCAL_TZ)
    return mid_local.replace(microsecond=0)
def approx_guess_local(year):
    rough={'입춘':(2,4),'경칩':(3,6),'청명':(4,5),'입하':(5,6),'망종':(6,6),'소서':(7,7),'입추':(8,8),'백로':(9,8),'한로':(10,8),'입동':(11,7),'대설':(12,7),'소한':(1,6)}
    out={}
    for name,(m,d) in rough.items(): out[name]=datetime(year,m,d,9,0,tzinfo=LOCAL_TZ)
    out['(전년)대설']=datetime(year-1,12,7,9,0,tzinfo=LOCAL_TZ)
    return out

def approx_guess_local_24(year):
    rough={'입춘':(2,4),'우수':(2,19),'경칩':(3,6),'춘분':(3,21),'청명':(4,5),'곡우':(4,20),'입하':(5,6),'소만':(5,21),'망종':(6,6),'하지':(6,21),'소서':(7,7),'대서':(7,23),'입추':(8,8),'처서':(8,23),'백로':(9,8),'추분':(9,23),'한로':(10,8),'상강':(10,23),'입동':(11,7),'소설':(11,22),'대설':(12,7),'동지':(12,22),'소한':(1,6),'대한':(1,20)}
    out={}
    for name,(m,d) in rough.items(): out[name]=datetime(year,m,d,9,0,tzinfo=LOCAL_TZ)
    return out

def compute_jie_times_calc(year):
    guesses=approx_guess_local(year); terms={}
    for name in JIE_ORDER: terms[name]=find_longitude_time_local(year,JIE_DEGREES[name],guesses[name])
    terms['(전년)대설']=find_longitude_time_local(year-1,JIE_DEGREES['대설'],guesses['(전년)대설'])
    return terms

def compute_jie24_times_calc(year):
    guesses=approx_guess_local_24(year); out={}
    for name in JIE24_ORDER:
        deg=JIE24_DEGREES[name]; approx=guesses[name]; calc_year=approx.year
        out[name]=find_longitude_time_local(calc_year,deg,approx)
    return out

def pillar_day_by_2300(dt_solar):
    return (dt_solar+timedelta(days=1)).date() if (dt_solar.hour,dt_solar.minute)>=(23,0) else dt_solar.date()

def day_ganji_solar(dt_solar, k_anchor=K_ANCHOR):
    d=pillar_day_by_2300(dt_solar); idx60=(jdn_0h_utc(d.year,d.month,d.day)+k_anchor)%60
    cidx,jidx=idx60%10,idx60%12; return CHEONGAN[cidx]+JIJI[jidx],cidx,jidx

def hour_branch_idx_2300(dt_solar):
    mins = dt_solar.hour * 60 + dt_solar.minute
    off = (mins - (23 * 60)) % 1440
    return off // 120
def sidu_zi_start_gan(day_gan):
    for pair,start in SIDU_START.items():
        if day_gan in pair: return start
    raise ValueError('invalid day gan')

def four_pillars_from_solar(dt_solar, k_anchor=K_ANCHOR):
    jie12 = compute_jie_times_calc(dt_solar.year)

    # 🔥 절기에도 동일 경도 보정 적용
    if st.session_state.get('apply_solar', True):
        lon = st.session_state.get('longitude', DEFAULT_LONGITUDE)
        for k in jie12:
            jie12[k] = to_solar_time(jie12[k], lon)

    jie_solar = jie12
    ipchun=jie_solar.get("입춘")
    y=dt_solar.year-1 if dt_solar<ipchun else dt_solar.year
    y_gidx=(y-4)%10; y_jidx=(y-4)%12
    year_pillar=CHEONGAN[y_gidx]+JIJI[y_jidx]
    order=list(jie_solar.items()); order.sort(key=lambda x:x[1])
    last='(전년)대설'
    for name,t in order:
        if dt_solar>=t: last=name
        else: break
    m_branch=JIE_TO_MONTH_JI[last]; m_bidx=MONTH_JI.index(m_branch)
    m_gidx=(month_start_gan_idx(y_gidx)+m_bidx)%10
    month_pillar=CHEONGAN[m_gidx]+m_branch
    day_pillar,d_cidx,d_jidx=day_ganji_solar(dt_solar,k_anchor)
    h_j_idx=hour_branch_idx_2300(dt_solar)
    zi_start=sidu_zi_start_gan(CHEONGAN[d_cidx])
    h_c_idx=(CHEONGAN.index(zi_start)+h_j_idx)%10
    hour_pillar=CHEONGAN[h_c_idx]+JIJI[h_j_idx]
    return {'year':year_pillar,'month':month_pillar,'day':day_pillar,'hour':hour_pillar,'y_gidx':y_gidx,'m_gidx':m_gidx,'m_bidx':m_bidx,'d_cidx':d_cidx}

def next_prev_jie(dt_solar, jie_solar_dict):
    items=[(n,t) for n,t in jie_solar_dict.items()]; items.sort(key=lambda x:x[1])
    prev_t=items[0][1]
    for _,t in items:
        if t>dt_solar: return prev_t,t
        prev_t=t
    return prev_t,prev_t

def round_half_up(x): return int(math.floor(x+0.5))

def dayun_start_age(dt_solar, jie12_solar, forward):
    prev_t,next_t=next_prev_jie(dt_solar,jie12_solar)
    delta_days=(next_t-dt_solar).total_seconds()/86400.0 if forward else (dt_solar-prev_t).total_seconds()/86400.0
    return max(0,round_half_up(delta_days/3.0))

def build_dayun_list(month_gidx, month_bidx, forward, start_age, count=11):
    dirv=1 if forward else -1; out=[]
    for i in range(1,count+1):
        g_i=(month_gidx+dirv*i)%10; b_i=(month_bidx+dirv*i)%12
        out.append({'start_age':start_age+(i-1)*10,'g_idx':g_i,'b_idx':b_i})
    return out

def calc_age_on(dob, now_dt):
    today=now_dt.date() if hasattr(now_dt,"date") else now_dt
    return today.year-dob.year-((today.month,today.day)<(dob.month,dob.day))

def lunar_to_solar(y,m,d,is_leap=False):
    if not HAS_LUNAR: raise RuntimeError('korean-lunar-calendar 미설치')
    c=KoreanLunarCalendar(); c.setLunarDate(y,m,d,is_leap); return date(c.solarYear,c.solarMonth,c.solarDay)

@dataclass
class Inputs:
    day_stem: str
    month_branch: str
    month_stem: str
    stems_visible: list
    branches_visible: list
    solar_dt: datetime
    first_term_dt: datetime
    mid_term_dt: datetime
    day_from_jieqi: int

def decide_geok(inp):
    ds=inp.day_stem; mb=inp.month_branch; ms=inp.month_stem
    stems=list(inp.stems_visible); branches=list(inp.branches_visible)
    ds_e=STEM_ELEM[ds]; ds_p=STEM_YY[ds]
    mb_main=BRANCH_MAIN[mb]; mb_e,mb_p=STEM_ELEM[mb_main],STEM_YY[mb_main]
    visible_set=set(stems); hidden_set=all_hidden_stems(branches); pool=visible_set|hidden_set
    if mb in {'자','오','묘','유','인','신','사','해'} and ds_e==mb_e:
        off_e=ELEM_OVER_ME[ds_e]
        jung_gwan=stem_with_polarity(off_e,'음' if ds_p=='양' else '양')
        pyeon_gwan=stem_with_polarity(off_e,ds_p)
        same_polarity=(ds_p==mb_p)
        any_jung_br=any(ten_god_for_branch(ds,b)=='정관' for b in branches)
        any_pyeon_br=any(ten_god_for_branch(ds,b)=='편관' for b in branches)
        if same_polarity:
            if (jung_gwan in visible_set) or any_jung_br:
                why=('정관 '+jung_gwan+' 천간 투간' if jung_gwan in visible_set else '지지 정관 존재')
                return '건록격',f'[특수] 월비+{why}->건록격'
            else: return '월비격','[특수] 월비, 정관 없음->월비격'
        else:
            if (pyeon_gwan in visible_set) or any_pyeon_br:
                why=('편관 '+pyeon_gwan+' 천간 투간' if pyeon_gwan in visible_set else '지지 편관 존재')
                return '양인격',f'[특수] 월겁+{why}->양인격'
            else: return '월겁격','[특수] 월겁, 편관 없음->월겁격'
    grp='자오묘유' if mb in {'자','오','묘','유'} else ('인신사해' if mb in {'인','신','사','해'} else '진술축미')
    if grp=='자오묘유':
        month_elem=STEM_ELEM[mb_main]
        same_elem_vis=[s for s in stems if STEM_ELEM.get(s)==month_elem]
        if same_elem_vis:
            pick=next((s for s in same_elem_vis if STEM_YY[s]!=ds_p),same_elem_vis[0])
            six=ten_god_for_stem(ds,pick); return f'{six}격',f'[자오묘유] {pick} 투간->{six}격'
        six=ten_god_for_stem(ds,mb_main); return f'{six}격',f'[자오묘유] 투간없음->체(본기 {mb_main}){six}격'
    if grp=='인신사해':
        rokji=mb_main; month_elem=STEM_ELEM[rokji]
        base_stems=set(stems_of_element(month_elem))
        base_vis=[s for s in inp.stems_visible if s in base_stems]
        if base_vis:
            pick=base_vis[0]
            if month_elem==STEM_ELEM[ds]:
                off_e=ELEM_OVER_ME[STEM_ELEM[ds]]
                jung_gwan=stem_with_polarity(off_e,'음' if STEM_YY[ds]=='양' else '양')
                pyeon_gwan=stem_with_polarity(off_e,STEM_YY[ds])
                if STEM_YY[pick]==STEM_YY[ds]:
                    if jung_gwan in inp.stems_visible: return '건록격',f'[인신사해] {pick}투간+정관{jung_gwan}->건록격'
                else:
                    if pyeon_gwan in inp.stems_visible: return '양인격',f'[인신사해] {pick}투간+편관{pyeon_gwan}->양인격'
            six=ten_god_for_stem(ds,pick); return f'{six}격',f'[인신사해] 록지{pick}투간->{six}격'
        tri_elem=MONTH_SAMHAP.get(mb,'')
        if tri_elem:
            tri_grp=SAMHAP[tri_elem]; others=set(tri_grp)-{mb}
            if others.issubset(set(inp.branches_visible)) and is_first_half_by_terms(inp.solar_dt,inp.first_term_dt,inp.mid_term_dt):
                tri_stems=stems_of_element(tri_elem)
                tri_vis=[s for s in tri_stems if s in inp.stems_visible]
                if tri_vis and tri_elem!=STEM_ELEM[ds]:
                    pick=tri_vis[0]; six=ten_god_for_stem(ds,pick)
                    return f'중기격({six})',f'[인신사해] 삼합+중기사령+{pick}투간->중기격'
        if ms: six=ten_god_for_stem(ds,ms); return f'{six}격',f'[인신사해] 록지투간없음->월간{ms}기준{six}격'
        six=ten_god_for_stem(ds,rokji); return f'{six}격',f'[인신사해] 폴백->본기({rokji}){six}격'
    if grp=='진술축미':
        h=BRANCH_HIDDEN.get(mb,[]); mb_main_l=BRANCH_MAIN[mb]; is_front12=(inp.day_from_jieqi<=11)
        tri_elem=MONTH_SAMHAP.get(mb,'')
        if tri_elem:
            tri_grp=SAMHAP[tri_elem]; others=set(tri_grp)-{mb}; partners=others&set(branches)
            if partners:
                if tri_elem==STEM_ELEM[ds]:
                    six=ten_god_for_stem(ds,mb_main_l); return f'{six}격',f'[진술축미] 반합{mb}+동일오행->체(본기){six}격'
                tri_stems=stems_of_element(tri_elem); tri_vis=[s for s in tri_stems if s in visible_set]
                mid_qi=h[1] if len(h)>=2 else (h[-1] if h else mb_main_l); mid_is_tri=(STEM_ELEM.get(mid_qi)==tri_elem)
                pick=tri_vis[0] if tri_vis else (mid_qi if mid_is_tri else stem_with_polarity(tri_elem,'음' if STEM_YY[ds]=='양' else '양'))
                six=ten_god_for_stem(ds,pick); return f'{six}격',f'[진술축미] 반합+{pick}기준{six}격'
        if is_front12:
            yeogi=h[0] if h else mb_main_l; y_elem=STEM_ELEM[yeogi]
            same_vis=[s for s in stems if STEM_ELEM.get(s)==y_elem]
            opp=[s for s in same_vis if STEM_YY[s]!=ds_p]
            pick=opp[0] if opp else (same_vis[0] if same_vis else yeogi)
            six=ten_god_for_stem(ds,pick); return f'{six}격',f'[진술축미] 절입후12일이내->여기사령({pick}){six}격'
        else:
            earth_vis=[s for s in ('무','기') if s in visible_set]
            opp=[s for s in earth_vis if STEM_YY[s]!=ds_p]
            pick=opp[0] if opp else (earth_vis[0] if earth_vis else mb_main_l)
            six=ten_god_for_stem(ds,pick); return f'{six}격',f'[진술축미] 절입13일이후->주왕토({pick}){six}격'
    six=ten_god_for_stem(ds,BRANCH_MAIN[mb]); return f'{six}격',f'[폴백]->체(본기{BRANCH_MAIN[mb]}){six}격'

def calc_wolun_accurate(year):
    jie12_prev=compute_jie_times_calc(year-1); jie12_this=compute_jie_times_calc(year); jie12_next=compute_jie_times_calc(year+1)
    jie24_prev=compute_jie24_times_calc(year-1); jie24_this=compute_jie24_times_calc(year); jie24_next=compute_jie24_times_calc(year+1)
    collected=[]
    for src_jie in [jie12_prev,jie12_this,jie12_next]:
        for jname in JIE_ORDER:
            if jname in src_jie:
                t = src_jie[jname]
                if t.year==year: collected.append((t,jname))
    collected.sort(key=lambda x:x[0])
    items=[]
    for t,jname in collected:
        t_calc = t + timedelta(seconds=1); fp=four_pillars_from_solar(t_calc)
        m_gan=fp['month'][0]; m_ji=fp['month'][1]
        t2_name=MONTH_TO_2TERMS[m_ji][1]; t2=None
        for src in [jie24_this,jie24_prev,jie24_next]:
            if t2_name in src:
                cand = src[t2_name]
                if cand>t: t2=cand; break
        jie_idx=JIE_ORDER.index(jname); next_jname=JIE_ORDER[(jie_idx+1)%12]; t_end=None
        for src in [jie12_this,jie12_next,jie12_prev]:
            if next_jname in src:
                nt = src[next_jname]
                if nt>t: t_end=nt; break
        items.append({'month':t.month,'gan':m_gan,'ji':m_ji,'t1':t,'t2':t2,'t_end':t_end})
    return items

def calc_ilun_strip(start_dt, end_dt, day_stem, k_anchor=K_ANCHOR):
    items=[]; cur=start_dt.replace(hour=12,minute=0,second=0,microsecond=0)
    if cur<start_dt: cur=cur+timedelta(days=1)
    while cur<end_dt:
        dj,dc,djidx=day_ganji_solar(cur,k_anchor); g,j=dj[0],dj[1]
        items.append({'date':cur.date(),'gan':g,'ji':j,'six':f'{six_for_stem(day_stem,g)}/{six_for_branch(day_stem,j)}'})
        cur=cur+timedelta(days=1)
    return items

# ── 사령(司令) 데이터 ──
SARYEONG = {
    "해": {"early_15": "갑", "late_15": "임"},
    "자": {"early_15": "임", "late_15": "계"},
    "축": {"early_15": "계", "late_15": "신"},
    "인": {"early_15": "병", "late_15": "갑"},
    "묘": {"early_15": "갑", "late_15": "을"},
    "진": {"early_15": "을", "late_15": "계"},
    "사": {"early_15": "경", "late_15": "병"},
    "오": {"early_15": "병", "late_15": "정"},
    "미": {"early_15": "을", "late_15": "정"},
    "신": {"early_15": "임", "late_15": "경"},
    "유": {"early_15": "경", "late_15": "신"},
    "술": {"early_15": "신", "late_15": "정"},
}

# ── 당령(當令) 데이터 ──
DANGRYEONG = [
    {"months":["자","축"],"period":"동지~입춘","heaven_mission":"계수","description":"깊이를 더하고, 내면을 정화하며, 감정과 지혜를 축적하는 사명을 받았습니다."},
    {"months":["인","묘"],"period":"입춘~춘분","heaven_mission":"갑목","description":"새로운 시작을 열고, 성장의 씨앗을 틔우는 개척의 사명을 받았습니다."},
    {"months":["묘","진"],"period":"춘분~입하","heaven_mission":"을목","description":"관계를 다듬고, 부드럽게 확장하며 조화를 이루는 사명을 받았습니다."},
    {"months":["사","오"],"period":"입하~하지","heaven_mission":"병화","description":"세상에 빛을 드러내고, 에너지를 외부로 확산하는 사명을 받았습니다."},
    {"months":["오","미"],"period":"하지~입추","heaven_mission":"정화","description":"따뜻함으로 사람을 연결하고, 관계 속에서 의미를 완성하는 사명을 받았습니다."},
    {"months":["신","유"],"period":"입추~추분","heaven_mission":"경금","description":"질서를 세우고, 불필요한 것을 정리하며 기준을 만드는 사명을 받았습니다."},
    {"months":["유","술"],"period":"추분~입동","heaven_mission":"신금","description":"정밀함과 통찰로 본질을 구분하고 다듬는 사명을 받았습니다."},
    {"months":["해","자"],"period":"입동~동지","heaven_mission":"임수","description":"포용과 흐름 속에서 세상을 연결하고 순환시키는 사명을 받았습니다."},
]

def get_saryeong_gan(month_branch, day_from_jieqi):
    sr = SARYEONG.get(month_branch)
    if not sr: return None, None
    if day_from_jieqi < 15:
        return sr["early_15"], "전반15일"
    else:
        return sr["late_15"], "후반15일"

def get_dangryeong(month_branch, dt_solar=None, jie24_solar=None):
    boundary_jie = {'오':'하지','묘':'춘분','유':'추분','자':'동지','해':'입동'}
    if month_branch in boundary_jie and dt_solar and jie24_solar:
        jie_name = boundary_jie[month_branch]
        jie_dt = jie24_solar.get(jie_name)
        if jie_dt:
            matched = [item for item in DANGRYEONG if month_branch in item['months']]
            if len(matched) >= 2:
                return matched[1] if dt_solar >= jie_dt else matched[0]
            elif matched:
                return matched[0]
    for item in DANGRYEONG:
        if month_branch in item['months']:
            return item
    return None

def get_nearby_jeolip(dt_solar):
    year = dt_solar.year
    all_jeolip = []
    for y in [year-1, year, year+1]:
        jie24 = compute_jie24_times_calc(y)
        for name in JIE24_ORDER:
            if name in jie24:
                t = jie24[name]
                all_jeolip.append((name, t))
    all_jeolip.sort(key=lambda x: x[1])
    prev_item = None
    next_item = None
    for item in all_jeolip:
        if item[1] <= dt_solar:
            prev_item = item
        elif next_item is None and item[1] > dt_solar:
            next_item = item
    return prev_item, next_item

# ── 격(格) 카드 데이터 ──
GYEOK_CARDS = [
    {"slug":"geonrok","card_title":"기반을 만드는 사람 · 건록격","icon":"🏛️",
     "one_liner":"배우고 정리해서, 모두가 안심할 수 있는 기준을 만들어주는 사람",
     "story_child":"이 아이는 뭔가 새로 알게 되면 꼭 정리하고 싶어하는 아이예요. 노트든 말이든, 자기만의 방식으로 갈무리해야 직성이 풀리죠. 이 아이에게는 '네가 정리한 거 정말 잘했다'는 칭찬이 큰 힘이 돼요. 스스로 배우고 정리하는 과정을 존중해주세요.",
     "story_young":"당신은 뭔가 새로 배우면 그냥 넘기지 못하고 꼭 정리하는 사람이에요. 노트든 머릿속이든, 배운 걸 나만의 방식으로 갈무리해야 직성이 풀리죠. 지금 쌓고 있는 것들이 나중에 엄청난 자산이 돼요. 당신의 꾸준함이 미래의 기반입니다.",
     "story_mature":"당신은 정리하고 기준을 세우는 일에 타고난 사람이에요. 지금까지 쌓아온 것이 있든 없든, 당신의 그 꼼꼼함과 꾸준함은 변하지 않는 강점이에요. 필요한 게 있으면 도움을 요청해도 괜찮고, 나눌 수 있는 게 있다면 그것도 좋아요. 당신 페이스대로 가면 돼요.",
     "strengths":["정리하고 가르치는 능력이 뛰어나요","꾸준히 공부하고 성장해요","한번 맡으면 끝까지 해내요","사람들에게 안정감을 줘요"],
     "growth_child":["완벽하게 정리될 때까지 시작을 못 하는 경우가 있어요. '일단 해보자'를 자주 말해주세요","자기 기준이 강한 편이라, 친구의 다른 방식도 괜찮다는 걸 알려주세요","혼자 하려는 경향이 있으니, 함께하는 경험을 많이 만들어주세요"],
     "growth_young":["너무 완벽하게 준비하려다 시작이 늦어질 수 있어요. 70%면 충분해요!","내 기준이 확실한 만큼, 다른 사람의 방식도 존중하는 연습이 도움 돼요","혼자 다 하려 하지 말고, 함께하는 연습도 해보세요"],
     "growth_mature":["새로운 방식이 낯설 수 있지만, 열린 마음으로 한번 들어보세요","오랜 경험에서 나온 원칙은 소중해요. 동시에 유연함도 큰 힘이에요","누군가에게 기준을 알려줄 때, 먼저 그 사람의 상황을 들어보면 더 좋아요"],
     "best_environment":["내가 배운 걸 누군가에게 전할 수 있을 때","장기적으로 꾸준히 쌓아가는 일을 할 때","나의 기준과 원칙이 존중받는 환경에서"],
     "praise_keywords":["믿고 맡길 수 있다","정리를 참 잘한다","품격이 있다","꾸준하다","함께하면 안심된다"],
     "keywords":["건록격","건록","월비격","월비"]},

    {"slug":"yangin","card_title":"사람을 지키는 사람 · 양인격","icon":"🛡️",
     "one_liner":"힘든 사람을 보면 가만히 못 있는, 마음이 뜨거운 리더",
     "story_child":"이 아이는 친구가 울면 먼저 다가가는 아이예요. 약한 친구를 괴롭히는 걸 보면 참지 못하고, 자기가 나서서 지키려 해요. 이 마음이 이 아이의 가장 큰 보물이에요. 다만 '너도 보호받을 자격이 있어'라는 말을 자주 해주세요.",
     "story_young":"당신은 친구가 힘들면 '내가 해줄게'가 먼저 나오는 사람이에요. 불공평한 일을 보면 참지 못하고, 내 사람은 꼭 지켜주고 싶어하죠. 그 뜨거운 마음이 당신의 가장 큰 힘이에요. 다만 나 자신도 돌보는 것, 잊지 마세요.",
     "story_mature":"당신은 주변 사람을 지키려는 마음이 누구보다 강한 사람이에요. 지금까지 얼마나 많은 걸 짊어져왔는지 당신이 제일 잘 알죠. 이제는 도움을 받는 것도 괜찮아요. 당신도 누군가에게 기대어 쉬어도 돼요. 그게 약한 게 아니에요.",
     "strengths":["책임감이 아주 강해요","내 사람을 끝까지 지켜줘요","위기 상황에서 더 빛나요","함께하면 든든한 리더예요"],
     "growth_child":["뭐든 자기가 해결하려 하니, '도움을 요청하는 것도 용기'라고 알려주세요","지는 걸 싫어할 수 있어요. 이기는 것보다 함께하는 가치를 알려주세요","에너지가 강한 아이라 운동이나 활동적인 놀이로 발산하게 해주세요"],
     "growth_young":["도와주기 전에 '이건 내가 할 일인가?' 한번 생각해보세요","의리도 중요하지만, 나의 경계도 소중해요","쉬는 것도 책임의 일부예요. 번아웃 조심!"],
     "growth_mature":["모든 걸 혼자 짊어지려 하지 않아도 돼요","도움을 받는 것도 용기예요. 자신을 돌보는 것이 먼저예요","가끔은 '내가 안 해도 괜찮다'고 내려놓는 연습을 해보세요"],
     "best_environment":["팀이나 가족을 이끌어가는 역할을 맡았을 때","내가 지키고 싶은 사람이 있을 때","정의롭고 공정한 환경에서 일할 때"],
     "praise_keywords":["정말 든든하다","의리가 있다","끝까지 책임진다","리더십이 있다","함께하면 힘이 난다"],
     "keywords":["양인격","양인","월겁격","월겁"]},

    {"slug":"sanggwan","card_title":"새 길을 여는 사람 · 상관격","icon":"🔧",
     "one_liner":"남들이 안 되는 걸 되게 만드는, 응용력 넘치는 아이디어뱅크",
     "story_child":"이 아이는 시키는 대로만 하면 답답해하는 아이예요. '왜 이렇게 해야 해?'라는 질문이 많고, 자기만의 방법을 찾으려 해요. 이건 반항이 아니라 창의력이에요! '네 방법도 한번 해볼까?'라고 말해주면 눈이 반짝거릴 거예요.",
     "story_young":"당신은 같은 걸 봐도 '이걸 이렇게 하면 더 좋지 않아?'가 먼저 떠오르는 사람이에요. 정해진 틀만 따르면 답답하고, 나만의 방식으로 풀어야 직성이 풀리죠. 그 톡톡 튀는 생각이 당신의 무기예요!",
     "story_mature":"당신은 어떤 상황에서든 방법을 찾아내는 사람이에요. 지금까지 수많은 문제를 나만의 방식으로 풀어왔죠. 상황이 좋든 어렵든, 그 응용력은 변하지 않는 당신의 재능이에요. 지금 필요한 건 그 재능을 어디에 쓸지 정하는 거예요.",
     "strengths":["어떤 상황에서든 방법을 찾아내요","아이디어가 풍부해요","변화에 강하고 적응이 빨라요","효율적인 길을 잘 찾아요"],
     "growth_child":["'왜?'라는 질문을 귀찮아하지 말고, 함께 답을 찾아가 주세요","규칙을 어기는 게 아니라 다르게 생각하는 거예요. 그 차이를 알려주세요","아이디어를 직접 시도해볼 기회를 많이 만들어주세요"],
     "growth_young":["좋은 아이디어는 실행해야 빛나요. 작게라도 시작해보세요","내 방식이 최고라는 생각이 들 때, 다른 의견도 한번 들어보세요","'나는 맞고 너는 틀려'보다 '서로 방식이 다를 뿐'이라고 생각해보세요"],
     "growth_mature":["다른 사람의 서툰 시도에도 격려를 해주면 좋아요","'내가 해봤는데'보다 '해봐, 응원할게'가 더 큰 힘이 돼요","가끔은 정해진 방식대로 하는 것도 마음이 편할 수 있어요"],
     "best_environment":["자유롭게 아이디어를 낼 수 있을 때","새로운 시도를 환영하는 분위기에서","다양한 사람들과 협업할 때"],
     "praise_keywords":["센스가 좋다","문제 해결이 빠르다","응용력이 탁월하다","혁신적이다","길을 만들어낸다"],
     "keywords":["상관격","상관"]},

    {"slug":"sikshin","card_title":"묵묵히 만드는 사람 · 식신격","icon":"🧪",
     "one_liner":"말보다 결과로 보여주는, 꾸준한 실력파",
     "story_child":"이 아이는 관심 있는 걸 찾으면 놀라울 정도로 집중하는 아이예요. 조용히 혼자 만들고, 실험하고, 반복하면서 실력을 키워가요. '빨리빨리' 재촉하기보다 '천천히 해도 괜찮아'라고 말해주세요. 이 아이는 자기 속도로 갈 때 가장 잘해요.",
     "story_young":"당신은 '직접 해봐야 안다'는 마음으로 하나씩 배워가는 사람이에요. 화려한 말보다 실력으로 보여주고 싶고, 조용히 몰입할 때 가장 행복하죠. 지금 쌓고 있는 실력이 나중에 엄청난 결과를 만들어낼 거예요.",
     "story_mature":"당신은 묵묵히 쌓아가는 것의 가치를 아는 사람이에요. 결과가 당장 보이지 않더라도, 당신의 꾸준함은 사라지지 않아요. 지금 상황이 어떻든, 당신이 가진 실력과 경험은 진짜예요. 자신을 믿어도 돼요.",
     "strengths":["한 가지에 깊이 몰입해요","결과로 증명하는 실력파예요","꾸준하게 성장해요","맡은 일을 책임감 있게 해내요"],
     "growth_child":["이 아이만의 속도가 있어요. 다른 아이와 비교하지 말아주세요","혼자 하는 걸 좋아하지만, 가끔 함께하는 기쁨도 알려주세요","완성하지 못해도 괜찮다고 말해주세요. 과정도 충분히 가치 있어요"],
     "growth_young":["혼자 다 하려 하지 말고, 도움을 요청하는 것도 실력이에요","완벽하지 않아도 중간 결과를 공유해보세요. 피드백이 성장을 빠르게 해요","가끔은 쉬어가면서 하는 게 더 좋은 결과를 만들어요"],
     "growth_mature":["당장 성과가 안 보여도, 당신의 실력은 사라지지 않아요","완벽하지 않아도 괜찮아요. 있는 그대로도 충분해요","이제는 성과만큼, 즐기는 것도 자신에게 허락해주세요"],
     "best_environment":["방해받지 않고 집중할 수 있을 때","내 전문성을 인정받는 환경에서","충분한 시간을 갖고 일할 수 있을 때"],
     "praise_keywords":["실력이 있다","뭐든 척척 해낸다","꾸준하다","믿을 수 있다","결과로 말한다"],
     "keywords":["식신격","식신"]},

    {"slug":"jeongin","card_title":"든든한 길잡이 · 정인격","icon":"📚",
     "one_liner":"아는 것을 나누며 사람들에게 방향을 알려주는 따뜻한 멘토",
     "story_child":"이 아이는 새로 배운 걸 동생이나 친구에게 설명하는 걸 좋아하는 아이예요. 차분하고 꼼꼼해서 실수가 적고, 선생님한테 신뢰를 받기 쉬워요. '네가 알려준 거 덕분에 도움이 됐어'라는 말이 이 아이에겐 최고의 칭찬이에요.",
     "story_young":"당신은 새로운 걸 배우면 깔끔하게 정리하고, 친구에게 '이거 이렇게 하면 돼!'라고 알려주는 걸 잘해요. 차분하고 꼼꼼해서 실수가 적고, 주변에서 신뢰를 받아요. 지금 쌓고 있는 지식이 나중에 가장 큰 무기가 될 거예요.",
     "story_mature":"당신은 차분하게 정리하고 방향을 잡아주는 데 타고난 사람이에요. 지금까지 배워온 것들이 많든 적든, 그 꼼꼼함은 변하지 않는 당신의 강점이에요. 새로운 걸 배우는 데 늦은 나이는 없어요. 궁금한 게 있으면 언제든 시작해도 괜찮아요.",
     "strengths":["복잡한 것도 쉽게 설명해요","차분하고 꼼꼼해요","사람들에게 방향을 잡아줘요","한번 맡으면 정확하게 해내요"],
     "growth_child":["새로운 것에 겁을 낼 수 있어요. '틀려도 괜찮아'를 자주 말해주세요","정답만 고집할 수 있으니, 여러 답이 있을 수 있다는 걸 알려주세요","아는 것을 자랑하지 않고 나누는 법을 가르쳐주세요"],
     "growth_young":["알고 있는 것에만 머물지 말고, 새로운 분야에도 도전해보세요","정답을 알려주기보다, 함께 찾아가는 것도 좋은 방법이에요","완벽하게 알아야 말할 수 있다는 생각을 조금 내려놓아보세요"],
     "growth_mature":["세상이 빠르게 변해도, 배우려는 마음만 있으면 충분해요","모르는 게 있어도 괜찮아요. 그게 부끄러운 일이 아니에요","가르칠 때 '내가 맞다'보다 '너는 어떻게 생각해?'를 먼저 물어보면 더 좋아요"],
     "best_environment":["누군가를 가르치거나 도울 수 있을 때","체계적으로 일할 수 있는 환경에서","나의 지식이 가치를 인정받을 때"],
     "praise_keywords":["아는 게 정말 많다","설명을 참 잘한다","꼼꼼하다","믿음직하다","함께하면 배울 게 많다"],
     "keywords":["정인격","정인"]},

    {"slug":"pyeonin","card_title":"가능성을 보는 사람 · 편인격","icon":"🌙",
     "one_liner":"남들이 못 보는 것을 먼저 느끼고, 새로운 의미를 만들어내는 사람",
     "story_child":"이 아이는 상상의 세계가 풍부한 아이예요. 혼자 공상에 빠져 있는 것처럼 보여도, 머릿속에서는 놀라운 이야기가 펼쳐지고 있어요. '뭐 하고 있어?'보다 '무슨 생각 해?'라고 물어봐주세요. 이 아이의 감수성은 정말 큰 재능이에요.",
     "story_young":"당신은 같은 상황에서도 다른 사람들이 보지 못하는 가능성을 느끼는 사람이에요. 상상력이 풍부하고 감수성이 깊어서, 당신의 아이디어에는 사람 마음을 움직이는 힘이 있어요. 그 감각을 믿고 키워가세요!",
     "story_mature":"당신은 남들이 못 보는 것을 느끼는 사람이에요. 그 직관과 감각은 지금도 여전해요. 상황이 어떻든, 당신의 그 깊은 감수성은 사라지지 않는 재능이에요. 필요하면 그 감각을 다시 꺼내 써도 돼요. 아직 늦지 않았어요.",
     "strengths":["상상력과 직관이 뛰어나요","사람의 마음을 잘 읽어요","의미 있는 것을 만들어내요","남들이 못 보는 것을 먼저 봐요"],
     "growth_child":["공상이 많다고 혼내지 말아주세요. 그게 이 아이의 창의력이에요","현실과 상상의 균형을 잡아주면 좋아요. '그걸 어떻게 만들어볼까?'라고 물어봐주세요","감정 표현을 잘 하게 도와주세요. '지금 기분이 어때?'를 자주 물어봐주세요"],
     "growth_young":["좋은 아이디어가 떠오르면 바로 적어두세요. 실행까지 연결하는 연습이 필요해요","상상만 하지 말고, 현실적인 첫 걸음부터 떼어보세요","감정을 표현한 뒤에 '그래서 이렇게 해줘'라는 구체적 요청을 붙여보세요"],
     "growth_mature":["직관을 믿되, 가끔은 현실적인 확인도 함께 해보세요","혼자만의 세계에 너무 오래 있지 말고, 사람들과 이야기하는 시간을 가져보세요","당신의 감각은 여전해요. 자신감을 가지셔도 돼요"],
     "best_environment":["자유롭게 상상하고 기획할 수 있을 때","감성을 존중하는 사람들과 함께할 때","새로운 시도에 열려 있는 환경에서"],
     "praise_keywords":["창의적이다","따뜻하다","상상력이 놀랍다","의미를 만든다","영감을 준다"],
     "keywords":["편인격","편인"]},

    {"slug":"jeongjae","card_title":"살림을 잘하는 사람 · 정재격","icon":"🧱",
     "one_liner":"현실적이고 알뜰하게, 오래가는 안정을 만들어가는 사람",
     "story_child":"이 아이는 또래보다 현실적이고 야무진 편이에요. 용돈을 아껴 모으거나, 물건을 아끼는 모습을 보일 거예요. '살림 잘하겠다'는 칭찬이 이 아이에겐 뿌듯한 말이에요. 다만 가끔은 '써도 괜찮아'라는 말도 해주세요.",
     "story_young":"당신은 또래보다 현실적이고 알뜰한 편이에요. 돈이든 시간이든 낭비하지 않고, 소중한 것을 잘 지키죠. 아직 젊지만 그 안정감이 주변 사람들에게 신뢰를 줘요. 가끔은 새로운 도전도 해보면 더 넓은 세상이 보일 거예요.",
     "story_mature":"당신은 현실 감각이 뛰어난 사람이에요. 상황이 좋든 어렵든, 소중한 것을 지키고 관리하는 능력은 변하지 않는 당신의 강점이에요. 지금 가진 것이 많든 적든, 당신의 알뜰함과 책임감은 진짜 실력이에요.",
     "strengths":["현실 감각이 뛰어나요","소중한 것을 잘 지켜요","계획적이고 알뜰해요","내 사람을 책임감 있게 돌봐요"],
     "growth_child":["아끼는 것도 좋지만, 나누는 기쁨도 알려주세요","새로운 것에 겁을 낼 수 있어요. 작은 모험을 함께 해보세요","물질적인 것 외에 감정 표현도 중요하다는 걸 알려주세요"],
     "growth_young":["너무 안전한 것만 고르다 보면 기회를 놓칠 수 있어요. 가끔은 모험도!","모든 걸 혼자 관리하려 하지 말고, 믿을 수 있는 사람에게 맡겨보세요","의미 있는 곳에 쓰는 것도 투자예요"],
     "growth_mature":["지금 가진 것의 가치를 스스로 인정해주세요","변화가 두려울 수 있지만, 작은 것부터 시도해보는 건 괜찮아요","가끔은 계획 없이 즉흥적으로 해보는 것도 마음이 편해질 수 있어요"],
     "best_environment":["안정적으로 쌓아갈 수 있는 일을 할 때","내가 관리하고 운영하는 역할을 맡았을 때","노력한 만큼 결과가 돌아오는 환경에서"],
     "praise_keywords":["한결같다","살림을 잘한다","안정적이다","믿음직하다","알뜰하다"],
     "keywords":["정재격","정재"]},

    {"slug":"pyeonjae","card_title":"세상을 넓히는 사람 · 편재격","icon":"🌍",
     "one_liner":"새로운 사람, 새로운 기회를 찾아 세상을 넓혀가는 사람",
     "story_child":"이 아이는 새로운 친구를 사귀는 걸 좋아하고, 모르는 곳에 가는 걸 신나해하는 아이예요. 에너지가 넘쳐서 가만히 있으면 답답해하죠. '여기저기 다니지 말고'보다 '오늘은 어디 가볼까?'가 이 아이에겐 더 좋은 말이에요.",
     "story_young":"당신은 같은 자리에만 있으면 답답한 사람이에요. 새로운 사람, 새로운 경험, 더 넓은 세상이 당신을 설레게 하죠. 그 에너지가 당신의 가장 큰 매력이에요!",
     "story_mature":"당신은 새로운 것을 찾아 움직이는 에너지가 있는 사람이에요. 지금 상황이 어떻든, 그 도전 정신은 사라지지 않아요. 다시 시작하고 싶다면 시작해도 돼요. 당신의 에너지는 나이와 상관없이 빛나요.",
     "strengths":["사람을 모으고 연결하는 힘이 있어요","새로운 기회를 잘 찾아요","도전을 두려워하지 않아요","함께하면 에너지가 넘쳐요"],
     "growth_child":["여러 가지를 동시에 하려 할 수 있어요. '하나만 먼저 끝내볼까?'라고 도와주세요","시작은 잘하지만 마무리가 약할 수 있어요. 끝까지 하는 습관을 길러주세요","가까운 사람에게도 관심을 쏟는 연습을 시켜주세요"],
     "growth_young":["동시에 너무 많은 일을 벌리지 말고, 2~3개에 집중해보세요","시작하는 것만큼 마무리하는 것도 중요해요","가까운 사람에게도 관심을 쏟아주세요"],
     "growth_mature":["새로 시작하는 것도 좋고, 이미 가진 것을 돌보는 것도 좋아요","에너지를 쓸 곳을 정할 때, 정말 중요한 것 2~3개에 집중해보세요","때로는 멈춰서 쉬는 것도 다음 도전을 위한 준비예요"],
     "best_environment":["다양한 사람들을 만날 수 있을 때","새로운 분야에 도전할 기회가 있을 때","자유롭게 움직이며 일할 수 있는 환경에서"],
     "praise_keywords":["진취적이다","도전적이다","에너지가 넘친다","사람을 모은다","세상을 넓힌다"],
     "keywords":["편재격","편재"]},

    {"slug":"jeonggwan","card_title":"함께 만드는 사람 · 정관격","icon":"⚖️",
     "one_liner":"모두가 공정하게 잘 지낼 수 있도록 조율하는 조화의 운영자",
     "story_child":"이 아이는 놀이할 때도 규칙을 중요하게 생각하는 아이예요. 누가 새치기하면 '그건 안 돼!'라고 말하고, 모두가 공평하길 바라요. 이 아이의 정의감을 칭찬해주세요. 다만 '규칙보다 마음이 먼저일 때도 있어'라는 것도 알려주세요.",
     "story_young":"당신은 불공평한 걸 보면 마음이 불편한 사람이에요. 모두가 공정하게 대우받길 바라고, 자연스럽게 '중재자' 역할을 맡게 되죠. 성실하고 책임감 있는 당신은 어디서든 신뢰를 받아요.",
     "story_mature":"당신은 사람들 사이에서 균형을 잡아주는 사람이에요. 지금까지 얼마나 많은 조율을 해왔는지 당신이 제일 잘 알죠. 상황이 어떻든, 당신의 공정함과 성실함은 변하지 않는 강점이에요. 지치면 쉬어가도 괜찮아요.",
     "strengths":["공정하고 균형 잡힌 시각이 있어요","사람들 사이를 잘 조율해요","맡은 일에 성실해요","함께 일하기 편안한 사람이에요"],
     "growth_child":["규칙을 너무 엄격하게 적용하면 친구들이 부담스러워할 수 있어요. 유연함도 알려주세요","'틀렸어!'보다 '이렇게 하면 어때?'로 말하는 연습을 도와주세요","완벽하지 않아도 괜찮다는 걸 자주 말해주세요"],
     "growth_young":["규칙을 지키는 것도 중요하지만, 예외도 있다는 걸 기억하세요","가끔은 원칙보다 사람의 마음을 먼저 읽어보세요","완벽한 공정함은 없어요. 최선을 다하는 것으로 충분해요"],
     "growth_mature":["오랜 시간 지켜온 기준은 소중해요. 동시에 시대에 맞게 유연해져도 괜찮아요","누군가에게 규칙을 알려줄 때, 먼저 이유를 설명해주면 더 좋아요","가끔은 규칙 밖에서 쉬어가는 것도 필요해요. 자신에게 너그러워지세요"],
     "best_environment":["팀이나 조직을 운영하는 역할을 맡았을 때","공정함이 중요한 환경에서","서로 협력하며 성과를 내는 분위기에서"],
     "praise_keywords":["성실하다","공정하다","함께하면 편하다","조직을 살린다","믿고 맡길 수 있다"],
     "keywords":["정관격","정관"]},

    {"slug":"pyeongwan","card_title":"핵심을 꿰뚫는 사람 · 편관격","icon":"🦅",
     "one_liner":"남들이 놓치는 것을 찾아내고, 흐름을 바로잡는 날카로운 눈",
     "story_child":"이 아이는 눈치가 빠르고 관찰력이 뛰어난 아이예요. 뭔가 이상하면 바로 알아차리고, '이건 왜 이래?'라고 물어요. 그 날카로움이 이 아이의 재능이에요. 다만 지적하기 전에 상대방 기분도 생각하는 법을 알려주세요.",
     "story_young":"당신은 뭔가 이상하면 바로 알아차리는 사람이에요. '이건 좀 아닌데?'를 빠르게 감지하고, 문제를 정면으로 마주해요. 그 날카로운 눈이 팀에서 빛을 발하죠!",
     "story_mature":"당신은 핵심을 정확히 짚어내는 사람이에요. 그 안목과 판단력은 지금도 여전해요. 상황이 좋든 어렵든, 문제를 파악하는 능력은 변하지 않는 당신의 강점이에요. 지금 필요한 건 그 눈을 어디에 쓸지 정하는 거예요.",
     "strengths":["문제를 빠르게 찾아내요","과감한 결단력이 있어요","위기에서 더 빛나는 사람이에요","핵심을 정확히 짚어요"],
     "growth_child":["지적하기 전에 '어떻게 말하면 좋을까?' 생각하는 연습을 시켜주세요","사람을 평가하지 않고 상황을 보는 법을 알려주세요","강한 에너지를 건강하게 쓸 수 있도록 운동이나 활동을 권해주세요"],
     "growth_young":["지적하기 전에 '어떻게 말하면 잘 전달될까?' 한번 생각해보세요","사람을 평가하기보다 상황에 집중해보세요","강한 말 뒤에 대안을 함께 제시하면 훨씬 좋아요"],
     "growth_mature":["직관은 여전히 정확해요. 다만 설명을 곁들이면 더 설득력이 있어요","누군가를 가르칠 때 '왜 안 되는지'보다 '이렇게 하면 된다'를 먼저 알려주면 좋아요","가끔은 눈감아주는 것도 여유예요. 자신에게도 너그러워지세요"],
     "best_environment":["내 판단과 결정이 존중받을 때","빠르게 변하는 환경에서 방향을 잡을 때","명확한 목표가 있는 프로젝트를 이끌 때"],
     "praise_keywords":["안목이 좋다","결단력 있다","핵심을 짚는다","위기를 잡는다","특출나다"],
     "keywords":["편관격","편관","중기격"]},
]

def find_geok_card(geok_name):
    geok_clean = geok_name.replace('격','').strip()
    for card in GYEOK_CARDS:
        for kw in card["keywords"]:
            if kw in geok_name or kw in geok_clean:
                return card
    return None

# ────────────────────────────────────────────
# ★ 신뢰 장치 3종: 보정값 표시 / 경계 경고 / 진태양시 비교
# ────────────────────────────────────────────
from korea_tz_history import describe_timezone_for_date, get_wall_clock_utc_offset

def tz_label_for_date(d):
    """날짜에 해당하는 표준시 라벨"""
    info = describe_timezone_for_date(d if isinstance(d, date) and not isinstance(d, datetime) else d.date() if hasattr(d, 'date') else d)
    label = info['standard']
    if info['dst_active']:
        label += '+DST'
    return f"{label} ({info['utc_string']})"

def calc_correction_detail(birth_date, longitude=DEFAULT_LONGITUDE):
    """보정값 상세 내역 (UI 표시용)"""
    info = describe_timezone_for_date(birth_date)
    std_meridian = info['meridian']
    lon_corr = (longitude - std_meridian) * 4  # 1도=4분
    dst_corr = -info['dst_advance_min'] if info['dst_active'] else 0
    total = lon_corr + dst_corr
    return {
        'standard': info['standard'],
        'utc_string': info['utc_string'],
        'dst_active': info['dst_active'],
        'dst_min': dst_corr,
        'std_meridian': std_meridian,
        'longitude': longitude,
        'lon_corr_min': round(lon_corr, 1),
        'total_min': round(total, 1),
    }

def render_correction_html(corr, eot_min=0):
    """보정값 상세 HTML"""
    parts = []
    parts.append(f"<b>📌 법정시</b>: {corr['standard']} ({corr['utc_string']})")
    if corr['dst_active']:
        parts.append(f"<b>☀️ 써머타임</b>: 적용 중 ({corr['dst_min']:+.0f}분)")
    parts.append(f"<b>🧭 경도보정</b>: 기준{corr['std_meridian']:.1f}°→출생지{corr['longitude']:.1f}° ({corr['lon_corr_min']:+.1f}분)")
    if abs(eot_min) > 0.5:
        parts.append(f"<b>⏱️ 균시차</b>: {eot_min:+.1f}분")
    total = corr['total_min'] + eot_min
    parts.append(f"<b>합계 보정</b>: <span style='font-size:14px;font-weight:bold;color:#8b4513;'>{total:+.0f}분</span>")
    return '<div class="tz-info-box">' + '<br>'.join(parts) + '</div>'

def check_boundary_warning(dt_solar, jie24_solar):
    """절입/시주 경계 경고"""
    warnings = []
    # 절입 ±2시간 체크
    for name, t in jie24_solar.items():
        diff_min = abs((dt_solar - t).total_seconds()) / 60
        if diff_min <= 120:
            warnings.append(f"⚠️ 절입 경계: <b>{name}</b> 시각과 <b>{diff_min:.0f}분</b> 차이 — 월주가 달라질 수 있어 정밀검증 권장")
            break
    # 시주 경계 ±30분 체크
    mins = dt_solar.hour * 60 + dt_solar.minute
    si_boundaries = [23*60, 1*60, 3*60, 5*60, 7*60, 9*60, 11*60, 13*60, 15*60, 17*60, 19*60, 21*60]
    for sb in si_boundaries:
        diff = abs((mins - sb + 720) % 1440 - 720)
        if diff <= 30:
            warnings.append(f"⚠️ 시주 경계: 시주 전환 시각과 <b>{diff}분</b> 차이 — 시주가 달라질 수 있어 정밀검증 권장")
            break
    return warnings

def render_tst_compare_html(dt_wall, dt_tst, fp_wall, fp_tst):
    """벽시계 vs 진태양시 비교 HTML"""
    diff = (dt_tst - dt_wall).total_seconds() / 60
    html = '<div class="tst-compare">'
    html += f'<b>🔬 정밀검증 (진태양시 비교)</b><br>'
    html += f'입력시각(벽시계): <b>{dt_wall.strftime("%H:%M")}</b> → 진태양시: <b>{dt_tst.strftime("%H:%M")}</b> (차이: {diff:+.0f}분)<br>'
    if fp_wall['hour'] != fp_tst['hour']:
        html += f'<span style="color:#c00;font-weight:bold;">⚠ 시주 차이 발견! 벽시계={fp_wall["hour"]} / 진태양시={fp_tst["hour"]}</span><br>'
    else:
        html += f'시주 동일: {fp_wall["hour"]} ✅<br>'
    if fp_wall['month'] != fp_tst['month']:
        html += f'<span style="color:#c00;font-weight:bold;">⚠ 월주 차이 발견! 벽시계={fp_wall["month"]} / 진태양시={fp_tst["month"]}</span>'
    else:
        html += f'월주 동일: {fp_wall["month"]} ✅'
    html += '</div>'
    return html

# ────────────────────────────────────────────

MOBILE_CSS = """
<style>
:root{--bg:#ffffff;--bg2:#f5f5f0;--card:#e8e4d8;--acc:#8b6914;--text:#2c2416;--sub:#6b5a3e;--r:10px;--bdr:#c8b87a;}
*{box-sizing:border-box;}
html{font-size:16px;}
body,.stApp{background:var(--bg)!important;color:var(--text)!important;font-family:"Noto Serif KR","Malgun Gothic",serif;-webkit-text-size-adjust:100%;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:0.3rem 0.5rem!important;max-width:480px!important;margin:0 auto!important;}
.stTextInput input,.stNumberInput input{background:#fff!important;color:var(--text)!important;border:1px solid var(--bdr)!important;border-radius:8px!important;font-size:16px!important;}
.stRadio label{color:var(--text)!important;font-size:15px!important;}
.stSelectbox label,.stCheckbox label{font-size:15px!important;}
.stButton>button{background:linear-gradient(135deg,#c8b87a,#a0945e)!important;color:#fff!important;border:1px solid var(--acc)!important;border-radius:6px!important;width:100%!important;font-size:12px!important;font-weight:bold!important;padding:2px 0px!important;white-space:nowrap!important;overflow:hidden;min-height:0!important;height:24px!important;line-height:1!important;}
.page-hdr{background:linear-gradient(135deg,#c8b87a,#a0945e);border-bottom:2px solid var(--acc);padding:6px;text-align:center;font-size:18px;font-weight:bold;color:#fff;letter-spacing:4px;margin-bottom:4px;}
.saju-wrap{background:var(--bg2);border:1px solid var(--bdr);border-radius:var(--r);padding:4px 4px 2px;margin-bottom:2px;}
.saju-table{width:100%;border-collapse:separate;border-spacing:4px;table-layout:fixed;}
.saju-table th{font-size:13px;color:var(--sub);text-align:center;padding:2px 0;}
.saju-table .lb td{font-size:12px;color:var(--sub);text-align:center;padding:1px 0;}
.gcell,.jcell{text-align:center;padding:0;}
.gcell div,.jcell div{display:flex;align-items:center;justify-content:center;width:100%;height:44px;border-radius:8px;font-weight:900;font-size:26px;border:1px solid rgba(0,0,0,.15);margin:1px auto;}
.sec-title{font-size:15px;color:var(--acc);font-weight:bold;padding:4px 8px;border-left:3px solid var(--acc);margin:6px 0 4px;}
.geok-box{background:rgba(200,184,122,.2);border:1px solid var(--acc);border-radius:8px;padding:6px 10px;margin:2px 0;font-size:13px;color:var(--text);}
.geok-name{font-size:17px;font-weight:900;color:#8b4513;margin-bottom:2px;}
.geok-why{font-size:12px;color:var(--sub);line-height:1.5;}
.today-banner{background:linear-gradient(135deg,#f5f0e8,#ede0c4);border:1px solid var(--acc);border-radius:8px;padding:4px 10px;margin-bottom:2px;font-size:13px;color:var(--sub);text-align:center;}
.sel-info{background:var(--card);border:1px solid var(--acc);border-radius:8px;padding:4px 10px;margin-bottom:4px;font-size:14px;color:var(--text);text-align:center;}
.cal-wrap{background:var(--bg2);border:1px solid var(--bdr);border-radius:var(--r);overflow:hidden;margin-bottom:6px;}
.cal-header{background:#c8b87a;text-align:center;padding:8px;font-size:16px;color:#fff;font-weight:bold;}
.cal-table{width:100%;border-collapse:collapse;}
.cal-table th{background:#d4c48a;color:#5a3e0a;font-size:12px;text-align:center;padding:5px 2px;border:1px solid var(--bdr);}
.cal-table td{text-align:center;padding:3px 1px;border:1px solid var(--bdr);font-size:12px;color:var(--text);vertical-align:top;min-width:42px;height:80px;}
.cal-table td.empty{background:#f0ece4;}
.cal-table td .dn{font-size:15px;font-weight:bold;margin-bottom:1px;}
.cal-table td.today-cell{background:#ffe8a0;border:1px solid var(--acc);}
.cal-table td.sun .dn{color:#E53935;}
.cal-table td.sat .dn{color:#1565C0;}
.geok-card-front{background:linear-gradient(135deg,rgba(200,184,122,.25),rgba(160,148,94,.15));border:1px solid var(--acc);border-radius:12px;padding:14px 16px;margin:4px 0 2px;cursor:pointer;}
.geok-card-title{font-size:16px;font-weight:900;color:#8b4513;}
.geok-card-oneliner{font-size:13px;color:var(--sub);line-height:1.5;margin-top:4px;}
.geok-card-detail{background:#faf6ed;border:1px solid #d4b86a;border-radius:10px;padding:14px 16px;margin:4px 0 8px;font-size:14px;color:var(--text);line-height:1.7;}
.geok-tag{display:inline-block;background:#f0e8c8;color:#7a5a1a;border:1px solid #c8a84a;border-radius:20px;padding:3px 10px;font-size:12px;margin:2px;}
.ai-section{background:linear-gradient(135deg,#fff0f5,#ffe4ee);border:1px solid #f4a0c0;border-radius:12px;padding:10px;margin:6px 0 2px;}
.bottom-btns{display:flex;gap:6px;margin:4px 0 4px;}
.bottom-btns a{flex:1;border:none;border-radius:10px;padding:10px 4px;text-align:center;font-size:13px;font-weight:bold;text-decoration:none!important;display:block;}
.bottom-btn-ai{background:linear-gradient(135deg,#a8d8ea,#82c4d8);color:#3a2a14!important;}
.bottom-btn-yt{background:linear-gradient(135deg,#f5d5a0,#e8bf78);color:#3a2a14!important;}
.bottom-btn-chat{background:linear-gradient(135deg,#c4e0b8,#a0cc8e);color:#3a2a14!important;}
label{color:var(--text)!important;font-size:15px!important;}
div[data-testid='stHorizontalBlock']{gap:2px!important;margin-bottom:-4px!important;}
div[data-testid='column']{padding:0 1px!important;}
div[data-testid='stVerticalBlock']>div{margin-bottom:-2px!important;}
div[data-testid='stVerticalBlock']>div:has(iframe){margin-top:-2px!important;margin-bottom:-2px!important;}
div[data-testid='stExpander']{margin-top:4px!important;}
/* ★ 신뢰 장치 3종 스타일 */
.tz-info-box{background:#f8f4e8;border:1px solid #d4c48a;border-radius:8px;padding:8px 10px;margin:4px 0;font-size:13px;color:var(--sub);line-height:1.6;}
.tz-info-box b{color:var(--text);}
.boundary-warn{background:#fff3e0;border:1px solid #f0a030;border-radius:8px;padding:8px 10px;margin:4px 0;font-size:13px;color:#8b4500;line-height:1.5;}
.tst-compare{background:#f0f4ff;border:1px solid #90a0d0;border-radius:8px;padding:8px 10px;margin:4px 0;font-size:13px;color:#2a3060;line-height:1.6;}
</style>
"""

def hanja_gan(g): return HANJA_GAN[CHEONGAN.index(g)]
def hanja_ji(j): return HANJA_JI[JIJI.index(j)]

def gan_card_html(g, size=52, fsize=26):
    bg=GAN_BG.get(g,"#888"); fg=gan_fg(g); hj=hanja_gan(g)
    return f'<div style="width:{size}px;height:{size}px;border-radius:8px;background:{bg};color:{fg};display:flex;align-items:center;justify-content:center;font-size:{fsize}px;font-weight:900;border:1px solid rgba(0,0,0,.15);">{hj}</div>'

def ji_card_html(j, size=52, fsize=26):
    bg=BR_BG.get(j,"#888"); fg=br_fg(j); hj=hanja_ji(j)
    return f'<div style="width:{size}px;height:{size}px;border-radius:8px;background:{bg};color:{fg};display:flex;align-items:center;justify-content:center;font-size:{fsize}px;font-weight:900;border:1px solid rgba(0,0,0,.15);">{hj}</div>'

def render_saju_table(fp, ilgan):
    yg,yj=fp['year'][0],fp['year'][1]; mg,mj=fp['month'][0],fp['month'][1]
    dg,dj=fp['day'][0],fp['day'][1]; sg,sj=fp['hour'][0],fp['hour'][1]
    cols=[(sg,sj,'시주'),(dg,dj,'일주'),(mg,mj,'월주'),(yg,yj,'년주')]
    ss_g=[six_for_stem(ilgan,sg),'일간',six_for_stem(ilgan,mg),six_for_stem(ilgan,yg)]
    ss_j=[six_for_branch(ilgan,sj),six_for_branch(ilgan,dj),six_for_branch(ilgan,mj),six_for_branch(ilgan,yj)]
    html='<div class="saju-wrap"><table class="saju-table"><thead><tr>'
    for g,j,lbl in cols: html+=f'<th>{lbl}</th>'
    html+='</tr><tr class="lb">'
    for i,(g,j,_) in enumerate(cols): html+=f'<td>{ss_g[i]}</td>'
    html+='</tr></thead><tbody><tr>'
    for g,j,_ in cols: html+=f'<td class="gcell">{gan_card_html(g)}</td>'
    html+='</tr><tr>'
    for g,j,_ in cols: html+=f'<td class="jcell">{ji_card_html(j)}</td>'
    html+='</tr><tr class="lb">'
    for i,(_,j,__) in enumerate(cols): html+=f'<td>{ss_j[i]}</td>'
    html+='</tr></tbody></table></div>'
    return html

def render_geok_card_html(card, show_detail=False, user_age=30):
    if not card: return ''
    icon_title = f'{card["icon"]} {card["card_title"]}'
    front = (
        '<div class="geok-card-front">'
        f'<div class="geok-card-title">{icon_title}</div>'
        f'<div class="geok-card-oneliner">{card["one_liner"]}</div>'
        '<div style="font-size:10px;color:#a0845e;margin-top:6px;text-align:right;">▼ 상세보기 클릭</div>'
        '</div>'
    )
    if not show_detail:
        return front
    # 3구간: ~19세(아이) / 20~44세(청년) / 45세~(중년이상)
    if user_age <= 19:
        tier = "child"
    elif user_age <= 44:
        tier = "young"
    else:
        tier = "mature"
    story = card.get(f"story_{tier}", card.get("story_young",""))
    growth = card.get(f"growth_{tier}", card.get("growth_young",[]))
    lbl_str = "💪 나의 강점"
    lbl_grow = "🌱 알아두면 좋은 점"
    lbl_env = "✨ 이런 환경에서 빛나요"
    lbl_praise = "🎉 나를 표현하는 말"
    strengths_html = ''.join([f'<span class="geok-tag">✦ {s}</span>' for s in card["strengths"]])
    growth_html = ''.join([f'<li style="margin-bottom:4px;">{t}</li>' for t in growth])
    env_html = ''.join([f'<li style="margin-bottom:4px;">{t}</li>' for t in card.get("best_environment", [])])
    praise_html = ''.join([f'<span class="geok-tag" style="background:#e8f8e8;color:#2a6a2a;border-color:#6ab46a;">✧ {p}</span>' for p in card["praise_keywords"]])
    detail = (
        '<div class="geok-card-detail">'
        f'<div style="font-size:15px;font-weight:900;color:#8b4513;margin-bottom:8px;">{icon_title}</div>'
        f'<div style="font-size:12px;margin-bottom:10px;line-height:1.7;color:#3a2a14;">{story}</div>'
        f'<div style="font-size:12px;font-weight:bold;color:#8b6914;margin-bottom:4px;">{lbl_str}</div>'
        f'<div style="margin-bottom:10px;">{strengths_html}</div>'
        f'<div style="font-size:12px;font-weight:bold;color:#c46014;margin-bottom:4px;">{lbl_grow}</div>'
        f'<ul style="margin:0 0 10px;padding-left:18px;font-size:11px;color:#5a3a14;">{growth_html}</ul>'
        f'<div style="font-size:12px;font-weight:bold;color:#8b6914;margin-bottom:4px;">{lbl_env}</div>'
        f'<ul style="margin:0 0 10px;padding-left:18px;font-size:11px;color:#2c2416;">{env_html}</ul>'
        f'<div style="font-size:12px;font-weight:bold;color:#2a6a2a;margin-bottom:4px;">{lbl_praise}</div>'
        f'<div>{praise_html}</div>'
        '</div>'
    )
    return detail

def render_daeun_card(age, g, j, ilgan, active, btn_key, dy_year=0):
    bg_g=GAN_BG.get(g,"#888"); tc_g=gan_fg(g)
    bg_j=BR_BG.get(j,"#888"); tc_j=br_fg(j)
    hj_g=hanja_gan(g); hj_j=hanja_ji(j)
    bdr='2px solid #8b6914' if active else '1px solid #c8b87a'
    bg_card='#d4c48a' if active else '#e8e4d8'
    six_g=six_for_stem(ilgan,g); six_j=six_for_branch(ilgan,j)
    st.markdown(
        f'<div style="text-align:center;font-size:10px;color:#6b5a3e;margin-bottom:0px">{age}세</div>'
        f'<div style="display:flex;flex-direction:column;align-items:center;border:{bdr};border-radius:10px;background:{bg_card};padding:3px 2px;">'
        f'<div style="font-size:9px;color:#5a3e0a;margin-bottom:1px;white-space:nowrap">{six_g}</div>'
        f'<div style="width:30px;height:30px;border-radius:5px;background:{bg_g};color:{tc_g};display:flex;align-items:center;justify-content:center;font-size:16px;font-weight:900;margin-bottom:1px">{hj_g}</div>'
        f'<div style="width:30px;height:30px;border-radius:5px;background:{bg_j};color:{tc_j};display:flex;align-items:center;justify-content:center;font-size:16px;font-weight:900;margin-bottom:1px">{hj_j}</div>'
        f'<div style="font-size:9px;color:#5a3e0a;white-space:nowrap">{six_j}</div>'
        '</div>',
        unsafe_allow_html=True
    )
    return st.button(f'{dy_year}', key=btn_key, use_container_width=True)

def main():
    st.set_page_config(page_title="Dr. Lee's Birth Energy Pattern Analysis", layout='centered', page_icon='🔮', initial_sidebar_state='collapsed')
    st.markdown(MOBILE_CSS, unsafe_allow_html=True)
    # 모바일 핀치줌 허용
    st.markdown('<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes, maximum-scale=5.0">', unsafe_allow_html=True)
    st.markdown('<div class="page-hdr">만 세 력</div>', unsafe_allow_html=True)
    for key,val in [('page','input'),('saju_data',None),('sel_daeun',0),('sel_seun',0),('sel_wolun',0),('show_geok_detail',False),('show_saju_interp',False)]:
        if key not in st.session_state: st.session_state[key]=val
    if st.session_state.page=='input': page_input()
    elif st.session_state.page=='saju': page_saju()
    elif st.session_state.page=='wolun': page_wolun()
    elif st.session_state.page=='ilun': page_ilun()

def page_input():
    now=datetime.now(LOCAL_TZ)
    st.markdown(
        '<div style="text-align:center;margin:4px 0 8px;">'
        '<div style="font-size:12px;color:#a0945e;">Ruach Energy Profile · 진태양시 정밀 계산</div>'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div style="background:#f8f4e8;border:1px solid #d4c48a;border-radius:10px;padding:12px 14px;margin-bottom:8px;">'
        '<div style="font-size:13px;font-weight:bold;color:#8b4513;margin-bottom:8px;">📅 출생 정보</div>',
        unsafe_allow_html=True
    )
    c1,c2=st.columns(2)
    with c1: gender=st.radio('성별',['남','여'],horizontal=True)
    with c2: cal_type=st.radio('달력',['양력','음력','음력윤달'],horizontal=True)
    birth_str=st.text_input('생년월일 (YYYYMMDD)',value=st.session_state.get('_birth_str','19840202'),max_chars=8)
    birth_time=st.text_input('출생시각 (HHMM, 모르면 0000)',value=st.session_state.get('_birth_time','0000'),max_chars=4)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<div style="background:#f8f4e8;border:1px solid #d4c48a;border-radius:10px;padding:12px 14px;margin-bottom:8px;">'
        '<div style="font-size:13px;font-weight:bold;color:#8b4513;margin-bottom:8px;">📍 출생지 · 보정 설정</div>',
        unsafe_allow_html=True
    )
    city = st.selectbox("출생지", list(city_options.keys()))
    longitude = city_options[city]
    apply_solar = st.checkbox("진태양시(경도) 보정 적용", value=True)
    show_tst = st.checkbox("🔬 정밀검증 모드 (벽시계 vs 진태양시 비교)", value=False)
    st.markdown('</div>', unsafe_allow_html=True)

    is_leap = (cal_type == '음력윤달')
    if st.button('🔮 사주 보기', use_container_width=True):
        try:
            bs=re.sub(r'\D','',birth_str); bt=re.sub(r'\D','',birth_time)
            y=int(bs[:4]); m=int(bs[4:6]); d=int(bs[6:8])
            hh=int(bt[:2]) if len(bt)>=2 else 0
            mm_t=int(bt[2:4]) if len(bt)==4 else 0
            base_date=date(y,m,d)
            if cal_type in ('음력','음력윤달') and HAS_LUNAR: base_date=lunar_to_solar(y,m,d,is_leap)
            dt_local=datetime.combine(base_date,time(hh,mm_t)).replace(tzinfo=LOCAL_TZ)
            if apply_solar:
                dt_solar = to_solar_time(dt_local, longitude)
            else:
                dt_solar = dt_local

            fp=four_pillars_from_solar(dt_solar)
            ilgan=fp['day'][0]
            jie12 = compute_jie_times_calc(dt_solar.year)
        
            year_gan=fp['year'][0]
            forward=(is_yang_stem(year_gan)==(gender=='남'))
            start_age=dayun_start_age(dt_solar,jie12,forward)
            daeun=build_dayun_list(fp['m_gidx'],fp['m_bidx'],forward,start_age)
            seun_start=base_date.year
            seun=[]
            for i in range(100):
                sy=seun_start+i; off=(sy-4)%60
                seun.append((sy,CHEONGAN[off%10],JIJI[off%12]))
            jie24 = compute_jie24_times_calc(dt_solar.year)

            if apply_solar:
                for k in jie24:
                    jie24[k] = to_solar_time(jie24[k], longitude)

            jie24_solar = jie24
            pair=MONTH_TO_2TERMS[fp['month'][1]]
            def nearest_t(name):
                cands=[(abs((t-dt_solar).total_seconds()),t) for n,t in jie24_solar.items() if n==name]
                if not cands: return dt_solar
                cands.sort(); return cands[0][1]
            t1=nearest_t(pair[0]); t2=nearest_t(pair[1])
            day_from_jieqi=int((dt_solar-t1).total_seconds()//86400)
            day_from_jieqi=max(0,min(29,day_from_jieqi))
            geok,why=decide_geok(Inputs(
                day_stem=fp['day'][0],month_branch=fp['month'][1],month_stem=fp['month'][0],
                stems_visible=[fp['year'][0],fp['month'][0],fp['day'][0],fp['hour'][0]],
                branches_visible=[fp['year'][1],fp['month'][1],fp['day'][1],fp['hour'][1]],
                solar_dt=dt_solar,first_term_dt=t1,mid_term_dt=t2,day_from_jieqi=day_from_jieqi
            ))
            age_now=calc_age_on(base_date,now)
            sel_du=0
            for idx,item in enumerate(daeun):
                if item['start_age']<=age_now: sel_du=idx
            sel_su=min(age_now, 99)
            st.session_state['_birth_str']=birth_str
            st.session_state['_birth_time']=birth_time

            # ★ 표준시 라벨
            tz_lbl = tz_label_for_date(base_date)
            # ★ 보정값 상세
            corr_detail = calc_correction_detail(base_date, longitude)
            eot_min = equation_of_time_minutes(dt_local.astimezone(timezone.utc)) if apply_solar else 0
            # ★ 경계 경고
            boundary_warns = check_boundary_warning(dt_solar, jie24_solar)
            # ★ 진태양시 비교용 (정밀검증 모드)
            fp_wall = None
            if show_tst and apply_solar:
                # 벽시계(보정 전) 기준 사주도 계산
                fp_wall = four_pillars_from_solar(dt_local)

            st.session_state.saju_data={
                'birth':(base_date.year,base_date.month,base_date.day,hh,mm_t),
                'dt_solar':dt_solar,'dt_local':dt_local,
                'gender':gender,'fp':fp,'daeun':daeun,
                'seun':seun,'seun_start':seun_start,'geok':geok,'why':why,
                't1':t1,'t2':t2,'day_from_jieqi':day_from_jieqi,
                'ilgan':ilgan,'start_age':start_age,'forward':forward,
                'jie24_solar':jie24_solar,
                'longitude': longitude,
                'apply_solar': apply_solar,
                'tz_label': tz_lbl,
                'corr_detail': corr_detail,
                'eot_min': eot_min,
                'boundary_warns': boundary_warns,
                'show_tst': show_tst,
                'fp_wall': fp_wall,
            }
            st.session_state.sel_daeun=sel_du
            st.session_state.sel_seun=sel_su
            st.session_state.sel_wolun=now.month-1
            st.session_state.show_geok_detail=False
            st.session_state.page='saju'
            st.rerun()
        except Exception as e: st.error(f'입력 오류: {e}')

def page_saju():
    data=st.session_state.saju_data
    if not data or 'fp' not in data: st.session_state.page='input'; st.rerun(); return
    now=datetime.now(LOCAL_TZ)
    fp=data['fp']; ilgan=data['ilgan']
    daeun=data['daeun']; seun=data['seun']
    geok=data['geok']; why=data['why']
    sel_du=st.session_state.sel_daeun
    birth_year=data['birth'][0]

    if st.button('← 입력으로'):
        st.session_state.page='input'; st.rerun()

    longitude = data.get('longitude', DEFAULT_LONGITUDE)
    apply_solar = data.get('apply_solar', True)

    if apply_solar:
        now_solar = to_solar_time(now, longitude)
    else:
        now_solar = now
    today_fp=four_pillars_from_solar(now_solar)
    yg,yj=today_fp['year'][0],today_fp['year'][1]
    dg,dj=today_fp['day'][0],today_fp['day'][1]
    mg,mj=today_fp['month'][0],today_fp['month'][1]
    hj_yg=hanja_gan(yg); hj_yj=hanja_ji(yj)
    hj_mg=hanja_gan(mg); hj_mj=hanja_ji(mj)
    hj_dg=hanja_gan(dg); hj_dj=hanja_ji(dj)
    b=data['birth']; birth_display=f'{b[0]}년 {b[1]}월 {b[2]}일 {b[3]:02d}:{b[4]:02d}'
    st.markdown(
        f'<div class="today-banner">'
        f'오늘 {now.strftime("%Y.%m.%d")} · {hj_yg}{hj_yj}년 {hj_mg}{hj_mj}월 {hj_dg}{hj_dj}일'
        f'<br><span style="font-size:11px;color:#8b6914;">입력 생년월일시 · 서기 {birth_display}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(render_saju_table(fp,ilgan), unsafe_allow_html=True)
    longitude = data.get('longitude', DEFAULT_LONGITUDE)
    apply_solar = data.get('apply_solar', True)

    # 표준시/보정 라벨은 expander 시간보정 상세에서 표시
    tz_lbl = data.get('tz_label', '')
    calc_info = f"🔎 {tz_lbl} · 경도 {longitude:.2f}° · "
    calc_info += "진태양시 보정" if apply_solar else "표준시 기준"

    month_ji=fp['month'][1]
    day_from=data['day_from_jieqi']
    du_dir='순행' if data['forward'] else '역행'
    du_age=data['start_age']

    saryeong_gan, saryeong_period = get_saryeong_gan(month_ji, day_from)
    saryeong_six = ten_god_for_stem(ilgan, saryeong_gan) if saryeong_gan else ''
    _jie24_s = data.get('jie24_solar') or {}
    dangryeong_item = get_dangryeong(month_ji, data['dt_solar'], _jie24_s)
    prev_jeolip, next_jeolip = get_nearby_jeolip(data['dt_solar'])
    prev_str = f"{prev_jeolip[0]} {prev_jeolip[1].strftime('%Y.%m.%d %H:%M:%S')}" if prev_jeolip else '-'
    next_str = f"{next_jeolip[0]} {next_jeolip[1].strftime('%Y.%m.%d %H:%M:%S')}" if next_jeolip else '-'

    dr_desc = dangryeong_item["description"] if dangryeong_item else ""
    dr_mission = dangryeong_item["heaven_mission"] if dangryeong_item else "-"
    dr_period = dangryeong_item["period"] if dangryeong_item else "-"

    # 격 박스: 윗줄만 (사령/당령/절입일은 expander로 이동)
    geok_box_html = (
        '<div class="geok-box">'
        f'<div class="geok-name">格 {geok} &nbsp;&nbsp;<span style="font-size:11px;color:var(--sub);font-weight:normal;">{why}</span>'
        f'&nbsp;&nbsp;·&nbsp;&nbsp;<span style="font-size:11px;color:var(--sub);">대운 {du_age}세 {du_dir}</span>'
        '</div>'
        '</div>'
    )
    st.markdown(geok_box_html, unsafe_allow_html=True)

    daeun_rev=list(reversed(daeun))
    cols_du=st.columns(len(daeun))
    for ci,col in enumerate(cols_du):
        real_idx=len(daeun)-1-ci
        item=daeun_rev[ci]
        age=item['start_age']
        g=CHEONGAN[item['g_idx']]; j=MONTH_JI[item['b_idx']]
        dy_year=birth_year+age
        with col:
            clicked=render_daeun_card(age,g,j,ilgan,real_idx==sel_du,f"du_{real_idx}",dy_year)
            if clicked:
                st.session_state.sel_daeun=real_idx
                birth_y=data['birth'][0]
                du_start_age=item['start_age']
                new_seun=[]
                for i in range(100):
                    sy=birth_y+i; off=(sy-4)%60
                    new_seun.append((sy,CHEONGAN[off%10],JIJI[off%12]))
                st.session_state.saju_data['seun']=new_seun
                st.session_state.sel_seun=du_start_age - 1
                st.session_state.page='saju'
                st.rerun()


    sel_su=st.session_state.sel_seun
    seun=data["seun"]
    du_item=daeun[sel_du]
    du_start=du_item['start_age']
    birth_y=data['birth'][0]

    # 현재 대운 구간 범위 (start_age는 1-indexed, seun배열은 0-indexed)
    seun_age_start = du_start - 1  # 31세 대운 → index 30
    seun_age_end = du_start + 8    # → index 39 (31~40세)

    # ★ 전체 세운 타임라인 (오른쪽→왼쪽, 1세가 오른쪽)
    max_age = min(len(seun), max(d['start_age'] for d in daeun) + 11)
    all_seun_reversed = list(range(max_age-1, -1, -1))  # 큰나이→작은나이

    seun_html = '<html><body style="margin:0;padding:0;background:transparent;overflow-y:hidden;">'
    seun_html += '<style>html,body{overflow-y:hidden!important;overflow-x:hidden!important;}'
    seun_html += '#seun-timeline{overflow-x:scroll!important;overflow-y:hidden!important;-webkit-overflow-scrolling:touch;}'
    seun_html += '#seun-timeline::-webkit-scrollbar{height:8px;display:block!important;}'
    seun_html += '#seun-timeline::-webkit-scrollbar-track{background:#ece8d8;border-radius:4px;}'
    seun_html += '#seun-timeline::-webkit-scrollbar-thumb{background:#c8b87a;border-radius:4px;min-width:40px;}'
    seun_html += '</style>'
    seun_html += '<div id="seun-timeline" style="overflow-x:scroll;overflow-y:hidden;-webkit-overflow-scrolling:touch;padding:0 0 2px;margin:0;width:100%;">'
    seun_html += '<div style="display:inline-flex;flex-wrap:nowrap;gap:2px;padding:0 4px;">'

    for age_i in all_seun_reversed:
        if age_i >= len(seun):
            continue
        sy, sg, sj = seun[age_i]
        bg_g = GAN_BG.get(sg, "#888"); tc_g = gan_fg(sg)
        bg_j = BR_BG.get(sj, "#888"); tc_j = br_fg(sj)
        hj_sg = hanja_gan(sg); hj_sj = hanja_ji(sj)
        six_g = six_for_stem(ilgan, sg); six_j = six_for_branch(ilgan, sj)

        in_range = (seun_age_start <= age_i <= seun_age_end)
        is_active = (age_i == sel_su)
        display_age = age_i + 1

        if is_active:
            bdr = '2px solid #8b6914'
            bg_card = '#d4c48a'
            opacity = '1'
        elif in_range:
            bdr = '1.5px solid #b8a87a'
            bg_card = '#e8e4d8'
            opacity = '1'
        else:
            bdr = '1px solid #d8d0c0'
            bg_card = '#f0ece0'
            opacity = '0.5'

        anchor = f'id="seun-{age_i}"' if is_active else ''
        range_anchor = f'id="seun-range-start"' if age_i == seun_age_end else ''
        use_anchor = anchor or range_anchor

        seun_html += (
            f'<div {use_anchor} style="display:flex;flex-direction:column;align-items:center;min-width:36px;opacity:{opacity};">'
            f'<div style="font-size:8px;color:#6b5a3e;margin-bottom:1px;white-space:nowrap">{sy}</div>'
            f'<div style="display:flex;flex-direction:column;align-items:center;border:{bdr};border-radius:8px;background:{bg_card};padding:2px 1px;">'
            f'<div style="font-size:8px;color:#5a3e0a;margin-bottom:1px;white-space:nowrap">{six_g}</div>'
            f'<div style="width:26px;height:26px;border-radius:4px;background:{bg_g};color:{tc_g};display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:900;">{hj_sg}</div>'
            f'<div style="width:26px;height:26px;border-radius:4px;background:{bg_j};color:{tc_j};display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:900;margin-top:1px;">{hj_sj}</div>'
            f'<div style="font-size:8px;color:#5a3e0a;margin-top:1px;white-space:nowrap">{six_j}</div>'
            '</div>'
            f'<div style="font-size:7px;color:#6b5a3e;margin-top:1px;">{display_age}</div>'
            '</div>'
        )

    seun_html += '</div></div>'
    # 스크롤 위치 표시 바
    seun_html += '<div style="margin:2px 8px 0;height:3px;background:#ece8d8;border-radius:2px;position:relative;">'
    seun_html += '<div id="scroll-indicator" style="height:3px;background:#c8b87a;border-radius:2px;width:20%;position:absolute;left:0;transition:left 0.1s;"></div>'
    seun_html += '</div>'

    # JS: 대운 구간 시작점으로 자동 스크롤 + 스크롤 인디케이터 연동
    seun_html += '''<script>
    (function(){
        var el = document.getElementById('seun-range-start');
        var container = document.getElementById('seun-timeline');
        var indicator = document.getElementById('scroll-indicator');
        if(el && container){
            var offset = el.offsetLeft - container.offsetLeft - (container.clientWidth / 2) + (el.offsetWidth * 5);
            container.scrollLeft = Math.max(0, offset);
        }
        if(container && indicator){
            function updateIndicator(){
                var ratio = container.scrollLeft / (container.scrollWidth - container.clientWidth);
                var trackWidth = container.clientWidth - 16;
                var thumbWidth = Math.max(trackWidth * 0.2, 30);
                indicator.style.width = thumbWidth + 'px';
                indicator.style.left = (ratio * (trackWidth - thumbWidth)) + 'px';
            }
            container.addEventListener('scroll', updateIndicator);
            updateIndicator();
        }
    })();
    </script></body></html>'''

    import streamlit.components.v1 as components
    # 세운 타임라인을 대운에 밀착
    st.markdown('<style>div[data-testid="stVerticalBlock"]>div:has(iframe[height="106"]){margin-top:-20px!important;margin-bottom:-10px!important;}</style>', unsafe_allow_html=True)
    components.html(seun_html, height=106, scrolling=False)

    # ★ 아래: 현재 대운 구간 10개 나이 버튼 (월운 이동용)
    seun_range = []
    for age_i in range(seun_age_start, seun_age_end + 1):
        if age_i < len(seun):
            sy, sg, sj = seun[age_i]
            seun_range.append((age_i, sy, sg, sj))
    seun_range_disp = list(reversed(seun_range))

    n_btn = len(seun_range_disp)
    if n_btn > 0:
        cols_su = st.columns(n_btn)
        for ci, (age_i, sy, sg, sj) in enumerate(seun_range_disp):
            display_age = age_i + 1
            with cols_su[ci]:
                if st.button(f'{display_age}', key=f'su_{age_i}', use_container_width=True):
                    st.session_state.sel_seun = age_i
                    st.session_state.sel_wolun = 0
                    st.session_state.page = 'wolun'
                    st.rerun()

    # ★ 사용법 안내
    st.markdown(
        '<div style="text-align:center;font-size:11px;color:#9a8a6a;margin:2px 0 2px;line-height:1.5;">'
        '💡 <b>년도</b>버튼 → 세운 보기 · <b>나이</b>버튼 → 월운 보기 · 월운에서 <b>월</b>버튼 → 일운(달력) 보기'
        '</div>',
        unsafe_allow_html=True
    )

    gpt_url='https://chatgpt.com/g/g-68d90b2d8f448191b87fb7511fa8f80a-rua-myeongrisajusangdamsa'
    bottom_html = (
        '<div class="bottom-btns">'
        f'<a href="{gpt_url}" target="_blank" class="bottom-btn-ai">🤖 AI무료상담</a>'
        '<a href="https://www.youtube.com/@psycologysalon" target="_blank" class="bottom-btn-yt">🎥 명리심리 유튜브</a>'
        '<a href="https://open.kakao.com/o/sWJUYGDh" target="_blank" class="bottom-btn-chat">💬 오픈채팅방</a>'
        '</div>'
    )
    st.markdown(bottom_html, unsafe_allow_html=True)

    # ★ 내 사주 해석 보기 (expander)
    with st.expander("📊 내 사주 해석 보기", expanded=False):
        # ① 사령 / 당령 / 절입일
        saryeong_html = (
            '<div class="geok-box" style="margin-bottom:10px;">'
            '<div class="geok-why">'
            f'<b>사령</b>: {saryeong_gan}({saryeong_six}) · {saryeong_period} · {month_ji}월 절입+{day_from}일'
            f'<br><b>당령</b>: {dr_mission} · {dr_period}<br>{dr_desc}'
            f'<br><b>절입일</b>: 이전 {prev_str} / 이후 {next_str}'
            '</div>'
            '</div>'
        )
        st.markdown(saryeong_html, unsafe_allow_html=True)

        # ② 격 카드 상세 (강점/성장팁/칭찬)
        geok_card2 = find_geok_card(geok)
        if geok_card2:
            from datetime import date as _d
            user_age = _d.today().year - birth_year
            st.markdown(render_geok_card_html(geok_card2, show_detail=True, user_age=user_age), unsafe_allow_html=True)

        # ③ 법정시 / 보정값 상세 (표준시 라벨 포함)
        corr = data.get('corr_detail')
        eot = data.get('eot_min', 0)
        if corr:
            st.markdown('<div class="sec-title">🕐 시간 보정 상세</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div style="text-align:center;font-size:11px;color:#6b5a3e;margin-bottom:6px;padding:4px;background:#f5f0e0;border-radius:6px;">{calc_info}</div>',
                unsafe_allow_html=True
            )
            st.markdown(render_correction_html(corr, eot), unsafe_allow_html=True)

        # ③ 경계 경고 (해당될 때만)
        warns = data.get('boundary_warns', [])
        if warns:
            warn_html = '<div class="boundary-warn">' + '<br>'.join(warns) + '</div>'
            st.markdown(warn_html, unsafe_allow_html=True)

        # ④ 진태양시 비교 (정밀검증 모드 ON일 때만)
        if data.get('show_tst') and data.get('fp_wall'):
            dt_local = data.get('dt_local')
            dt_solar = data.get('dt_solar')
            fp_wall = data.get('fp_wall')
            if dt_local and dt_solar:
                st.markdown(render_tst_compare_html(dt_local, dt_solar, fp_wall, fp), unsafe_allow_html=True)

def page_wolun():
    data=st.session_state.saju_data
    if not data or 'fp' not in data: st.session_state.page='input'; st.rerun(); return
    now=datetime.now(LOCAL_TZ)
    ilgan=data['ilgan']
    seun=data["seun"]
    sel_su=st.session_state.sel_seun
    sy,sg,sj=seun[sel_su]
    if st.button('← 사주로'): st.session_state.page='saju'; st.rerun()
    hj_sg=hanja_gan(sg); hj_sj=hanja_ji(sj)
    display_age = sel_su + 1
    st.markdown(f'<div class="sel-info">{sy}년 {display_age}세 {hj_sg}{hj_sj} 월운 ({six_for_stem(ilgan,sg)}/{six_for_branch(ilgan,sj)})</div>', unsafe_allow_html=True)

    wolun=calc_wolun_accurate(sy)
    sel_wu=st.session_state.sel_wolun
    wolun_rev=list(reversed(wolun))
    MONTH_KR=['1월','2월','3월','4월','5월','6월','7월','8월','9월','10월','11월','12월']
    for row_start in [6,0]:
        row_items=wolun_rev[row_start:row_start+6]
        cols=st.columns(len(row_items))
        for ci,col in enumerate(cols):
            if ci>=len(row_items): break
            real_wu=11-(row_start+ci)
            wm=row_items[ci]["month"]
            wg=row_items[ci]["gan"]; wj=row_items[ci]["ji"]
            with col:
                active=(real_wu==sel_wu)
                bg_g=GAN_BG.get(wg,"#888"); tc_g=gan_fg(wg)
                bg_j=BR_BG.get(wj,"#888"); tc_j=br_fg(wj)
                hj_wg=hanja_gan(wg); hj_wj=hanja_ji(wj)
                bdr='2px solid #8b6914' if active else '1px solid #c8b87a'
                bg_card='#d4c48a' if active else '#e8e4d8'
                six_g=six_for_stem(ilgan,wg); six_j=six_for_branch(ilgan,wj)
                st.markdown(
                    f'<div style="text-align:center;font-size:10px;color:#6b5a3e;margin-bottom:1px">{MONTH_KR[wm-1]}</div>'
                    f'<div style="display:flex;flex-direction:column;align-items:center;border:{bdr};border-radius:10px;background:{bg_card};padding:2px 2px;">'
                    f'<div style="font-size:9px;color:#5a3e0a;margin-bottom:1px;white-space:nowrap">{six_g}</div>'
                    f'<div style="width:34px;height:34px;border-radius:6px;background:{bg_g};color:{tc_g};display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:900;margin-bottom:1px">{hj_wg}</div>'
                    f'<div style="width:34px;height:34px;border-radius:6px;background:{bg_j};color:{tc_j};display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:900;margin-bottom:1px">{hj_wj}</div>'
                    f'<div style="font-size:9px;color:#5a3e0a;white-space:nowrap">{six_j}</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
                if st.button(f'{wm}월',key=f'wu_{real_wu}',use_container_width=True):
                    st.session_state.sel_wolun=real_wu
                    st.session_state.page='ilun'
                    st.rerun()

    gpt_url='https://chatgpt.com/g/g-68d90b2d8f448191b87fb7511fa8f80a-rua-myeongrisajusangdamsa'
    bottom_html = (
        '<div class="bottom-btns">'
        f'<a href="{gpt_url}" target="_blank" class="bottom-btn-ai">🤖 AI무료상담</a>'
        '<a href="https://www.youtube.com/@psycologysalon" target="_blank" class="bottom-btn-yt">🎥 명리심리 유튜브</a>'
        '<a href="https://open.kakao.com/o/sWJUYGDh" target="_blank" class="bottom-btn-chat">💬 오픈채팅방</a>'
        '</div>'
    )
    st.markdown(bottom_html, unsafe_allow_html=True)

def page_ilun():
    data=st.session_state.saju_data
    if not data or 'fp' not in data: st.session_state.page='input'; st.rerun(); return
    now=datetime.now(LOCAL_TZ)
    longitude = data.get('longitude', DEFAULT_LONGITUDE)
    apply_solar = data.get('apply_solar', True)
    ilgan=data['ilgan']
    seun=data["seun"]
    sel_su=st.session_state.sel_seun
    sy,sg,sj=seun[sel_su]
    sel_wu=st.session_state.sel_wolun
    wolun=calc_wolun_accurate(sy)
    wm_data=wolun[sel_wu]
    wm=wm_data["month"]; wg=wm_data["gan"]; wj=wm_data["ji"]
    if st.button('← 월운으로'): st.session_state.page='wolun'; st.rerun()
    hj_wg=hanja_gan(wg); hj_wj=hanja_ji(wj)
    hj_sg=hanja_gan(sg); hj_sj=hanja_ji(sj)
    display_age = sel_su + 1
    st.markdown(f'<div class="sel-info">{sy}년({display_age}세) {wm}월 ({hj_wg}{hj_wj}) 일운</div>', unsafe_allow_html=True)

    _,days_in_month=cal_mod.monthrange(sy,wm)
    first_weekday,_=cal_mod.monthrange(sy,wm)
    first_wd=(first_weekday+1)%7
    # 이 달의 절기 계산
    jie24_this = compute_jie24_times_calc(sy)

    if apply_solar:
        for k in jie24_this:
            jie24_this[k] = to_solar_time(jie24_this[k], longitude)

    jie24_solar_ilun = jie24_this
    # 이 달의 절기 목록 (날짜 -> 절기명,시각)
    month_jie_map={}
    for jname,jt in jie24_solar_ilun.items():
        if jt.year==sy and jt.month==wm:
            month_jie_map[jt.day]=(jname,jt)
    # 이 달의 절기 2개 텍스트 (상단 표시용)
    month_terms_list=sorted(month_jie_map.items())
    month_terms_str=' / '.join([f"{v[0]} ({v[1].strftime('%d일 %H:%M')})" for k,v in month_terms_list])
    # 음력 변환
    def solar_to_lunar_str(y,m,d):
        if not HAS_LUNAR: return ''
        try:
            c=KoreanLunarCalendar()
            c.setSolarDate(y,m,d)
            lm=c.lunarMonth; ld=c.lunarDay; is_l=c.isIntercalation
            leap_str='윤' if is_l else ''
            return f'{leap_str}{lm}/{ld}'
        except: return ''
    day_items=[]
    for d in range(1, days_in_month+1):
        dt_local=datetime(sy,wm,d,12,0,tzinfo=LOCAL_TZ)

        if apply_solar:
            dt_solar = to_solar_time(dt_local, longitude)
        else:
            dt_solar = dt_local
        dj,dc,djidx=day_ganji_solar(dt_solar)
        g,j=dj[0],dj[1]
        sg_six=six_for_stem(ilgan,g); sj_six=six_for_branch(ilgan,j)
        lunar_str=solar_to_lunar_str(sy,wm,d)
        jie_info=month_jie_map.get(d,None)
        jie_str=jie_info[0] if jie_info else ''
        day_items.append({'day':d,'gan':g,'ji':j,'sg_six':sg_six,'sj_six':sj_six,'lunar':lunar_str,'jie':jie_str})

    html='<div class="cal-wrap">'
    html+=f'<div class="cal-header">{sy}년({hj_sg}{hj_sj}) {wm}월({hj_wg}{hj_wj})</div>'
    if month_terms_str:
        html+=f'<div style="background:#f5eed8;padding:4px 8px;font-size:11px;color:#7a5a1a;text-align:center;border-bottom:1px solid #c8b87a;">🌿 절기: {month_terms_str}</div>'
    html+='<table class="cal-table"><thead><tr>'
    for dn in ['일','월','화','수','목','금','토']: html+=f'<th>{dn}</th>'
    html+='</tr></thead><tbody><tr>'
    for _ in range(first_wd): html+='<td class="empty"></td>'
    col_pos=first_wd
    for item in day_items:
        if col_pos==7: html+='</tr><tr>'; col_pos=0
        d_num=item["day"]; dow=(first_wd+d_num-1)%7
        is_today=(sy==now.year and wm==now.month and d_num==now.day)
        cls='today-cell' if is_today else ''
        if dow==0: cls+=' sun'
        elif dow==6: cls+=' sat'
        hj_dg=hanja_gan(item["gan"]); hj_dj=hanja_ji(item["ji"])
        sg6=item["sg_six"]; sj6=item["sj_six"]
        lunar6=item.get("lunar",""); jie6=item.get("jie","")
        jie_html=f'<div style="font-size:8px;color:#b06000;font-weight:bold;">{jie6}</div>' if jie6 else ''
        lunar_html=f'<div style="font-size:8px;color:#5a5a8a;">{lunar6}</div>' if lunar6 else ''
        html+=f'<td class="{cls.strip()}">{jie_html}<div class="dn">{d_num}</div>{lunar_html}<div style="font-size:9px;color:#888;">{sg6}</div><div style="font-size:14px;font-weight:bold;">{hj_dg}</div><div style="font-size:14px;font-weight:bold;">{hj_dj}</div><div style="font-size:9px;color:#888;">{sj6}</div></td>'
        col_pos+=1
    while col_pos%7!=0 and col_pos>0: html+='<td class="empty"></td>'; col_pos+=1
    html+='</tr></tbody></table></div>'
    st.markdown(html,unsafe_allow_html=True)

    gpt_url='https://chatgpt.com/g/g-68d90b2d8f448191b87fb7511fa8f80a-rua-myeongrisajusangdamsa'
    bottom_html = (
        '<div class="bottom-btns">'
        f'<a href="{gpt_url}" target="_blank" class="bottom-btn-ai">🤖 AI무료상담</a>'
        '<a href="https://www.youtube.com/@psycologysalon" target="_blank" class="bottom-btn-yt">🎥 명리심리 유튜브</a>'
        '<a href="https://open.kakao.com/o/sWJUYGDh" target="_blank" class="bottom-btn-chat">💬 오픈채팅방</a>'
        '</div>'
    )
    st.markdown(bottom_html, unsafe_allow_html=True)

if __name__=='__main__':
    main()
