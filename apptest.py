"""
MAG 7+2 공매도 분석 - Streamlit App
Magnificent Seven + Bitcoin 공매도 분석
- 로그인 시스템
- Gemini/OpenAI AI 분석 (기본/Deep Dive)
- Advanced Quant Chatbot
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from io import StringIO
import time
import json

# 페이지 설정
st.set_page_config(
    page_title="MAG 7+2 Quant Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 세션 상태 초기화 ====================
if 'password_correct' not in st.session_state:
    st.session_state['password_correct'] = False
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'analysis_data' not in st.session_state:
    st.session_state['analysis_data'] = None
    
# ==================== 로그인 시스템 ====================
def check_password():
    """비밀번호 확인 및 로그인 상태 관리"""
    if st.session_state.get('password_correct', False):
        return True
    
    st.title("🔒 MAG 7+2 공매도분석")
    st.markdown("### MAG 7+2 공매도 분석")
    
    with st.form("credentials"):
        username = st.text_input("아이디 (ID)", key="username")
        password = st.text_input("비밀번호 (Password)", type="password", key="password")
        submit_btn = st.form_submit_button("로그인", type="primary")
    
    if submit_btn:
        if username in st.secrets["passwords"] and password == st.secrets["passwords"][username]:
            st.session_state['password_correct'] = True
            st.rerun()
        else:
            st.error("😕 아이디 또는 비밀번호가 올바르지 않습니다.")
    
    return False

if not check_password():
    st.stop()


# ==================== MAG 7+2 정의 ====================
MAG7_STOCKS = {
    'AAPL': {
        'name': 'Apple Inc.',
        'description': '아이폰, 생태계, 온디바이스 AI',
        'sector': 'Technology',
        'industry': 'Consumer Electronics'
    },
    'MSFT': {
        'name': 'Microsoft Corporation',
        'description': '클라우드(Azure), 생성형 AI (OpenAI 대주주)',
        'sector': 'Technology',
        'industry': 'Software'
    },
    'GOOGL': {
        'name': 'Alphabet Inc.',
        'description': '구글 검색, 유튜브, AI (Gemini)',
        'sector': 'Communication Services',
        'industry': 'Internet Content & Information'
    },
    'AMZN': {
        'name': 'Amazon.com Inc.',
        'description': '전자상거래, 클라우드(AWS) 1위',
        'sector': 'Consumer Cyclical',
        'industry': 'Internet Retail'
    },
    'NVDA': {
        'name': 'NVIDIA Corporation',
        'description': 'AI 반도체(GPU) 독점적 지배자',
        'sector': 'Technology',
        'industry': 'Semiconductors'
    },
    'META': {
        'name': 'Meta Platforms Inc.',
        'description': '페이스북, 인스타그램, AI(Llama)',
        'sector': 'Communication Services',
        'industry': 'Internet Content & Information'
    },
    'TSLA': {
        'name': 'Tesla Inc.',
        'description': '전기차, 자율주행, 로봇',
        'sector': 'Consumer Cyclical',
        'industry': 'Auto Manufacturers'
    },
    'COIN': {
        'name': 'Coinbase Global Inc.',
        'description': '미국 최대 암호화폐 거래소, 비트코인 직접 노출',
        'sector': 'Financial Services',
        'industry': 'Cryptocurrency Exchange'
    },
    'IBIT': {
        'name': 'iShares Bitcoin Trust ETF',
        'description': 'BlackRock 비트코인 현물 ETF, 순수 BTC 노출',
        'sector': 'ETF',
        'industry': 'Bitcoin Spot ETF'
    }
}

# ==================== 데이터 수집 함수 ====================
@st.cache_data(ttl=3600)
def get_current_quarter_start():
    """현재 분기 시작일 계산"""
    return datetime(2025, 1, 1)

@st.cache_data(ttl=3600)
def calculate_anchored_vwap(df):
    """Anchored VWAP 계산"""
    df = df.copy()
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['TP_Volume'] = df['Typical_Price'] * df['Volume']
    df['Cumulative_TP_Volume'] = df['TP_Volume'].cumsum()
    df['Cumulative_Volume'] = df['Volume'].cumsum()
    df['Anchored_VWAP'] = df['Cumulative_TP_Volume'] / df['Cumulative_Volume']
    return df

@st.cache_data(ttl=3600)
def get_quarterly_vwap_analysis(ticker):
    """분기별 Anchored VWAP 분석"""
    try:
        quarter_start = get_current_quarter_start()
        end_date = datetime.now()
        quarter_num = (quarter_start.month - 1) // 3 + 1

        stock = yf.Ticker(ticker)
        
        # history 호출 시 예외 처리 강화
        try:
            df = stock.history(start=quarter_start, end=end_date)
        except Exception:
            return None

        if df.empty or len(df) < 5:
            return None
         
        df = calculate_anchored_vwap(df)

        current_price = df['Close'].iloc[-1]
        current_vwap = df['Anchored_VWAP'].iloc[-1]
        above_vwap_ratio = (df['Close'] > df['Anchored_VWAP']).sum() / len(df) * 100
        recent_5days_avg = df['Close'].tail(5).mean()
        recent_10days_avg = df['Close'].tail(10).mean()

        recent_20 = df['Close'].tail(min(20, len(df)))
        uptrend_strength = (recent_20.diff() > 0).sum() / len(recent_20) * 100 if len(recent_20) > 1 else 50

        recent_volume = df['Volume'].tail(5).mean()
        avg_volume = df['Volume'].mean()
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1

        info = stock.info
        quarter_start_price = df['Close'].iloc[0]
        quarter_return = ((current_price - quarter_start_price) / quarter_start_price * 100)

        return {
            'Ticker': ticker,
            'Company': MAG7_STOCKS[ticker]['name'],
            'Description': MAG7_STOCKS[ticker]['description'],
            'Sector': MAG7_STOCKS[ticker]['sector'],
            'Industry': MAG7_STOCKS[ticker]['industry'],  # [추가] Industry 필드 추가
            'Quarter': f'{quarter_start.year} Q{quarter_num}',
            'Quarter_Start_Date': quarter_start.strftime('%Y-%m-%d'),
            'Trading_Days': len(df),
            'Current_Price': round(current_price, 2),
            'Anchored_VWAP': round(current_vwap, 2),
            'Quarter_Start_Price': round(quarter_start_price, 2),
            'Quarter_Return_%': round(quarter_return, 2),
            'Price_vs_VWAP_%': round((current_price - current_vwap) / current_vwap * 100, 2),
            'Above_VWAP_Days_%': round(above_vwap_ratio, 1),
            'Recent_5D_Avg': round(recent_5days_avg, 2),
            'Recent_10D_Avg': round(recent_10days_avg, 2),
            'Uptrend_Strength_%': round(uptrend_strength, 1),
            'Volume_Ratio': round(volume_ratio, 2),
            'Is_Above_VWAP': current_price > current_vwap,
            'Market_Cap': info.get('marketCap', 0),
            'Buy_Signal_Score': 0
        }

    except Exception as e:
        st.error(f"Error processing {ticker}: {str(e)}")
        return None

def calculate_buy_score(row):
    """매수 신호 점수 계산"""
    score = 0
    if row['Is_Above_VWAP']:
        score += 30

    price_diff = row['Price_vs_VWAP_%']
    if 0 < price_diff <= 5:
        score += 20
    elif 5 < price_diff <= 10:
        score += 10
    elif price_diff > 10:
        score += 5

    if row['Above_VWAP_Days_%'] >= 80:
        score += 20
    elif row['Above_VWAP_Days_%'] >= 60:
        score += 15
    elif row['Above_VWAP_Days_%'] >= 40:
        score += 10

    if row['Uptrend_Strength_%'] >= 60:
        score += 15
    elif row['Uptrend_Strength_%'] >= 50:
        score += 10

    if row['Volume_Ratio'] >= 1.2:
        score += 15
    elif row['Volume_Ratio'] >= 1.0:
        score += 10

    return min(score, 100)

@st.cache_data(ttl=3600)
def get_finra_short_volume_csv(ticker, days_back=10):
    """FINRA에서 일별 공매도 거래량 CSV 파일 다운로드"""
    try:
        today = datetime.now()
        short_volume_data = []

        for days in range(days_back):
            check_date = today - timedelta(days=days)
            if check_date.weekday() >= 5:
                continue

            date_str = check_date.strftime('%Y%m%d')
            url = f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date_str}.txt"

            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    df = pd.read_csv(StringIO(response.text), sep='|')
                    
                    if 'Symbol' in df.columns or 'symbol' in df.columns:
                        df.columns = df.columns.str.strip()
                        symbol_col = 'Symbol' if 'Symbol' in df.columns else 'symbol'
                        ticker_data = df[df[symbol_col].str.upper() == ticker.upper()]

                        if not ticker_data.empty:
                            row = ticker_data.iloc[0]
                            short_vol = row.get('ShortVolume', row.get('shortVolume', 0))
                            total_vol = row.get('TotalVolume', row.get('totalVolume', 0))

                            if pd.notna(short_vol) and pd.notna(total_vol) and total_vol > 0:
                                short_volume_data.append({
                                    'date': check_date.strftime('%Y-%m-%d'),
                                    'short_volume': int(short_vol),
                                    'total_volume': int(total_vol),
                                    'short_ratio': round(short_vol / total_vol * 100, 2)
                                })
            except:
                continue

        if short_volume_data:
            df_short = pd.DataFrame(short_volume_data)
            avg_short_ratio = df_short['short_ratio'].mean()
            latest_short_ratio = df_short.iloc[0]['short_ratio'] if len(df_short) > 0 else 0

            return {
                'ticker': ticker,
                'latest_date': df_short.iloc[0]['date'] if len(df_short) > 0 else 'N/A',
                'latest_short_ratio': latest_short_ratio,
                'avg_short_ratio_10d': round(avg_short_ratio, 2),
                'data_points': len(df_short),
                'historical_data': df_short
            }
        return None
    except:
        return None


@st.cache_data(ttl=3600)
def get_short_interest_from_yfinance(ticker):
    """Yahoo Finance에서 공매도 데이터 수집"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        short_data = {
            'ticker': ticker,
            'short_ratio': info.get('shortRatio', 0),
            'short_percent_float': info.get('shortPercentOfFloat', 0) * 100 if info.get('shortPercentOfFloat') else 0,
            'shares_short': info.get('sharesShort', 0),
            'shares_short_prior_month': info.get('sharesShortPriorMonth', 0),
            'short_percent_shares_out': info.get('sharesPercentSharesOut', 0) * 100 if info.get('sharesPercentSharesOut') else 0
        }

        if short_data['shares_short_prior_month'] > 0:
            short_data['short_change_pct'] = ((short_data['shares_short'] - short_data['shares_short_prior_month']) /
                                               short_data['shares_short_prior_month'] * 100)
        else:
            short_data['short_change_pct'] = 0

        return short_data

    except Exception as e:
        return None


@st.cache_data(ttl=3600)
def get_comprehensive_short_data(ticker):
    """여러 소스에서 공매도 데이터 종합 수집"""
    yf_data = get_short_interest_from_yfinance(ticker)
    finra_data = get_finra_short_volume_csv(ticker, days_back=10)

    combined_data = {
        'ticker': ticker,
        'short_ratio_days': 0,
        'short_percent_float': 0,
        'shares_short_millions': 0,
        'short_change_pct': 0,
        'daily_short_ratio': 0,
        'avg_daily_short_ratio_10d': 0,
        'finra_latest_date': 'N/A',
        'data_source': [],
        'finra_historical': None
    }

    if yf_data:
        combined_data.update({
            'short_ratio_days': round(yf_data.get('short_ratio', 0), 2),
            'short_percent_float': round(yf_data.get('short_percent_float', 0), 2),
            'shares_short_millions': round(yf_data.get('shares_short', 0) / 1e6, 2),
            'short_change_pct': round(yf_data.get('short_change_pct', 0), 2),
        })
        combined_data['data_source'].append('Yahoo Finance')

    if finra_data:
        if 'latest_short_ratio' in finra_data:
            combined_data['daily_short_ratio'] = finra_data['latest_short_ratio']
            combined_data['avg_daily_short_ratio_10d'] = finra_data['avg_short_ratio_10d']
            combined_data['finra_latest_date'] = finra_data.get('latest_date', 'N/A')
            combined_data['finra_historical'] = finra_data.get('historical_data')
            combined_data['data_source'].append(f"FINRA ({finra_data.get('data_points', 0)}일)")

    combined_data['data_source'] = ' + '.join(combined_data['data_source']) if combined_data['data_source'] else 'N/A'
    return combined_data

@st.cache_data(ttl=3600)
def collect_all_data():
    """모든 데이터 수집 (VWAP + Yahoo Finance + FINRA)"""
    mag7_tickers = list(MAG7_STOCKS.keys())
    
    results = []
    short_data_list = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, ticker in enumerate(mag7_tickers):
        status_text.text(f"분석 중: {ticker} ({MAG7_STOCKS[ticker]['name']})...")

        # [수정 1] 요청 간 딜레이 추가 (Yahoo Finance 차단 방지)
        time.sleep(1.5)  # 1.5초 대기
        
        # VWAP 분석
        result = get_quarterly_vwap_analysis(ticker)
        if result:
            results.append(result)
        
        # 공매도 데이터 (Yahoo Finance + FINRA 통합)
        # [수정 2] 연속 호출 방지를 위해 여기도 딜레이
        time.sleep(0.5)
        short_data = get_comprehensive_short_data(ticker)
        if short_data:
            short_data_list.append(short_data)
        
        progress_bar.progress((idx + 1) / len(mag7_tickers))
    
    status_text.empty()
    progress_bar.empty()
    
    df_results = pd.DataFrame(results)
    # [수정 3] 데이터가 하나도 없을 경우 에러 처리 (빈 데이터프레임 오류 방지)
    if df_results.empty:
        st.error("❌ 데이터 수집에 실패했습니다. (Too Many Requests). 잠시 후 다시 시도하거나, 로컬 환경에서 실행해 보세요.")
        return pd.DataFrame() # 빈 데이터프레임 반환하여 앱 충돌 방지
    df_short = pd.DataFrame(short_data_list)
    
    # 매수 신호 점수 계산
    df_results['Buy_Signal_Score'] = df_results.apply(calculate_buy_score, axis=1)
    
    df_results['Market_Cap_Trillion'] = (df_results['Market_Cap'] / 1e12).round(3)
    
    # 공매도 데이터 병합
    df_results = df_results.merge(df_short, left_on='Ticker', right_on='ticker', how='left')
    
    # 공매도 스코어
    def calculate_short_score(row):
        short_pct = row.get('short_percent_float', 0)
        if short_pct < 5:
            return 20
        elif short_pct < 10:
            return 15
        elif short_pct < 20:
            return 10
        else:
            return 5
    
    df_results['Short_Score'] = df_results.apply(calculate_short_score, axis=1)
    df_results['Total_Investment_Score'] = df_results['Buy_Signal_Score'] + df_results['Short_Score']
    df_results = df_results.sort_values('Total_Investment_Score', ascending=False)
    
    return df_results

# ==================== AI 분석 함수 ====================
def analyze_with_gemini(df_results, analysis_type="basic"):
    """Gemini로 AI 분석 (실제 API 사용)"""
    try:
        import google.generativeai as genai
        
        # API 키 설정
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # 프롬프트 생성
        prompt = create_gemini_prompt(df_results, analysis_type)
        
        # API 호출
        with st.spinner("🤖 Gemini AI 분석 중..."):
            response = model.generate_content(prompt)
            return response.text
            
    except Exception as e:
        st.error(f"⚠️ Gemini API 오류: {str(e)}")
        st.info("💡 API 키를 확인하거나 데모 모드로 전환합니다.")
        # 폴백: 데모 모드
        return analyze_with_gemini_demo(df_results, analysis_type)


def create_gemini_prompt(df_results, analysis_type):
    """Gemini용 프롬프트 생성"""
    
    # 데이터 요약 - 더 상세하게
    current_date = datetime.now().strftime('%Y년 %m월 %d일')
    quarter_start = get_current_quarter_start()
    quarter_num = (quarter_start.month - 1) // 3 + 1
    
    # 종목별 상세 데이터 (Top 9)
    detailed_data = []
    for idx, row in df_results.head(9).iterrows():
        # [수정] Industry를 MAG7_STOCKS에서 가져오기
        ticker = row['Ticker']
        industry = MAG7_STOCKS.get(ticker, {}).get('industry', 'N/A')

        
        detailed_data.append(f"""
**{row['Ticker']} - {row['Company']}**
- 섹터/산업: {row['Sector']} / {row['Industry']}
- 시가총액: ${row['Market_Cap_Trillion']:.2f}T
- 현재가: ${row['Current_Price']:.2f}
- Anchored VWAP: ${row['Anchored_VWAP']:.2f}
- VWAP 대비: {row['Price_vs_VWAP_%']:+.2f}%
- {quarter_start.year} Q{quarter_num} 수익률: {row['Quarter_Return_%']:+.2f}%
- VWAP 위 거래일: {row['Above_VWAP_Days_%']:.1f}%
- 상승 강도: {row['Uptrend_Strength_%']:.1f}%
- 거래량 비율: {row['Volume_Ratio']:.2f}x
- 공매도 비율(Float): {row['short_percent_float']:.2f}%
- 공매도 청산일: {row['short_ratio_days']:.1f}일
- 공매도 변화(MoM): {row['short_change_pct']:+.1f}%
- FINRA 일일 공매도: {row['daily_short_ratio']:.1f}%
- 기술적 점수: {row['Buy_Signal_Score']}/100
- 종합 투자 점수: {row['Total_Investment_Score']}/120
- 주요 특징: {row['Description']}
""")
    
    data_summary = f"""
===========================================
📊 MAG 7+2 종합 분석 리포트
===========================================
**분석 기준일**: {current_date}
**분석 기간**: {quarter_start.year} Q{quarter_num} ({quarter_start.strftime('%Y-%m-%d')} ~)
**분석 대상**: {len(df_results)}개 종목

-------------------------------------------
📈 포트폴리오 전체 통계
-------------------------------------------
- 평균 투자 점수: {df_results['Total_Investment_Score'].mean():.1f}/120
- VWAP 위 거래 종목: {df_results['Is_Above_VWAP'].sum()}개 ({df_results['Is_Above_VWAP'].sum()/len(df_results)*100:.1f}%)
- 평균 분기 수익률: {df_results['Quarter_Return_%'].mean():+.2f}%
- 평균 공매도 비율: {df_results['short_percent_float'].mean():.2f}%
- 공매도 5% 미만 종목: {len(df_results[df_results['short_percent_float'] < 5])}개
- 공매도 10% 이상 종목: {len(df_results[df_results['short_percent_float'] >= 10])}개

-------------------------------------------
🏆 상위 5개 종목 (투자 점수 기준)
-------------------------------------------
{chr(10).join([f"{idx+1}. {row['Ticker']}: {row['Total_Investment_Score']}/120점" for idx, row in df_results.head(5).iterrows()])}

-------------------------------------------
📊 종목별 상세 데이터
-------------------------------------------
{chr(10).join(detailed_data)}

-------------------------------------------
📉 공매도 분석
-------------------------------------------
- 최고 공매도 비율: {df_results['short_percent_float'].max():.2f}% ({df_results.loc[df_results['short_percent_float'].idxmax(), 'Ticker']})
- 최저 공매도 비율: {df_results['short_percent_float'].min():.2f}% ({df_results.loc[df_results['short_percent_float'].idxmin(), 'Ticker']})
- 공매도 증가 종목: {len(df_results[df_results['short_change_pct'] > 0])}개
- 공매도 감소 종목: {len(df_results[df_results['short_change_pct'] < 0])}개
"""
    
    if analysis_type == "basic":
        prompt = f"""
당신은 월스트리트 Top Tier 투자은행의 수석 퀀트 애널리스트입니다. 
20년 이상의 경력으로 MAG 7 기술주와 암호화폐 시장을 전문적으로 분석해왔습니다.

다음 데이터를 바탕으로 **간결하면서도 핵심적인** 투자 인사이트를 제공하세요.

{data_summary}

-------------------------------------------
📋 요구사항
-------------------------------------------
다음 형식으로 분석을 제공하세요:

### 🤖 Gemini AI 기본 분석

**🌍 종합 시장 전망**
(2-3문장으로 현재 MAG 7+2 포트폴리오의 전반적인 상황과 시장 심리를 요약)

**🏆 Top Pick 추천**
(1위 종목에 대한 구체적인 추천 이유, Entry Price, Target Price 포함)

**⚠️ 주의 종목**
(공매도 비율이 높거나 기술적으로 약한 종목 1-2개와 그 이유)

**💡 투자 전략**
(현재 시점에서의 구체적인 행동 지침 - 3-4문장)

-------------------------------------------
✅ 작성 가이드
-------------------------------------------
- 한국어로 작성
- 전문적이면서도 이해하기 쉽게
- 구체적인 가격과 수치 포함
- 불필요한 인사말이나 서론 없이 바로 본론으로
"""
    
    else:  # deep dive
        prompt = f"""
당신은 Bridgewater Associates, Renaissance Technologies, Citadel 출신의 전설적인 퀀트 애널리스트입니다. 
25년간 기술주 포트폴리오 관리와 알고리즘 트레이딩 전략 개발을 해왔으며, 
MAG 7 종목들과 비트코인 관련 자산에 대한 깊은 통찰력을 보유하고 있습니다.

현재는 글로벌 헤지펀드의 Chief Investment Officer로서 
$50B AUM의 기술주 롱숏 전략을 총괄하고 있습니다.

다음 데이터를 바탕으로 **기관투자자 수준의 심층 분석 리포트**를 작성하세요.

{data_summary}

-------------------------------------------
📋 분석 프레임워크
-------------------------------------------
다음 형식으로 **매우 상세하고 전문적인** 분석을 제공하세요:

### 🔬 Gemini AI Deep Dive 분석
*MAG 7+2 Portfolio - Institutional Grade Research Report*

---

## 1️⃣ 거시경제 및 시장 환경 분석

**현재 시장 컨텍스트:**
- 연준 금리 정책과 빅테크 밸류에이션에 미치는 영향
- AI 투자 사이클의 현재 단계 (Early/Mid/Late Stage)
- 비트코인 및 암호화폐 시장과의 상관관계
- 2025년 {quarter_start.year} Q{quarter_num} 주요 모멘텀과 리스크 요인

**섹터별 트렌드:**
- Technology (반도체/소프트웨어/하드웨어)
- Communication Services (소셜미디어/검색)
- Consumer Cyclical (전기차/이커머스)
- Crypto Exposure (COIN, IBIT)

---

## 2️⃣ Top 5 종목 심층 분석

각 종목에 대해 다음 구조로 분석:

### 1위. [TICKER] - [회사명]

**📊 기술적 분석 (Technical Deep Dive)**
- Anchored VWAP 분석: 현재 위치와 의미
- 분기 트렌드 강도: {row['Uptrend_Strength_%']}% → 해석
- 거래량 패턴: {row['Volume_Ratio']}x → 매집/분산 판단
- 가격 모멘텀: 단기(5일)/중기(10일) 이동평균 관계
- 지지선/저항선 레벨 설정

**🔴 공매도 상황 종합 평가**
- Float 대비: {row['short_percent_float']}% (업계 평균 대비)
- Days to Cover: {row['short_ratio_days']}일 → Short Squeeze 가능성
- MoM 변화: {row['short_change_pct']}% → 트렌드 방향
- FINRA 일일 데이터: {row['daily_short_ratio']}% → 단기 압력
- 베어/불 세력 균형 평가

**💰 Entry/Target/Stop Loss 전략**
- Entry Zone: $XXX - $XXX (구체적 근거)
- 1차 Target: $XXX (+X%)
- 2차 Target: $XXX (+X%)
- 최종 Target: $XXX (+X%)
- Stop Loss: $XXX (-X%, VWAP 또는 주요 지지선 기준)
- Risk/Reward Ratio: X:1

**🎯 투자 의견 및 포지션 사이징**
- 추천: BUY / ACCUMULATE / HOLD / REDUCE / SELL
- 신뢰도: High / Medium / Low
- 권장 비중: X% of portfolio
- 시간 프레임: 단기(1-2주) / 중기(1-3개월) / 장기(6개월+)
- 핵심 촉매: 실적 발표, 제품 출시, 정책 변화 등

**🔮 시나리오 분석**
- Bull Case (확률 X%): 목표가 $XXX
- Base Case (확률 X%): 목표가 $XXX  
- Bear Case (확률 X%): 목표가 $XXX

*[나머지 4개 종목도 동일한 구조로 분석]*

---

## 3️⃣ 포트폴리오 구성 전략

### 🔥 공격적 포트폴리오 (Target: +30%+ / Risk: High)
**목표**: 단기 알파 극대화, 높은 변동성 수용

**구성**:
- [TICKER1]: 35% - 이유와 기대 수익률
- [TICKER2]: 30% - 이유와 기대 수익률
- [TICKER3]: 20% - 이유와 기대 수익률
- Cash: 15% - 기회 포착용

**리밸런싱**: 주 1회
**예상 Sharpe Ratio**: X.XX
**최대 손실 예상**: -XX%
**적합 투자자**: 고위험 감수, 단기 트레이더

### ⚖️ 균형 포트폴리오 (Target: +15-20% / Risk: Medium)
**목표**: 위험 조정 수익 최적화

**구성**:
- [TICKER1]: 20%
- [TICKER2]: 20%
- [TICKER3]: 15%
- [TICKER4]: 15%
- [TICKER5]: 15%
- Cash: 15%

**리밸런싱**: 월 1회
**예상 Sharpe Ratio**: X.XX
**최대 손실 예상**: -XX%
**적합 투자자**: 성장 + 안정성 추구

### 🛡️ 보수적 포트폴리오 (Target: +8-12% / Risk: Low)
**목표**: 자본 보존 우선, 안정적 수익

**구성**:
- [안전 종목들 - 공매도 <3%]: 각 15-20%
- Cash: 40% - 조정 시 매수 대기

**리밸런싱**: 분기 1회
**예상 Sharpe Ratio**: X.XX
**최대 손실 예상**: -XX%
**적합 투자자**: 위험 회피, 장기 투자자

---

## 4️⃣ 매매 시그널 및 타이밍

### 🟢 즉시 매수 (Immediate Buy - Strong Conviction)
**[TICKER1]**: 
- Entry: $XXX
- Target: $XXX (1차), $XXX (2차)
- Stop: $XXX
- 근거: (3-4문장으로 구체적인 이유)
- 비중: X%

**[TICKER2]**: 
- [동일 구조]

### 🟡 조정 시 매수 (Buy on Dip - Conditional)
**[TICKER3]**: 
- 현재가: $XXX
- 대기 매수가: $XXX (VWAP / 지지선 기준)
- Target: $XXX
- Stop: $XXX
- 근거: (좋은 종목이지만 현재 과매수 등)
- 비중: X%

### 🔴 회피/청산 (Avoid / Reduce)
**[TICKER X]**: 
- 이유: (공매도 과다, 기술적 약세, 펀더멘털 악화 등)
- 대안: (더 나은 선택지)

---

## 5️⃣ 리스크 관리 프레임워크

**포지션 관리 규칙**:
1. **단일 종목 최대 비중**: 30% (공격적) / 20% (균형) / 15% (보수적)
2. **섹터 집중도 한도**: Technology 최대 60%, Crypto Exposure 최대 25%
3. **손절 원칙**: 
   - VWAP 이탈 시 즉시 검토
   - -5% 손실 시 포지션 50% 축소
   - -8% 손실 시 전량 청산
4. **이익 실현**: 
   - +10% 달성 시 50% 익절
   - +20% 달성 시 추가 30% 익절
   - 나머지는 Trailing Stop 적용

**시장 리스크 대응**:
- VIX 30 초과 시: 현금 비중 40%로 증가
- 급락장 (-5% in a day): 단계적 매수 (30% → 30% → 40%)
- 급등장 (+5% in a day): 차익 실현 고려

**공매도 모니터링**:
- Short % of Float 10% 초과 종목: 주간 체크
- Days to Cover 3일 초과: Short Squeeze 경계
- FINRA 데이터 50% 초과: 단기 약세 압력 주의

**포트폴리오 헤징 전략**:
- Beta 조정: S&P 500 대비 포트폴리오 Beta XX
- VIX 콜옵션: 극단적 변동성 대비
- Put Spread: 주요 보유 종목 하방 보호

---

## 6️⃣ 최종 권장사항 및 Action Plan

### 📅 이번 주 (Week of {current_date})
**즉시 실행**:
1. [TICKER]: $XXX에 X% 비중 매수
2. [TICKER]: 조정 시 $XXX 대기 매수 주문
3. [TICKER]: 현재 보유분 일부 익절 (과매수 구간)

**모니터링**:
- [이벤트]: 실적 발표 (X월 X일)
- [지표]: FINRA 공매도 데이터 일일 체크
- [가격]: VWAP 지지/저항 레벨

### 📊 이번 달 (This Month)
**포트폴리오 리밸런싱**:
- 목표 구성: (구체적 비중)
- 조정 방향: (매수/매도 종목)

**이벤트 캘린더**:
- X월 X일: [회사] 실적 발표
- X월 X일: FOMC 회의
- X월 X일: 옵션 만기

### 🎯 분기 전략 ({quarter_start.year} Q{quarter_num})
**목표 수익률**: +XX% (포트폴리오 타입별)
**핵심 테마**: AI 인프라 / 클라우드 성장 / 비트코인 ETF 수요
**주요 리스크**: 금리 변동성 / 빅테크 규제 / 지정학적 긴장

**Success Metrics**:
- Sharpe Ratio > X.XX
- Max Drawdown < XX%
- Win Rate > XX%

---

### 🔔 다음 리포트 업데이트
**예정일**: {(datetime.now() + timedelta(days=7)).strftime('%Y년 %m월 %d일')}
**포함 내용**: 주간 성과 리뷰, 포지션 조정, 신규 시그널

---

**⚠️ 면책조항**
본 분석은 정보 제공 목적이며, 투자 권유가 아닙니다. 
모든 투자 결정은 본인의 판단과 책임 하에 이루어져야 합니다.
과거 수익률이 미래 수익을 보장하지 않습니다.

---
*Report Generated by Gemini 2.5 Flash AI*
*Powered by Advanced Quantitative Analysis Engine*

-------------------------------------------
✅ 작성 가이드라인
-------------------------------------------
1. **전문성**: 기관투자자 수준의 용어와 분석 깊이
2. **구체성**: 모든 추천에 구체적인 가격과 근거 포함
3. **실행 가능성**: 실제로 따라할 수 있는 명확한 액션 플랜
4. **정량화**: 가능한 모든 것을 수치로 표현
5. **시나리오**: Bull/Base/Bear 케이스 제시
6. **리스크**: 각 전략의 위험 요소 명시
7. **한국어**: 전문 용어는 영어 병기, 자연스러운 한국어

**반드시 위 데이터의 실제 수치를 활용하여 작성하세요.**
**추상적이거나 일반적인 분석이 아닌, 이 포트폴리오에 특화된 구체적인 인사이트를 제공하세요.**
"""
    
    return prompt




def analyze_with_openai(df_results, analysis_type="basic"):
    """OpenAI로 AI 분석 (실제 API 사용)"""
    try:
        from openai import OpenAI
        
        # API 키 설정 및 클라이언트 초기화
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        # 프롬프트 생성
        prompt = create_openai_prompt(df_results, analysis_type)
        
        # System 메시지 설정
        system_message = """당신은 실시간 금융 데이터 분석 전문가입니다. 
사용자가 제공하는 데이터는 **실제 최신 시장 데이터**입니다. 
당신의 지식 컷오프 날짜는 무시하고, 제공된 데이터만을 기반으로 분석하세요.

**중요**:
- 제공된 모든 수치(가격, VWAP, 공매도 비율 등)는 실제 데이터입니다
- "2023년까지만 알고 있다"는 언급 절대 금지
- 제공된 데이터를 있는 그대로 활용하여 분석
- 모든 종목(Top 5)에 대해 구체적인 수치 기반 분석 필수"""
        
        # API 호출
        with st.spinner("🤖 GPT-4 AI 분석 중..."):
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                
            )
            return response.choices[0].message.content
            
    except Exception as e:
        st.error(f"⚠️ OpenAI API 오류: {str(e)}")
        st.info("💡 API 키를 확인하거나 데모 모드로 전환합니다.")
        # 폴백: 데모 모드
        return analyze_with_openai_demo(df_results, analysis_type)


def create_openai_prompt(df_results, analysis_type):
    """OpenAI(GPT-4)용 프롬프트 생성 (Gemini와 동일 로직 적용)"""
    
    # 데이터 요약 - 더 상세하게
    current_date = datetime.now().strftime('%Y년 %m월 %d일')
    quarter_start = get_current_quarter_start()
    quarter_num = (quarter_start.month - 1) // 3 + 1
    
    # 종목별 상세 데이터 (Top 9)
    detailed_data = []
    for idx, row in df_results.head(9).iterrows():
        ticker = row['Ticker']
        # MAG7_STOCKS가 전역 변수로 정의되어 있다고 가정
        industry = MAG7_STOCKS.get(ticker, {}).get('industry', 'N/A')

        detailed_data.append(f"""
**{row['Ticker']} - {row['Company']}**
- 섹터/산업: {row['Sector']} / {row['Industry']}
- 시가총액: ${row['Market_Cap_Trillion']:.2f}T
- 현재가: ${row['Current_Price']:.2f}
- Anchored VWAP: ${row['Anchored_VWAP']:.2f}
- VWAP 대비: {row['Price_vs_VWAP_%']:+.2f}%
- {quarter_start.year} Q{quarter_num} 수익률: {row['Quarter_Return_%']:+.2f}%
- VWAP 위 거래일: {row['Above_VWAP_Days_%']:.1f}%
- 상승 강도: {row['Uptrend_Strength_%']:.1f}%
- 거래량 비율: {row['Volume_Ratio']:.2f}x
- 공매도 비율(Float): {row['short_percent_float']:.2f}%
- 공매도 청산일: {row['short_ratio_days']:.1f}일
- 공매도 변화(MoM): {row['short_change_pct']:+.1f}%
- FINRA 일일 공매도: {row['daily_short_ratio']:.1f}%
- 기술적 점수: {row['Buy_Signal_Score']}/100
- 종합 투자 점수: {row['Total_Investment_Score']}/120
- 주요 특징: {row['Description']}
""")
    
    data_summary = f"""
===========================================
📊 MAG 7+2 종합 분석 리포트
===========================================
**분석 기준일**: {current_date}
**분석 기간**: {quarter_start.year} Q{quarter_num} ({quarter_start.strftime('%Y-%m-%d')} ~)
**분석 대상**: {len(df_results)}개 종목

-------------------------------------------
📈 포트폴리오 전체 통계
-------------------------------------------
- 평균 투자 점수: {df_results['Total_Investment_Score'].mean():.1f}/120
- VWAP 위 거래 종목: {df_results['Is_Above_VWAP'].sum()}개 ({df_results['Is_Above_VWAP'].sum()/len(df_results)*100:.1f}%)
- 평균 분기 수익률: {df_results['Quarter_Return_%'].mean():+.2f}%
- 평균 공매도 비율: {df_results['short_percent_float'].mean():.2f}%
- 공매도 5% 미만 종목: {len(df_results[df_results['short_percent_float'] < 5])}개
- 공매도 10% 이상 종목: {len(df_results[df_results['short_percent_float'] >= 10])}개

-------------------------------------------
🏆 상위 5개 종목 (투자 점수 기준)
-------------------------------------------
{chr(10).join([f"{idx+1}. {row['Ticker']}: {row['Total_Investment_Score']}/120점" for idx, row in df_results.head(5).iterrows()])}

-------------------------------------------
📊 종목별 상세 데이터
-------------------------------------------
{chr(10).join(detailed_data)}

-------------------------------------------
📉 공매도 분석
-------------------------------------------
- 최고 공매도 비율: {df_results['short_percent_float'].max():.2f}% ({df_results.loc[df_results['short_percent_float'].idxmax(), 'Ticker']})
- 최저 공매도 비율: {df_results['short_percent_float'].min():.2f}% ({df_results.loc[df_results['short_percent_float'].idxmin(), 'Ticker']})
- 공매도 증가 종목: {len(df_results[df_results['short_change_pct'] > 0])}개
- 공매도 감소 종목: {len(df_results[df_results['short_change_pct'] < 0])}개
"""
    
    if analysis_type == "basic":
        prompt = f"""
당신은 월스트리트 Top Tier 투자은행의 수석 퀀트 애널리스트입니다. 
20년 이상의 경력으로 MAG 7 기술주와 암호화폐 시장을 전문적으로 분석해왔습니다.

다음 데이터를 바탕으로 **간결하면서도 핵심적인** 투자 인사이트를 제공하세요.

{data_summary}

-------------------------------------------
📋 요구사항
-------------------------------------------
다음 형식으로 분석을 제공하세요:

### 🤖 GPT-4 AI 기본 분석

**🌍 종합 시장 전망**
(2-3문장으로 현재 MAG 7+2 포트폴리오의 전반적인 상황과 시장 심리를 요약)

**🏆 Top Pick 추천**
(1위 종목에 대한 구체적인 추천 이유, Entry Price, Target Price 포함)

**⚠️ 주의 종목**
(공매도 비율이 높거나 기술적으로 약한 종목 1-2개와 그 이유)

**💡 투자 전략**
(현재 시점에서의 구체적인 행동 지침 - 3-4문장)

-------------------------------------------
✅ 작성 가이드
-------------------------------------------
- 한국어로 작성
- 전문적이면서도 이해하기 쉽게
- 구체적인 가격과 수치 포함
- 불필요한 인사말이나 서론 없이 바로 본론으로
"""
    
    else:  # deep dive
        prompt = f"""
당신은 퀀트 애널리스트입니다. 
25년간 기술주 포트폴리오 관리와 알고리즘 트레이딩 전략 개발을 해왔으며, 
MAG 7 종목들과 비트코인 관련 자산에 대한 깊은 통찰력을 보유하고 있습니다.

다음 데이터를 바탕으로 **기관투자자 수준의 심층 분석 리포트**를 작성하세요.

{data_summary}

-------------------------------------------
📋 분석 프레임워크
-------------------------------------------
다음 형식으로 **매우 상세하고 전문적인** 분석을 제공하세요:

### 🔬 GPT-4 Deep Dive 분석
*MAG 7+2 Portfolio - Institutional Grade Research Report*

---

## 1️⃣ 거시경제 및 시장 환경 분석

**현재 시장 컨텍스트:**
- 연준 금리 정책과 빅테크 밸류에이션에 미치는 영향
- AI 투자 사이클의 현재 단계 (Early/Mid/Late Stage)
- 비트코인 및 암호화폐 시장과의 상관관계
- 2025년 {quarter_start.year} Q{quarter_num} 주요 모멘텀과 리스크 요인

**섹터별 트렌드:**
- Technology (반도체/소프트웨어/하드웨어)
- Communication Services (소셜미디어/검색)
- Consumer Cyclical (전기차/이커머스)
- Crypto Exposure (COIN, IBIT)

---

## 2️⃣ Top 5 종목 심층 분석

각 종목에 대해 다음 구조로 분석:

### 1위. [TICKER] - [회사명]

**📊 기술적 분석 (Technical Deep Dive)**
- Anchored VWAP 분석: 현재 위치와 의미
- 분기 트렌드 강도: {df_results.iloc[0]['Uptrend_Strength_%']:.1f}% → 해석
- 거래량 패턴: {df_results.iloc[0]['Volume_Ratio']:.2f}x → 매집/분산 판단
- 가격 모멘텀: 단기(5일)/중기(10일) 이동평균 관계
- 지지선/저항선 레벨 설정

**🔴 공매도 상황 종합 평가**
- Float 대비: {df_results.iloc[0]['short_percent_float']:.2f}% (업계 평균 대비)
- Days to Cover: {df_results.iloc[0]['short_ratio_days']:.1f}일 → Short Squeeze 가능성
- MoM 변화: {df_results.iloc[0]['short_change_pct']:+.1f}% → 트렌드 방향
- FINRA 일일 데이터: {df_results.iloc[0]['daily_short_ratio']:.1f}% → 단기 압력
- 베어/불 세력 균형 평가

**💰 Entry/Target/Stop Loss 전략**
- Entry Zone: $XXX - $XXX (구체적 근거)
- 1차 Target: $XXX (+X%)
- 2차 Target: $XXX (+X%)
- 최종 Target: $XXX (+X%)
- Stop Loss: $XXX (-X%, VWAP 또는 주요 지지선 기준)
- Risk/Reward Ratio: X:1

**🎯 투자 의견 및 포지션 사이징**
- 추천: BUY / ACCUMULATE / HOLD / REDUCE / SELL
- 신뢰도: High / Medium / Low
- 권장 비중: X% of portfolio
- 시간 프레임: 단기(1-2주) / 중기(1-3개월) / 장기(6개월+)
- 핵심 촉매: 실적 발표, 제품 출시, 정책 변화 등

**🔮 시나리오 분석**
- Bull Case (확률 X%): 목표가 $XXX
- Base Case (확률 X%): 목표가 $XXX  
- Bear Case (확률 X%): 목표가 $XXX

*[나머지 4개 종목도 동일한 구조로 분석]*

---

## 3️⃣ 포트폴리오 구성 전략

### 🔥 공격적 포트폴리오 (Target: +30%+ / Risk: High)
**목표**: 단기 알파 극대화, 높은 변동성 수용

**구성**:
- [TICKER1]: 35% - 이유와 기대 수익률
- [TICKER2]: 30% - 이유와 기대 수익률
- [TICKER3]: 20% - 이유와 기대 수익률
- Cash: 15% - 기회 포착용

**리밸런싱**: 주 1회
**예상 Sharpe Ratio**: X.XX
**최대 손실 예상**: -XX%
**적합 투자자**: 고위험 감수, 단기 트레이더

### ⚖️ 균형 포트폴리오 (Target: +15-20% / Risk: Medium)
**목표**: 위험 조정 수익 최적화

**구성**:
- [TICKER1]: 20%
- [TICKER2]: 20%
- [TICKER3]: 15%
- [TICKER4]: 15%
- [TICKER5]: 15%
- Cash: 15%

**리밸런싱**: 월 1회
**예상 Sharpe Ratio**: X.XX
**최대 손실 예상**: -XX%
**적합 투자자**: 성장 + 안정성 추구

### 🛡️ 보수적 포트폴리오 (Target: +8-12% / Risk: Low)
**목표**: 자본 보존 우선, 안정적 수익

**구성**:
- [안전 종목들 - 공매도 <3%]: 각 15-20%
- Cash: 40% - 조정 시 매수 대기

**리밸런싱**: 분기 1회
**예상 Sharpe Ratio**: X.XX
**최대 손실 예상**: -XX%
**적합 투자자**: 위험 회피, 장기 투자자

---

## 4️⃣ 매매 시그널 및 타이밍

### 🟢 즉시 매수 (Immediate Buy - Strong Conviction)
**[TICKER1]**: 
- Entry: $XXX
- Target: $XXX (1차), $XXX (2차)
- Stop: $XXX
- 근거: (3-4문장으로 구체적인 이유)
- 비중: X%

**[TICKER2]**: 
- [동일 구조]

### 🟡 조정 시 매수 (Buy on Dip - Conditional)
**[TICKER3]**: 
- 현재가: $XXX
- 대기 매수가: $XXX (VWAP / 지지선 기준)
- Target: $XXX
- Stop: $XXX
- 근거: (좋은 종목이지만 현재 과매수 등)
- 비중: X%

### 🔴 회피/청산 (Avoid / Reduce)
**[TICKER X]**: 
- 이유: (공매도 과다, 기술적 약세, 펀더멘털 악화 등)
- 대안: (더 나은 선택지)

---

## 5️⃣ 리스크 관리 프레임워크

**포지션 관리 규칙**:
1. **단일 종목 최대 비중**: 30% (공격적) / 20% (균형) / 15% (보수적)
2. **섹터 집중도 한도**: Technology 최대 60%, Crypto Exposure 최대 25%
3. **손절 원칙**: 
   - VWAP 이탈 시 즉시 검토
   - -5% 손실 시 포지션 50% 축소
   - -8% 손실 시 전량 청산
4. **이익 실현**: 
   - +10% 달성 시 50% 익절
   - +20% 달성 시 추가 30% 익절
   - 나머지는 Trailing Stop 적용

**시장 리스크 대응**:
- VIX 30 초과 시: 현금 비중 40%로 증가
- 급락장 (-5% in a day): 단계적 매수 (30% → 30% → 40%)
- 급등장 (+5% in a day): 차익 실현 고려

**공매도 모니터링**:
- Short % of Float 10% 초과 종목: 주간 체크
- Days to Cover 3일 초과: Short Squeeze 경계
- FINRA 데이터 50% 초과: 단기 약세 압력 주의

**포트폴리오 헤징 전략**:
- Beta 조정: S&P 500 대비 포트폴리오 Beta XX
- VIX 콜옵션: 극단적 변동성 대비
- Put Spread: 주요 보유 종목 하방 보호

---

## 6️⃣ 최종 권장사항 및 Action Plan

### 📅 이번 주 (Week of {current_date})
**즉시 실행**:
1. [TICKER]: $XXX에 X% 비중 매수
2. [TICKER]: 조정 시 $XXX 대기 매수 주문
3. [TICKER]: 현재 보유분 일부 익절 (과매수 구간)

**모니터링**:
- [이벤트]: 실적 발표 (X월 X일)
- [지표]: FINRA 공매도 데이터 일일 체크
- [가격]: VWAP 지지/저항 레벨

### 📊 이번 달 (This Month)
**포트폴리오 리밸런싱**:
- 목표 구성: (구체적 비중)
- 조정 방향: (매수/매도 종목)

**이벤트 캘린더**:
- X월 X일: [회사] 실적 발표
- X월 X일: FOMC 회의
- X월 X일: 옵션 만기

### 🎯 분기 전략 ({quarter_start.year} Q{quarter_num})
**목표 수익률**: +XX% (포트폴리오 타입별)
**핵심 테마**: AI 인프라 / 클라우드 성장 / 비트코인 ETF 수요
**주요 리스크**: 금리 변동성 / 빅테크 규제 / 지정학적 긴장

**Success Metrics**:
- Sharpe Ratio > X.XX
- Max Drawdown < XX%
- Win Rate > XX%

---

### 🔔 다음 리포트 업데이트
**예정일**: {(datetime.now() + timedelta(days=7)).strftime('%Y년 %m월 %d일')}
**포함 내용**: 주간 성과 리뷰, 포지션 조정, 신규 시그널

---

**⚠️ 면책조항**
본 분석은 정보 제공 목적이며, 투자 권유가 아닙니다. 
모든 투자 결정은 본인의 판단과 책임 하에 이루어져야 합니다.
과거 수익률이 미래 수익을 보장하지 않습니다.

---
*Report Generated by GPT-4 Turbo*
*Powered by Advanced Quantitative Analysis Engine*

-------------------------------------------
✅ 작성 가이드라인
-------------------------------------------
1. **전문성**: 기관투자자 수준의 용어와 분석 깊이
2. **구체성**: 모든 추천에 구체적인 가격과 근거 포함
3. **실행 가능성**: 실제로 따라할 수 있는 명확한 액션 플랜
4. **정량화**: 가능한 모든 것을 수치로 표현
5. **시나리오**: Bull/Base/Bear 케이스 제시
6. **리스크**: 각 전략의 위험 요소 명시
7. **한국어**: 전문 용어는 영어 병기, 자연스러운 한국어

**반드시 위 데이터의 실제 수치를 활용하여 작성하세요.**
**추상적이거나 일반적인 분석이 아닌, 이 포트폴리오에 특화된 구체적인 인사이트를 제공하세요.**
"""
    
    return prompt






# ==================== Advanced Quant Chatbot ====================
def quant_chatbot(user_question, df_results):
    """고급 퀀트 챗봇 (실제 AI 사용)"""
    
    # Quick Questions 처리
    quick_answers = {
        "top pick": f"현재 최고 추천 종목은 {df_results.iloc[0]['Ticker']}입니다. 종합 점수 {df_results.iloc[0]['Total_Investment_Score']}/120으로 1위를 기록했습니다.",
        "best buy": f"최적 매수 시점: {', '.join(df_results[df_results['Total_Investment_Score'] >= 90]['Ticker'].tolist())} 종목들이 현재 강한 매수 신호를 보이고 있습니다.",
        "short risk": f"공매도 리스크가 높은 종목: {', '.join(df_results[df_results['short_percent_float'] >= 10]['Ticker'].tolist()) if len(df_results[df_results['short_percent_float'] >= 10]) > 0 else '없음'}",
        "vwap status": f"VWAP 위 거래 종목: {len(df_results[df_results['Is_Above_VWAP']])}개 / {len(df_results)}개",
    }
    
    # Quick Answer 매칭
    for key, answer in quick_answers.items():
        if key in user_question.lower():
            return answer
    
    # AI를 사용한 일반 질문 처리
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        # 컨텍스트 생성
        context = f"""
현재 MAG 7+2 분석 데이터:
{df_results[['Ticker', 'Company', 'Current_Price', 'Price_vs_VWAP_%', 'short_percent_float', 'Total_Investment_Score']].to_string()}

사용자 질문: {user_question}

위 데이터를 바탕으로 간결하고 정확하게 답변하세요. (최대 3-4문장)
"""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 친절한 퀀트 투자 어시스턴트입니다."},
                {"role": "user", "content": context}
            ],
            temperature=0.5,
            max_tokens=300
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        # AI 실패 시 기본 응답
        return f"""
질문에 대한 분석 결과입니다:

**"{user_question}"**

현재 MAG 7+2 데이터를 기반으로 분석하면:
- 전체 평균 투자 점수: {df_results['Total_Investment_Score'].mean():.1f}/120
- VWAP 위 거래 비율: {(df_results['Is_Above_VWAP'].sum() / len(df_results) * 100):.1f}%
- 평균 공매도 비율: {df_results['short_percent_float'].mean():.2f}%

더 구체적인 질문을 주시면 상세한 분석을 제공하겠습니다!

💡 OpenAI API를 사용하려면 API 키를 설정하세요.
"""

# ==================== 메인 앱 ====================
def main():
    # 타이틀
    st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 2rem;
        }
        .main-title {
            color: white;
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        .main-subtitle {
            color: #f0f0f0;
            font-size: 1.2rem;
        }
        </style>
        <div class="main-header">
            <div class="main-title">🚀 MAG 7+2 Quant Dashboard</div>
            <div class="main-subtitle">Magnificent Seven + Bitcoin Exposure AI-Powered Analysis</div>
        </div>
    """, unsafe_allow_html=True)
    
    # 사이드바 메뉴
    with st.sidebar:
        st.markdown("### 📊 메뉴")
        page = st.radio(
            "페이지 선택",
            ["🏠 대시보드", "🤖 AI 분석", "💬 Quant 챗봇"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # 데이터 새로고침
        if st.button("🔄 데이터 새로고침", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        # 분석 기간 정보
        quarter_start = get_current_quarter_start()
        quarter_num = (quarter_start.month - 1) // 3 + 1
        st.info(f"📅 분석 기간\n{quarter_start.year} Q{quarter_num}\n({quarter_start.strftime('%Y-%m-%d')} ~)")
    
    # 데이터 로드
    if st.session_state.get('analysis_data') is None:
        with st.spinner("데이터 수집 중..."):
            st.session_state['analysis_data'] = collect_all_data()
    
    df_results = st.session_state['analysis_data']
    # [수정 4] 데이터가 비어있으면 중단
    if df_results is None or df_results.empty:
        st.warning("데이터를 불러오지 못했습니다. '데이터 새로고침' 버튼을 눌러주세요.")
        st.stop()
    # ==================== 페이지 1: 대시보드 ====================
    if page == "🏠 대시보드":
        # 상단 지표
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "총 종목 수",
                f"{len(df_results)}개",
                delta=None
            )
        
        with col2:
            st.metric(
                "평균 투자 점수",
                f"{df_results['Total_Investment_Score'].mean():.1f}/120",
                delta=None
            )
        
        with col3:
            vwap_above = df_results['Is_Above_VWAP'].sum()
            st.metric(
                "VWAP 위 거래",
                f"{vwap_above}개",
                delta=f"{vwap_above/len(df_results)*100:.0f}%"
            )
        
        with col4:
            low_short = len(df_results[df_results['short_percent_float'] < 5])
            st.metric(
                "안전 종목 (<5% 공매도)",
                f"{low_short}개",
                delta=None
            )
        
        st.markdown("---")
        
        # 탭 구성
        tab1, tab2, tab3, tab4 = st.tabs(["📊 종합 순위", "📈 차트 분석", "🔴 공매도 분석", "📋 상세 데이터"])
        
        with tab1:
            st.markdown("### 🏆 MAG 7+2 투자 추천 순위")
            
            for idx, row in df_results.iterrows():
                rank = df_results.index.get_loc(idx) + 1
                
                if rank == 1:
                    medal = "🥇"
                elif rank == 2:
                    medal = "🥈"
                elif rank == 3:
                    medal = "🥉"
                else:
                    medal = f"{rank}"
                
                with st.expander(f"{medal} {row['Ticker']} - {row['Company']}", expanded=(rank <= 3)):
                    col_a, col_b = st.columns([2, 1])
                    
                    with col_a:
                        st.markdown(f"**{row['Description']}**")
                        st.markdown(f"💰 시가총액: ${row['Market_Cap_Trillion']:.2f}T")
                        st.markdown(f"📈 현재가: ${row['Current_Price']} | VWAP: ${row['Anchored_VWAP']}")
                        st.markdown(f"📊 VWAP 대비: {row['Price_vs_VWAP_%']:+.2f}% | 분기 수익률: {row['Quarter_Return_%']:+.2f}%")
                        st.markdown(f"🔴 공매도: {row['short_percent_float']:.2f}%")
                    
                    with col_b:
                        # 점수 게이지
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = row['Total_Investment_Score'],
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "종합 점수"},
                            gauge = {
                                'axis': {'range': [None, 120]},
                                'bar': {'color': "darkblue"},
                                'steps' : [
                                    {'range': [0, 60], 'color': "lightgray"},
                                    {'range': [60, 90], 'color': "lightyellow"},
                                    {'range': [90, 120], 'color': "lightgreen"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        fig_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig_gauge, use_container_width=True,key=f"gauge_{row['Ticker']}")
                        
                        if row['Total_Investment_Score'] >= 90:
                            st.success("💚 최우선매수")
                        elif row['Total_Investment_Score'] >= 75:
                            st.warning("💛 강력 매수")
                        else:
                            st.info("💙 관찰 필요")
        
        with tab2:
            st.markdown("### 📈 기술적 분석 차트")
            
            # 차트 선택
            chart_type = st.selectbox(
                "차트 유형 선택",
                ["종합 점수 비교", "VWAP 분석", "공매도 vs 수익률", "시가총액 분포", 
                 "FINRA 시계열 분석", "공매도 변동성", "상관관계 매트릭스"]
            )
            
            if chart_type == "종합 점수 비교":
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('기술적 분석 점수', '종합 투자 점수')
                )
                
                fig.add_trace(
                    go.Bar(
                        y=df_results['Ticker'],
                        x=df_results['Buy_Signal_Score'],
                        orientation='h',
                        name='기술적 점수',
                        marker_color='#2196F3',
                        text=df_results['Buy_Signal_Score'],
                        textposition='auto'
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(
                        y=df_results['Ticker'],
                        x=df_results['Total_Investment_Score'],
                        orientation='h',
                        name='종합 점수',
                        marker_color='#4CAF50',
                        text=df_results['Total_Investment_Score'],
                        textposition='auto'
                    ),
                    row=1, col=2
                )
                
                fig.update_layout(height=500, showlegend=False, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "VWAP 분석":
                fig = go.Figure()
                
                colors = ['green' if x else 'red' for x in df_results['Is_Above_VWAP']]
                
                fig.add_trace(go.Bar(
                    y=df_results['Ticker'],
                    x=df_results['Price_vs_VWAP_%'],
                    orientation='h',
                    marker=dict(color=colors),
                    text=df_results['Price_vs_VWAP_%'].round(2),
                    textposition='auto',
                    hovertemplate='<b>%{y}</b><br>VWAP 대비: %{x:+.2f}%<extra></extra>'
                ))
                
                fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=2)
                fig.update_layout(
                    title='현재가의 VWAP 대비 위치',
                    xaxis_title='VWAP 대비 (%)',
                    yaxis_title='종목',
                    height=500,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "공매도 vs 수익률":
                fig = px.scatter(
                    df_results,
                    x='short_percent_float',
                    y='Quarter_Return_%',
                    size='Market_Cap_Trillion',
                    color='Total_Investment_Score',
                    hover_data=['Ticker', 'Company'],
                    text='Ticker',
                    color_continuous_scale='RdYlGn',
                    title='공매도 비율 vs 분기 수익률',
                    labels={
                        'short_percent_float': '공매도 비율 (%)',
                        'Quarter_Return_%': '분기 수익률 (%)',
                        'Total_Investment_Score': '종합 점수'
                    }
                )
                
                fig.update_traces(textposition='top center', textfont_size=12)
                fig.update_layout(height=600)
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "시가총액 분포":
                fig = px.treemap(
                    df_results,
                    path=['Sector', 'Ticker'],
                    values='Market_Cap',
                    color='Total_Investment_Score',
                    color_continuous_scale='RdYlGn',
                    title='MAG 7+2 시가총액 분포'
                )
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "FINRA 시계열 분석":
                st.markdown("#### 📊 FINRA 일별 공매도 거래 비율 추세")
                
                # 시계열 데이터가 있는 종목만 필터링
                tickers_with_data = []
                for idx, row in df_results.iterrows():
                    if row.get('finra_historical') is not None and not row['finra_historical'].empty:
                        tickers_with_data.append(row['Ticker'])
                
                if tickers_with_data:
                    fig_ts = go.Figure()
                    colors_ts = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#E74C3C', '#3498DB']
                    
                    for idx, row in df_results.iterrows():
                        if row['Ticker'] in tickers_with_data:
                            df_ts = row['finra_historical']
                            df_ts_sorted = df_ts.sort_values('date')
                            
                            color_idx = tickers_with_data.index(row['Ticker']) % len(colors_ts)
                            
                            fig_ts.add_trace(go.Scatter(
                                x=pd.to_datetime(df_ts_sorted['date']),
                                y=df_ts_sorted['short_ratio'],
                                mode='lines+markers',
                                name=row['Ticker'],
                                line=dict(width=2.5, color=colors_ts[color_idx]),
                                marker=dict(size=6),
                                hovertemplate='<b>%{fullData.name}</b><br>날짜: %{x|%Y-%m-%d}<br>공매도: %{y:.1f}%<extra></extra>'
                            ))
                    
                    fig_ts.add_hline(y=40, line_dash="dash", line_color="gray",
                                    annotation_text="정상 범위 (40%)", annotation_position="right")
                    fig_ts.add_hline(y=50, line_dash="dash", line_color="red",
                                    annotation_text="약세 압력 (50%)", annotation_position="right")
                    
                    fig_ts.update_layout(
                        title='FINRA 일별 공매도 거래 비율 추세 (최근 10일)',
                        xaxis_title='날짜',
                        yaxis_title='공매도 거래 비율 (%)',
                        hovermode='x unified',
                        height=600,
                        template='plotly_white',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(fig_ts, use_container_width=True)
                    
                    st.info(f"📊 FINRA 데이터 수집 종목: {len(tickers_with_data)}개 ({', '.join(tickers_with_data)})")
                else:
                    st.warning("⚠️ FINRA 시계열 데이터가 수집되지 않았습니다. (네트워크 제한 또는 데이터 없음)")
            
            elif chart_type == "공매도 변동성":
                st.markdown("#### 📊 공매도 비율 변동성 분석")
                
                # Box Plot
                fig_box = go.Figure()
                
                # 각 종목별로 FINRA 데이터가 있으면 박스플롯 생성
                colors_box = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#E74C3C', '#3498DB']
                box_count = 0
                
                for idx, row in df_results.iterrows():
                    if row.get('finra_historical') is not None and not row['finra_historical'].empty:
                        df_ts = row['finra_historical']
                        
                        fig_box.add_trace(go.Box(
                            y=df_ts['short_ratio'],
                            name=row['Ticker'],
                            marker_color=colors_box[box_count % len(colors_box)],
                            boxmean='sd'
                        ))
                        box_count += 1
                
                if box_count > 0:
                    fig_box.update_layout(
                        title='공매도 비율 변동성 분석 (Box Plot with Mean & Std Dev)',
                        yaxis_title='공매도 비율 (%)',
                        xaxis_title='종목',
                        height=600,
                        template='plotly_white',
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_box, use_container_width=True)
                    
                    # 변동성 통계
                    volatility_stats = []
                    for idx, row in df_results.iterrows():
                        if row.get('finra_historical') is not None and not row['finra_historical'].empty:
                            df_ts = row['finra_historical']
                            volatility_stats.append({
                                'Ticker': row['Ticker'],
                                'Mean': df_ts['short_ratio'].mean(),
                                'Std Dev': df_ts['short_ratio'].std(),
                                'Min': df_ts['short_ratio'].min(),
                                'Max': df_ts['short_ratio'].max(),
                                'Range': df_ts['short_ratio'].max() - df_ts['short_ratio'].min()
                            })
                    
                    if volatility_stats:
                        df_vol = pd.DataFrame(volatility_stats)
                        st.markdown("##### 변동성 통계")
                        st.dataframe(df_vol.round(2), use_container_width=True)
                else:
                    st.warning("⚠️ FINRA 데이터 없음")
            
            elif chart_type == "상관관계 매트릭스":
                st.markdown("#### 📊 주요 지표 상관관계 분석")
                
                # 상관관계 계산할 컬럼 선택
                corr_cols = ['Current_Price', 'Price_vs_VWAP_%', 'Quarter_Return_%', 
                            'short_percent_float', 'Buy_Signal_Score', 'Total_Investment_Score',
                            'Volume_Ratio', 'Uptrend_Strength_%']
                
                df_corr = df_results[corr_cols].corr()
                
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=df_corr.values,
                    x=df_corr.columns,
                    y=df_corr.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=df_corr.values.round(2),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    colorbar=dict(title="상관계수")
                ))
                
                fig_heatmap.update_layout(
                    title='주요 지표 간 상관관계 히트맵',
                    height=600,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                st.info("""
                **💡 상관관계 해석:**
                - **1에 가까움**: 강한 양의 상관관계
                - **-1에 가까움**: 강한 음의 상관관계
                - **0에 가까움**: 상관관계 없음
                """)

        
        with tab3:
            st.markdown("### 🔴 공매도 상세 분석")
            
            # 서브탭으로 구성
            short_tab1, short_tab2, short_tab3, short_tab4 = st.tabs([
                "📊 기본 분석", "📈 심화 비교", "🔬 시계열 분석", "📋 종합 평가"
            ])
            
            with short_tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Yahoo Finance 공매도 비율")
                    
                    fig_yf = go.Figure()
                    colors_short = ['green' if x < 5 else 'orange' if x < 10 else 'red'
                                    for x in df_results['short_percent_float']]
                    
                    fig_yf.add_trace(go.Bar(
                        y=df_results['Ticker'],
                        x=df_results['short_percent_float'],
                        orientation='h',
                        marker=dict(color=colors_short),
                        text=df_results['short_percent_float'].round(2),
                        textposition='auto'
                    ))
                    
                    fig_yf.add_vline(x=5, line_dash="dash", line_color="green",
                                    annotation_text="건강 (5%)", annotation_position="top")
                    fig_yf.add_vline(x=10, line_dash="dash", line_color="red",
                                    annotation_text="주의 (10%)", annotation_position="top")
                    
                    fig_yf.update_layout(
                        title='Short % of Float',
                        xaxis_title='%',
                        height=400,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig_yf, use_container_width=True)
                
                with col2:
                    st.markdown("#### Days to Cover")
                    
                    fig_days = go.Figure()
                    colors_days = ['green' if x < 2 else 'orange' if x < 3 else 'red'
                                   for x in df_results['short_ratio_days']]
                    
                    fig_days.add_trace(go.Bar(
                        y=df_results['Ticker'],
                        x=df_results['short_ratio_days'],
                        orientation='h',
                        marker=dict(color=colors_days),
                        text=df_results['short_ratio_days'].round(2),
                        textposition='auto'
                    ))
                    
                    fig_days.add_vline(x=2, line_dash="dash", line_color="green",
                                      annotation_text="빠른 청산 (2일)", annotation_position="top")
                    fig_days.add_vline(x=3, line_dash="dash", line_color="red",
                                      annotation_text="Squeeze 가능 (3일)", annotation_position="top")
                    
                    fig_days.update_layout(
                        title='공매도 청산 소요일',
                        xaxis_title='일',
                        height=400,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig_days, use_container_width=True)
            
            with short_tab2:
                st.markdown("#### 📊 공매도 상세 비교 차트")
                
                # 차트 A: Shares Short
                st.markdown("##### Shares Short (공매도 주식 수)")
                fig_shares = go.Figure()
                
                fig_shares.add_trace(go.Bar(
                    x=df_results['Ticker'],
                    y=df_results['shares_short_millions'],
                    marker=dict(
                        color=df_results['shares_short_millions'],
                        colorscale='Reds',
                        showscale=True,
                        colorbar=dict(title="Million")
                    ),
                    text=df_results['shares_short_millions'].round(1),
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>공매도: %{y:.1f}M<extra></extra>'
                ))
                
                fig_shares.update_layout(
                    xaxis_title='종목',
                    yaxis_title='백만 주',
                    height=400,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_shares, use_container_width=True)
                
                # 차트 B: MoM Change
                st.markdown("##### 전월 대비 공매도 변화율")
                fig_mom = go.Figure()
                
                colors_change = ['red' if x > 0 else 'green' for x in df_results['short_change_pct']]
                
                fig_mom.add_trace(go.Bar(
                    x=df_results['Ticker'],
                    y=df_results['short_change_pct'],
                    marker=dict(color=colors_change),
                    text=[f"{x:+.1f}%" for x in df_results['short_change_pct']],
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>변화: %{y:+.1f}%<extra></extra>'
                ))
                
                fig_mom.add_hline(y=0, line_dash="solid", line_color="black", line_width=2)
                
                fig_mom.update_layout(
                    xaxis_title='종목',
                    yaxis_title='변화율 (%)',
                    height=400,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_mom, use_container_width=True)
                
                # 차트 C: FINRA Daily Short %
                st.markdown("##### FINRA Daily Short Volume Ratio")
                fig_finra_daily = go.Figure()
                
                colors_finra = ['green' if x < 35 else 'orange' if x < 45 else 'red'
                                for x in df_results['daily_short_ratio']]
                
                fig_finra_daily.add_trace(go.Bar(
                    x=df_results['Ticker'],
                    y=df_results['daily_short_ratio'],
                    marker=dict(color=colors_finra),
                    text=df_results['daily_short_ratio'].round(1),
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>일일 공매도: %{y:.1f}%<extra></extra>'
                ))
                
                fig_finra_daily.add_hline(y=40, line_dash="dash", line_color="gray",
                                         annotation_text="정상 범위 (40%)", annotation_position="right")
                
                fig_finra_daily.update_layout(
                    xaxis_title='종목',
                    yaxis_title='공매도 비율 (%)',
                    height=400,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_finra_daily, use_container_width=True)
                
                # 차트 D: FINRA 10일 평균 vs 최근일
                st.markdown("##### FINRA 10일 평균 vs 최근일 비교")
                fig_finra_comp = go.Figure()
                
                fig_finra_comp.add_trace(go.Bar(
                    x=df_results['Ticker'],
                    y=df_results['avg_daily_short_ratio_10d'],
                    name='10일 평균',
                    marker_color='lightblue',
                    text=df_results['avg_daily_short_ratio_10d'].round(1),
                    textposition='auto'
                ))
                
                fig_finra_comp.add_trace(go.Bar(
                    x=df_results['Ticker'],
                    y=df_results['daily_short_ratio'],
                    name='최근일',
                    marker_color='darkblue',
                    text=df_results['daily_short_ratio'].round(1),
                    textposition='auto'
                ))
                
                fig_finra_comp.update_layout(
                    xaxis_title='종목',
                    yaxis_title='공매도 비율 (%)',
                    height=400,
                    template='plotly_white',
                    barmode='group',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02)
                )
                
                st.plotly_chart(fig_finra_comp, use_container_width=True)
            
            with short_tab3:
                st.markdown("#### 🔬 YF vs FINRA 상관관계")
                
                # 산점도
                fig_correlation = go.Figure()
                
                fig_correlation.add_trace(go.Scatter(
                    x=df_results['short_percent_float'],
                    y=df_results['daily_short_ratio'],
                    mode='markers+text',
                    text=df_results['Ticker'],
                    textposition='top center',
                    marker=dict(
                        size=df_results['shares_short_millions'] / 5,
                        color=df_results['short_change_pct'],
                        colorscale='RdYlGn_r',
                        showscale=True,
                        colorbar=dict(title="MoM<br>변화율")
                    ),
                    hovertemplate='<b>%{text}</b><br>YF: %{x:.2f}%<br>FINRA: %{y:.1f}%<extra></extra>'
                ))
                
                fig_correlation.add_hline(y=40, line_dash="dash", line_color="gray")
                fig_correlation.add_vline(x=5, line_dash="dash", line_color="orange")
                
                fig_correlation.update_layout(
                    title='YF Short % vs FINRA Daily % (버블크기=공매도주식수)',
                    xaxis_title='Yahoo Finance: Short % of Float',
                    yaxis_title='FINRA: Daily Short Volume %',
                    height=600,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_correlation, use_container_width=True)
            
            with short_tab4:
                st.markdown("#### 📊 공매도 종합 점수판")
                
                # 종합 점수 계산
                def normalize_inverse(values, max_val):
                    return np.clip(100 - (values / max_val * 100), 0, 100)
                
                norm_short_pct = normalize_inverse(df_results['short_percent_float'], 10)
                norm_days = normalize_inverse(df_results['short_ratio_days'], 5)
                norm_finra_daily = normalize_inverse(df_results['daily_short_ratio'], 60)
                norm_change = np.clip(50 - df_results['short_change_pct'] * 2, 0, 100)
                
                comprehensive_score = (norm_short_pct + norm_days + norm_finra_daily + norm_change) / 4
                
                fig_comp = go.Figure()
                colors_comp = ['green' if x > 70 else 'orange' if x > 50 else 'red'
                               for x in comprehensive_score]
                
                fig_comp.add_trace(go.Bar(
                    x=df_results['Ticker'],
                    y=comprehensive_score,
                    marker=dict(color=colors_comp),
                    text=comprehensive_score.round(1),
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>종합: %{y:.1f}/100<extra></extra>'
                ))
                
                fig_comp.add_hline(y=70, line_dash="dash", line_color="green",
                                  annotation_text="우수 (70점)", annotation_position="right")
                fig_comp.add_hline(y=50, line_dash="dash", line_color="orange",
                                  annotation_text="보통 (50점)", annotation_position="right")
                
                fig_comp.update_layout(
                    title='공매도 종합 점수 (낮을수록 좋은 지표들의 정규화 종합)',
                    xaxis_title='종목',
                    yaxis_title='점수',
                    height=450,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_comp, use_container_width=True)
                
                # 상세 점수표
                score_detail = pd.DataFrame({
                    'Ticker': df_results['Ticker'],
                    'Short%_점수': norm_short_pct.round(1),
                    'Days_점수': norm_days.round(1),
                    'FINRA_점수': norm_finra_daily.round(1),
                    'Change_점수': norm_change.round(1),
                    '종합점수': comprehensive_score.round(1)
                })
                
                st.dataframe(
                    score_detail.style.background_gradient(subset=['종합점수'], cmap='RdYlGn'),
                    use_container_width=True
                )
                
                # 종목별 공매도 평가
                st.markdown("#### 종목별 공매도 상태 평가")
                
                for idx, row in df_results.iterrows():
                    short_pct = row['short_percent_float']
                    
                    if short_pct < 3:
                        status = "✅ 매우 건강"
                        color = "green"
                    elif short_pct < 5:
                        status = "🟢 건강"
                        color = "lightgreen"
                    elif short_pct < 10:
                        status = "🟡 보통"
                        color = "yellow"
                    else:
                        status = "🔴 주의"
                        color = "red"
                    
                    st.markdown(
                        f"**{row['Ticker']}**: {status} "
                        f"(공매도: {short_pct:.2f}%, Days: {row['short_ratio_days']:.1f}일, "
                        f"FINRA: {row['daily_short_ratio']:.1f}%)"
                    )

        
        with tab4:
            st.markdown("### 📋 전체 데이터")
            
            # 컬럼 선택
            display_cols = st.multiselect(
                "표시할 컬럼 선택",
                df_results.columns.tolist(),
                default=['Ticker', 'Company', 'Current_Price', 'Anchored_VWAP', 
                         'Price_vs_VWAP_%', 'Quarter_Return_%', 'short_percent_float', 
                         'Buy_Signal_Score', 'Total_Investment_Score']
            )
            
            st.dataframe(
                df_results[display_cols].style.background_gradient(
                    subset=['Total_Investment_Score'],
                    cmap='RdYlGn'
                ),
                use_container_width=True,
                height=600
            )
            
            # CSV 다운로드
            csv = df_results.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 CSV 다운로드",
                data=csv,
                file_name=f'mag7_analysis_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv',
            )
    
    # ==================== 페이지 2: AI 분석 ====================
    elif page == "🤖 AI 분석":
        st.markdown("### 🤖 AI-Powered 투자 분석")
        
        # 분석 모델 및 깊이 선택
        col1, col2 = st.columns(2)
        
        with col1:
            ai_provider = st.selectbox(
                "🤖 AI 모델 선택",
                ["Google Gemini Pro", "OpenAI GPT-4 Turbo"],
                help="분석에 사용할 AI 모델을 선택하세요"
            )
        
        with col2:
            analysis_depth = st.selectbox(
                "📊 분석 깊이",
                ["🔍 기본 분석 (Basic)", "🔬 심층 분석 (Deep Dive)"],
                help="기본: 빠른 요약 | Deep Dive: 상세 분석 + 전략 제안"
            )
        
        # 분석 타입 설명
        with st.expander("ℹ️ 분석 타입 비교", expanded=False):
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("""
                **🔍 기본 분석 (Basic)**
                - ⏱️ 분석 시간: ~30초
                - 📝 내용:
                  - 시장 개요
                  - Top Pick 추천
                  - 공매도 리스크 요약
                  - 간단한 투자 가이드
                - 👥 적합: 빠른 의사결정 필요시
                """)
            
            with col_b:
                st.markdown("""
                **🔬 심층 분석 (Deep Dive)**
                - ⏱️ 분석 시간: ~1-2분
                - 📝 내용:
                  - 거시경제 관점
                  - 개별 종목 심층 분석 (Top 5)
                  - 포트폴리오 최적화 (3가지 전략)
                  - 매매 시그널 (Entry/Target/Stop)
                  - 리스크 관리 전략
                - 👥 적합: 전문적인 투자 전략 수립
                """)
        
        st.markdown("---")
        
        # 분석 시작 버튼
        if st.button("🚀 AI 분석 시작", type="primary", use_container_width=True):
            analysis_type = "basic" if "기본" in analysis_depth else "deep"
            
            # 진행 상태 표시
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text(f"🤖 {ai_provider} 모델 로딩 중...")
            progress_bar.progress(20)
            
            time.sleep(0.5)
            
            status_text.text(f"📊 데이터 분석 중... ({len(df_results)}개 종목)")
            progress_bar.progress(40)
            
            time.sleep(0.5)
            
            status_text.text(f"🔬 {'심층' if analysis_type == 'deep' else '기본'} 분석 수행 중...")
            progress_bar.progress(60)
            
            # AI 분석 실행
            if "Gemini" in ai_provider:
                result = analyze_with_gemini(df_results, analysis_type)
            else:
                result = analyze_with_openai(df_results, analysis_type)
            
            progress_bar.progress(100)
            status_text.text("✅ 분석 완료!")
            
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            
            # 결과 표시
            st.success(f"✅ {ai_provider} {'Deep Dive' if analysis_type == 'deep' else 'Basic'} 분석 완료!")
            
            # 분석 결과 컨테이너
            result_container = st.container()
            
            with result_container:
                st.markdown(result)
            
            st.markdown("---")
            
            # 다운로드 및 추가 옵션
            col_dl1, col_dl2, col_dl3 = st.columns(3)
            
            with col_dl1:
                st.download_button(
                    label="💾 Markdown 저장",
                    data=result,
                    file_name=f'{ai_provider.replace(" ", "_")}_{analysis_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md',
                    mime='text/markdown',
                    use_container_width=True
                )
            
            with col_dl2:
                # PDF 변환 (시뮬레이션)
                st.button(
                    "📄 PDF 변환",
                    help="분석 결과를 PDF로 변환 (Pro 기능)",
                    use_container_width=True,
                    disabled=True
                )
            
            with col_dl3:
                # 이메일 전송 (시뮬레이션)
                st.button(
                    "📧 이메일 전송",
                    help="분석 결과를 이메일로 전송 (Pro 기능)",
                    use_container_width=True,
                    disabled=True
                )
        
        # 안내 메시지
        st.markdown("---")
        
        # 탭으로 구성
        info_tab1, info_tab2, info_tab3 = st.tabs(["💡 사용 가이드", "🔧 실제 API 연동", "📊 분석 예시"])
        
        with info_tab1:
            st.markdown("""
            ### 💡 AI 분석 사용 가이드
            
            **1. 모델 선택**
            - **Google Gemini**: 빠른 분석, 창의적 인사이트
            - **OpenAI GPT-4**: 정량적 분석, 구조화된 전략
            
            **2. 분석 깊이**
            - **기본**: 빠른 시장 파악 (30초)
            - **Deep Dive**: 전문 투자 전략 (1-2분)
            
            **3. 활용 팁**
            - 아침: 기본 분석으로 시장 체크
            - 투자 결정 전: Deep Dive로 상세 검토
            - 정기적: 주간 Deep Dive 리포트
            """)
        
        with info_tab2:
            st.markdown("""
            ### 🔧 실제 API 연동 방법
            
            현재는 **데모 모드**입니다. 실제 AI API를 사용하려면:
            
            **1. API 키 설정**
            ```toml
            # .streamlit/secrets.toml
            GEMINI_API_KEY = "your-gemini-api-key"
            OPENAI_API_KEY = "your-openai-api-key"
            ```
            
            **2. 코드 수정**
            ```python
            # ai_helpers.py의 함수 사용
            from ai_helpers import analyze_with_gemini_real, analyze_with_openai_real
            
            # 기존 함수 대체
            if "Gemini" in ai_provider:
                result = analyze_with_gemini_real(df_results, analysis_type)
            else:
                result = analyze_with_openai_real(df_results, analysis_type)
            ```
            
            **3. 필요 패키지**
            ```bash
            pip install google-generativeai openai
            ```
            """)
        
        with info_tab3:
            st.markdown("""
            ### 📊 분석 결과 예시
            
            **기본 분석 예시:**
            ```
            🤖 Gemini AI 기본 분석
            
            시장 개요:
            현재 MAG 7+2 포트폴리오는 혼조세...
            
            Top Pick: NVDA
            - 현재가: $XXX
            - 추천 이유: AI 반도체 수요 급증
            
            투자 전략: 단기 모멘텀 전략 권장
            ```
            
            **Deep Dive 예시:**
            ```
            🔬 GPT-4 Deep Dive 분석
            
            1. 거시경제 관점
            2. 개별 종목 분석 (Top 5)
               - NVDA: BUY, Entry $XXX, Target $XXX
               - AAPL: HOLD, ...
            3. 포트폴리오 전략
               - 공격적: NVDA 40%, MSFT 30%, ...
               - 균형: 각 20%씩
               - 보수적: Cash 40% 보유
            4. 리스크 관리
            5. 매매 타이밍
            ```
            """)
        
        # 실시간 API 상태
        st.markdown("---")
        
        with st.expander("🔌 API 상태 확인", expanded=False):
            col_api1, col_api2 = st.columns(2)
            
            # Gemini API 상태
            with col_api1:
                st.markdown("**Google Gemini API**")
                try:
                    gemini_key = st.secrets.get("GEMINI_API_KEY", "")
                    if gemini_key and gemini_key != "your-gemini-api-key-here":
                        st.success("🟢 활성화됨")
                        st.caption(f"API 키: ...{gemini_key[-8:]}")
                    else:
                        st.warning("🟡 API 키 미설정")
                        st.caption("데모 모드로 작동")
                except:
                    st.error("🔴 설정 파일 없음")
                    st.caption("secrets.toml 확인 필요")
            
            # OpenAI API 상태
            with col_api2:
                st.markdown("**OpenAI API**")
                try:
                    openai_key = st.secrets.get("OPENAI_API_KEY", "")
                    if openai_key and openai_key != "your-openai-api-key-here":
                        st.success("🟢 활성화됨")
                        st.caption(f"API 키: ...{openai_key[-8:]}")
                    else:
                        st.warning("🟡 API 키 미설정")
                        st.caption("데모 모드로 작동")
                except:
                    st.error("🔴 설정 파일 없음")
                    st.caption("secrets.toml 확인 필요")
            
            st.markdown("---")
            st.info("""
            **💡 API 키 설정 방법:**
            
            `.streamlit/secrets.toml` 파일에 추가:
            ```toml
            GEMINI_API_KEY = "your-actual-key"
            OPENAI_API_KEY = "sk-..."
            ```
            """)
    
    # ==================== 페이지 3: Quant 챗봇 ====================
    else:  # 💬 Quant 챗봇
        st.markdown("### 💬 Advanced Quant Chatbot")
        
        # Quick Questions
        st.markdown("#### ⚡ Quick Questions")
        
        quick_questions = {
            "🏆 최고 추천 종목은?": "top pick",
            "💰 지금 매수하기 좋은 종목은?": "best buy",
            "⚠️ 공매도 리스크가 높은 종목은?": "short risk",
            "📊 VWAP 상태는?": "vwap status",
        }
        
        cols = st.columns(len(quick_questions))
        
        for idx, (label, query) in enumerate(quick_questions.items()):
            with cols[idx]:
                if st.button(label, use_container_width=True):
                    answer = quant_chatbot(query, df_results)
                    st.session_state.chat_history.append(("user", label))
                    st.session_state.chat_history.append(("bot", answer))
        
        st.markdown("---")
        
        # Chat Interface
        st.markdown("#### 💭 자유 질문")
        
        # 채팅 히스토리 표시
        chat_container = st.container()
        
        with chat_container:
            for sender, message in st.session_state.chat_history:
                if sender == "user":
                    st.markdown(f"**👤 You:** {message}")
                else:
                    st.markdown(f"**🤖 Bot:** {message}")
                st.markdown("---")
        
        # 입력창
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "질문을 입력하세요:",
                placeholder="예: NVDA의 현재 상태는? / 공매도 비율이 가장 낮은 종목은?",
                label_visibility="collapsed"
            )
            
            submitted = st.form_submit_button("📤 전송", use_container_width=True)
        
        if submitted and user_input:
            # 사용자 질문 추가
            st.session_state.chat_history.append(("user", user_input))
            
            # 봇 응답 생성
            with st.spinner("분석 중..."):
                bot_response = quant_chatbot(user_input, df_results)
                st.session_state.chat_history.append(("bot", bot_response))
            
            st.rerun()
        
        # 채팅 초기화
        if st.button("🗑️ 채팅 기록 지우기"):
            st.session_state.chat_history = []
            st.rerun()
        
        # 안내
        st.markdown("---")
        st.info("""
        **💡 챗봇 사용 가이드**
        
        - **Quick Questions**: 자주 묻는 질문을 빠르게 확인
        - **자유 질문**: 종목별 상세 정보, 비교 분석 등 자유롭게 질문
        
        예시 질문:
        - "NVDA와 AMD를 비교해줘"
        - "공매도 비율이 5% 미만인 종목은?"
        - "VWAP 위에서 거래되는 종목들은?"
        """)

if __name__ == "__main__":
    main()
