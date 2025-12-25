import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from io import StringIO
import time
import google.generativeai as genai  # 추가!

warnings.filterwarnings('ignore')

# Gemini API 설정
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    GEMINI_ENABLED = True
except:
    GEMINI_ENABLED = False
    gemini_model = None

# AI 분석 함수
def analyze_with_gemini(analysis_type, market_data, short_data, technical_data):
    if not GEMINI_ENABLED:
        return "❌ API 설정 필요"
    
    prompts = {
        "종합분석": f"""MAG 7+2 종합 분석:
시장:{market_data}
공매도:{short_data}
기술:{technical_data}

1.시장상황 2.Top3추천 3.주의종목 4.전망""",
        
        "공매도분석": f"""공매도 분석:
{short_data}
시장:{market_data}

1.공매도상황 2.Squeeze가능성 3.안전종목 4.전략""",
        
        "기술적분석": f"""기술적 분석:
{technical_data}
시장:{market_data}

1.기술상황 2.모멘텀종목 3.타이밍 4.손익기준""",
        
        "비교분석": f"""비교분석:
시장:{market_data}
기술:{technical_data}
공매도:{short_data}

1.빅테크vs비트코인 2.섹터매력도 3.포트폴리오 4.리스크"""
    }
    
    try:
        response = gemini_model.generate_content(prompts.get(analysis_type, prompts["종합분석"]))
        return response.text
    except Exception as e:
        return f"오류: {str(e)}"

def get_market_summary(df_results):
    summary = ""
    for idx, row in df_results.head(9).iterrows():
        summary += f"{row['Ticker']}: ${row['Current_Price']:.2f} ({row['Quarter_Return_%']:+.1f}%) "
        summary += f"VWAP{row['Price_vs_VWAP_%']:+.1f}%\n"
    return summary

def get_short_summary(df_results):
    summary = ""
    for idx, row in df_results.head(9).iterrows():
        summary += f"{row['Ticker']}: 공매도{row.get('short_percent_float', 0):.1f}% "
        summary += f"청산{row.get('short_ratio_days', 0):.1f}일\n"
    return summary

def get_technical_summary(df_results):
    summary = ""
    for idx, row in df_results.head(9).iterrows():
        vwap = 'VWAP위✅' if row['Is_Above_VWAP'] else 'VWAP아래⚠️'
        summary += f"{row['Ticker']}: {vwap} 점수{row['Buy_Signal_Score']}/100\n"
    return summary
# ==================== 여기까지 추가! ====================

# ==================== 페이지 설정 ====================
st.set_page_config(
    page_title="MAG 7+2 종합 분석 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 로그인 시스템 ====================
def check_password():
    """비밀번호 확인 및 로그인 상태 관리"""
    if st.session_state.get('password_correct', False):
        return True
    
    st.title("🔒 MAG 7+2 퀀트 대시보드 로그인")
    st.markdown("### Magnificent Seven + Bitcoin Exposure 종합 분석")
    
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

# ==================== 로그아웃 버튼 ====================
with st.sidebar:
    st.success(f"✅ 로그인 성공!")
    if st.button("🚪 로그아웃"):
        st.session_state['password_correct'] = False
        st.rerun()
 # ==================== 여기에 추가! ====================
    st.markdown("---")
    st.markdown("### 🤖 AI 분석 상태")
    if GEMINI_ENABLED:
        st.success("✅ Gemini AI 활성화")
        st.info("**모델:** Gemini 2.5 Flash\n**할당량:** 일일 1,500 무료")
    else:
        st.error("❌ Gemini AI 비활성화")
        st.warning("secrets.toml에 GEMINI_API_KEY 추가")
    # ==================== 여기까지 추가! ====================
# ==================== MAG 7+2 정의 ====================
MAG7_STOCKS = {
    'AAPL': {'name': 'Apple Inc.', 'description': '아이폰, 생태계, 온디바이스 AI', 'sector': 'Technology', 'industry': 'Consumer Electronics'},
    'MSFT': {'name': 'Microsoft Corporation', 'description': '클라우드(Azure), 생성형 AI (OpenAI 대주주)', 'sector': 'Technology', 'industry': 'Software'},
    'GOOGL': {'name': 'Alphabet Inc.', 'description': '구글 검색, 유튜브, AI (Gemini)', 'sector': 'Communication Services', 'industry': 'Internet Content & Information'},
    'AMZN': {'name': 'Amazon.com Inc.', 'description': '전자상거래, 클라우드(AWS) 1위', 'sector': 'Consumer Cyclical', 'industry': 'Internet Retail'},
    'NVDA': {'name': 'NVIDIA Corporation', 'description': 'AI 반도체(GPU) 독점적 지배자', 'sector': 'Technology', 'industry': 'Semiconductors'},
    'META': {'name': 'Meta Platforms Inc.', 'description': '페이스북, 인스타그램, AI(Llama)', 'sector': 'Communication Services', 'industry': 'Internet Content & Information'},
    'TSLA': {'name': 'Tesla Inc.', 'description': '전기차, 자율주행, 로봇', 'sector': 'Consumer Cyclical', 'industry': 'Auto Manufacturers'},
    'COIN': {'name': 'Coinbase Global Inc.', 'description': '미국 최대 암호화폐 거래소, 비트코인 직접 노출', 'sector': 'Financial Services', 'industry': 'Cryptocurrency Exchange'},
    'IBIT': {'name': 'iShares Bitcoin Trust ETF', 'description': 'BlackRock 비트코인 현물 ETF, 순수 BTC 노출', 'sector': 'ETF', 'industry': 'Bitcoin Spot ETF'}
}

# ==================== 유틸리티 함수 ====================
@st.cache_data(ttl=3600)
def get_current_quarter_start():
    now = datetime.now()
    quarter = (now.month - 1) // 3
    quarter_start_month = quarter * 3 + 1
    return datetime(now.year, quarter_start_month, 1)

def calculate_anchored_vwap(df):
    df = df.copy()
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['TP_Volume'] = df['Typical_Price'] * df['Volume']
    df['Cumulative_TP_Volume'] = df['TP_Volume'].cumsum()
    df['Cumulative_Volume'] = df['Volume'].cumsum()
    df['Anchored_VWAP'] = df['Cumulative_TP_Volume'] / df['Cumulative_Volume']
    return df

@st.cache_data(ttl=3600)
def get_finra_short_volume_csv(ticker, days_back=10):
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
            return {
                'ticker': ticker,
                'latest_date': df_short.iloc[0]['date'],
                'latest_short_ratio': df_short.iloc[0]['short_ratio'],
                'avg_short_ratio_10d': round(df_short['short_ratio'].mean(), 2),
                'data_points': len(df_short),
                'historical_data': df_short
            }
        return None
    except:
        return None

@st.cache_data(ttl=3600)
def get_short_interest_from_yfinance(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        short_data = {
            'ticker': ticker,
            'short_ratio': info.get('shortRatio', 0),
            'short_percent_float': info.get('shortPercentOfFloat', 0) * 100 if info.get('shortPercentOfFloat') else 0,
            'shares_short': info.get('sharesShort', 0),
            'shares_short_prior_month': info.get('sharesShortPriorMonth', 0),
        }
        
        if short_data['shares_short_prior_month'] > 0:
            short_data['short_change_pct'] = ((short_data['shares_short'] - short_data['shares_short_prior_month']) / 
                                               short_data['shares_short_prior_month'] * 100)
        else:
            short_data['short_change_pct'] = 0
        return short_data
    except:
        return None

@st.cache_data(ttl=3600)
def get_comprehensive_short_data(ticker):
    yf_data = get_short_interest_from_yfinance(ticker)
    finra_data = get_finra_short_volume_csv(ticker, days_back=60)
    
    combined_data = {
        'ticker': ticker, 'short_ratio_days': 0, 'short_percent_float': 0,
        'shares_short_millions': 0, 'short_change_pct': 0, 'daily_short_ratio': 0,
        'avg_daily_short_ratio_10d': 0, 'finra_latest_date': 'N/A',
        'finra_historical': None, 'data_source': []
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
        combined_data['daily_short_ratio'] = finra_data['latest_short_ratio']
        combined_data['avg_daily_short_ratio_10d'] = finra_data['avg_short_ratio_10d']
        combined_data['finra_latest_date'] = finra_data.get('latest_date', 'N/A')
        combined_data['finra_historical'] = finra_data.get('historical_data')
        combined_data['data_source'].append(f"FINRA ({finra_data.get('data_points', 0)}일)")
    
    combined_data['data_source'] = ' + '.join(combined_data['data_source']) if combined_data['data_source'] else 'N/A'
    return combined_data

@st.cache_data(ttl=3600)
def get_quarterly_vwap_analysis(ticker):
    try:
        quarter_start = get_current_quarter_start()
        end_date = datetime.now()
        quarter_num = (quarter_start.month - 1) // 3 + 1

        stock = yf.Ticker(ticker)
        df = stock.history(start=quarter_start, end=end_date)

        if df.empty or len(df) < 5:
            return None

        df = calculate_anchored_vwap(df)
        current_price = df['Close'].iloc[-1]
        current_vwap = df['Anchored_VWAP'].iloc[-1]
        above_vwap_ratio = (df['Close'] > df['Anchored_VWAP']).sum() / len(df) * 100
        
        recent_20 = df['Close'].tail(min(20, len(df)))
        uptrend_strength = (recent_20.diff() > 0).sum() / len(recent_20) * 100 if len(recent_20) > 1 else 50
        
        recent_volume = df['Volume'].tail(5).mean()
        avg_volume = df['Volume'].mean()
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1

        info = stock.info
        quarter_start_price = df['Close'].iloc[0]
        quarter_return = ((current_price - quarter_start_price) / quarter_start_price * 100)

        return {
            'Ticker': ticker, 'Company': MAG7_STOCKS[ticker]['name'],
            'Description': MAG7_STOCKS[ticker]['description'],
            'Current_Price': round(current_price, 2),
            'Anchored_VWAP': round(current_vwap, 2),
            'Quarter_Return_%': round(quarter_return, 2),
            'Price_vs_VWAP_%': round((current_price - current_vwap) / current_vwap * 100, 2),
            'Above_VWAP_Days_%': round(above_vwap_ratio, 1),
            'Uptrend_Strength_%': round(uptrend_strength, 1),
            'Volume_Ratio': round(volume_ratio, 2),
            'Is_Above_VWAP': current_price > current_vwap,
            'Market_Cap': info.get('marketCap', 0),
        }
    except Exception as e:
        return None

def calculate_buy_score(row):
    score = 0
    if row['Is_Above_VWAP']: score += 30
    price_diff = row['Price_vs_VWAP_%']
    if 0 < price_diff <= 5: score += 20
    elif 5 < price_diff <= 10: score += 10
    elif price_diff > 10: score += 5
    if row['Above_VWAP_Days_%'] >= 80: score += 20
    elif row['Above_VWAP_Days_%'] >= 60: score += 15
    if row['Uptrend_Strength_%'] >= 60: score += 15
    elif row['Uptrend_Strength_%'] >= 50: score += 10
    if row['Volume_Ratio'] >= 1.2: score += 15
    elif row['Volume_Ratio'] >= 1.0: score += 10
    return min(score, 100)

def calculate_short_score(row):
    short_pct = row.get('short_percent_float', 0)
    if short_pct < 5: return 20
    elif short_pct < 10: return 15
    elif short_pct < 20: return 10
    else: return 5

# ==================== 메인 앱 ====================
st.title("🌟 MAGNIFICENT SEVEN + BITCOIN "Short" EXPOSURE 종합 분석")
st.markdown(f"**데이터 수집 시간:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (KST)")

# 사이드바
with st.sidebar:
    st.header("⚙️ 설정")
    selected_tickers = st.multiselect(
        "분석할 종목 선택",
        list(MAG7_STOCKS.keys()),
        default=list(MAG7_STOCKS.keys())
    )
    
    st.markdown("---")
    st.subheader("📊 고급 차트 옵션")
    show_timeseries = st.checkbox("시계열 분석 차트", value=True)
    show_correlation = st.checkbox("상관관계 분석", value=True)
    show_volatility = st.checkbox("변동성 분석", value=True)
    
    st.markdown("---")
    if st.button("🔄 데이터 새로고침", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# 탭 생성
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 종합 대시보드", 
    "🔴 공매도 기본 분석", 
    "📈 공매도 시계열 분석",
    "🎯 고급 분석",
    "📋 데이터",
    "🤖 AI 분석"  # ← 추가!
])
# 데이터 수집
with st.spinner("데이터 수집 중..."):
    results = []
    short_data_list = []
    
    progress_bar = st.progress(0)
    for idx, ticker in enumerate(selected_tickers):
        result = get_quarterly_vwap_analysis(ticker)
        if result:
            results.append(result)
        
        short_data = get_comprehensive_short_data(ticker)
        if short_data:
            short_data_list.append(short_data)
        
        progress_bar.progress((idx + 1) / len(selected_tickers))
    progress_bar.empty()

if not results:
    st.error("데이터를 수집하지 못했습니다.")
    st.stop()

df_results = pd.DataFrame(results)
df_short = pd.DataFrame(short_data_list)
df_results = df_results.merge(df_short, left_on='Ticker', right_on='ticker', how='left')
df_results['Market_Cap_Trillion'] = (df_results['Market_Cap'] / 1e12).round(3)
df_results['Buy_Signal_Score'] = df_results.apply(calculate_buy_score, axis=1)
df_results['Short_Score'] = df_results.apply(calculate_short_score, axis=1)
df_results['Total_Investment_Score'] = df_results['Buy_Signal_Score'] + df_results['Short_Score']
df_results = df_results.sort_values('Total_Investment_Score', ascending=False)

# TAB 1: 종합 대시보드
with tab1:
    st.header("📊 종합 투자 순위")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🥇 1위 종목", df_results.iloc[0]['Ticker'], f"{df_results.iloc[0]['Total_Investment_Score']:.0f}/120점")
    with col2:
        above_vwap = len(df_results[df_results['Is_Above_VWAP'] == True])
        st.metric("✅ VWAP 위", f"{above_vwap}개")
    with col3:
        st.metric("📈 평균 분기 수익률", f"{df_results['Quarter_Return_%'].mean():+.1f}%")
    with col4:
        low_short = len(df_results[df_results['short_percent_float'] < 5])
        st.metric("🟢 낮은 공매도", f"{low_short}개")
    
    st.markdown("---")
    
    # 종합 점수 비교 차트
    st.subheader("🏆 종합 투자 점수 비교")
    st.caption("💡 **기술적 분석(VWAP)과 공매도 분석을 결합한 종합 평가** - 종합 점수가 높을수록 투자 매력도 높음")
    fig_total_score = make_subplots(
        rows=1, cols=2,
        subplot_titles=('기술적 분석 점수', '종합 투자 점수 (기술적 + 공매도)'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    fig_total_score.add_trace(
        go.Bar(
            y=df_results['Ticker'],
            x=df_results['Buy_Signal_Score'],
            orientation='h',
            name='기술적 점수',
            marker_color='#2196F3',
            text=df_results['Buy_Signal_Score'],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>기술적 점수: %{x}/100<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig_total_score.add_trace(
        go.Bar(
            y=df_results['Ticker'],
            x=df_results['Total_Investment_Score'],
            orientation='h',
            name='종합 점수',
            marker_color='#4CAF50',
            text=df_results['Total_Investment_Score'],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>종합 점수: %{x}/120<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig_total_score.update_xaxes(title_text="점수", row=1, col=1)
    fig_total_score.update_xaxes(title_text="점수", row=1, col=2)
    fig_total_score.update_yaxes(title_text="종목", row=1, col=1)
    
    fig_total_score.update_layout(
        height=500,
        showlegend=False,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_total_score, use_container_width=True)
    
    st.markdown("---")
    
    # 공매도 vs 분기 수익률 산점도
    st.subheader("📊 공매도 비율 vs 분기 수익률")
    st.caption("""
    💡 **공매도와 실제 수익률의 관계 분석**
    - 왼쪽 상단(낮은 공매도 + 높은 수익률): 최적 투자 대상
    - 오른쪽 하단(높은 공매도 + 낮은 수익률): 위험 종목
    - 버블 크기는 시가총액, 색상은 종합 투자 점수
    """)
    fig_scatter_performance = px.scatter(
        df_results,
        x='short_percent_float',
        y='Quarter_Return_%',
        size='Market_Cap_Trillion',
        color='Total_Investment_Score',
        hover_data=['Ticker', 'Company'],
        text='Ticker',
        color_continuous_scale='RdYlGn',
        labels={
            'short_percent_float': '공매도 비율 (%)',
            'Quarter_Return_%': '분기 수익률 (%)',
            'Total_Investment_Score': '종합 점수'
        }
    )
    
    fig_scatter_performance.update_traces(textposition='top center', textfont_size=12)
    fig_scatter_performance.update_layout(height=500)
    st.plotly_chart(fig_scatter_performance, use_container_width=True)
    
    st.markdown("---")
    
    # 상세 순위표
    for idx, row in df_results.iterrows():
        rank = df_results.index.get_loc(idx) + 1
        with st.expander(f"**#{rank} {row['Ticker']} - {row['Company'][:30]}**", expanded=(rank <= 3)):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**🎯 {row['Description']}**")
                st.markdown(f"💰 시가총액: ${row['Market_Cap_Trillion']:.2f}T")
                st.markdown(f"📈 현재가: ${row['Current_Price']:.2f} | VWAP: ${row['Anchored_VWAP']:.2f}")
                st.markdown(f"📊 VWAP 대비: {row['Price_vs_VWAP_%']:+.2f}% | 분기수익률: {row['Quarter_Return_%']:+.2f}%")
                st.markdown(f"🔴 공매도 비율: {row['short_percent_float']:.2f}% | 커버 소요일: {row['short_ratio_days']:.1f}일")
            with col2:
                score = row['Total_Investment_Score']
                signal = "최우선 매수" if score >= 90 else "강력 매수" if score >= 75 else "눌림목 대기"
                st.metric("종합 점수", f"{score:.0f}/120", signal)
                st.progress(score / 120)

# TAB 2: 공매도 기본 분석
with tab2:
    st.header("🔴 공매도 기본 분석")
    
    # 비교표
    st.subheader("📋 Yahoo Finance vs FINRA 상세 비교")
    
    # 컬럼 설명 - Expander 형태로 변경
    with st.expander("📖 컬럼 설명 보기", expanded=False):
        st.markdown("""
        **Yahoo Finance 데이터 (공매도 잔고 - 월 2회 업데이트):**
        
        - **YF 공매도%**: 유통주식(Float) 대비 공매도 비율. 5% 미만이 건강
        - **YF 청산일**: Days to Cover. 공매도 잔고를 일평균 거래량으로 나눈 값
        - **YF 공매도주식(M)**: 현재 공매도된 총 주식 수 (백만 주)
        - **YF 전월대비%**: 전월 대비 공매도 증감률. (+)는 증가, (-)는 감소
        
        **FINRA 데이터 (일일 공매도 거래량 - 매일 업데이트):**
        
        - **FINRA 일평균%**: 최근 거래일의 공매도 거래 비율
        - **FINRA 10일평균%**: 최근 10거래일 평균 공매도 비율
        - **FINRA 날짜**: 데이터 수집 날짜
        
        💡 **중요**: YF는 "잔고"(누적), FINRA는 "거래량"(일일)으로 서로 다른 지표입니다.
        """)
    
    comparison_df = df_results[['Ticker', 'Company', 'short_percent_float', 'short_ratio_days', 
                                  'shares_short_millions', 'short_change_pct', 'daily_short_ratio', 
                                  'avg_daily_short_ratio_10d', 'finra_latest_date']].copy()
    
    # 컬럼명 한글화
    comparison_df.columns = ['티커', '회사명', 'YF 공매도%', 'YF 청산일', 'YF 공매도주식(M)', 
                              'YF 전월대비%', 'FINRA 일평균%', 'FINRA 10일평균%', 'FINRA 날짜']
    
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # 상세 비교 차트 - Expander 형태로 변경
    st.subheader("📊 상세 비교 차트")
    
    with st.expander("📖 차트 해석 가이드", expanded=False):
        st.markdown("""
        ## 🔴 MAG 7+2 공매도 분석 지표 종합 정리
    
        ### 1. 📊 차트 A: YF Short % of Float (유통주식 대비 공매도 비율)
    
        **📌 지표 설명**  
        유통 가능한 주식 수 대비 공매도된 주식 수의 비율로, 시장에서 해당 주식에 대한 누적된 약세 베팅의 강도를 나타냅니다.
    
        **✅ 투자 원칙**  
        낮을수록 좋음. 5% 미만 권장.
    
        **🎨 색상 기준**
        - 🟢 **초록색 (2% 미만)**: 매우 건강한 상태, 시장의 강한 신뢰
        - 🟠 **주황색 (2% ~ 5%)**: 정상 범위, 일반적인 수준의 공매도
        - 🔴 **빨간색 (5% 이상)**: 공매도 압력 존재, 주의 필요
        
        **💡 시사점**  
        비율이 높을수록 주식에 대한 약세 심리가 강함을 의미하며, 주가 하방 압력으로 작용합니다. 
        하지만 동시에 청산을 위한 잠재적인 매수 압력(환매수)이 될 수도 있습니다.
        
        ---
        
        ### 2. 📅 차트 B: Days to Cover (청산 소요일)
        
        **📌 지표 설명**  
        **공매도 잔고(Short Interest)**를 일평균 거래량으로 나눈 값입니다.  
        공매도 투자자들이 모든 포지션을 청산(환매수)하는 데 며칠이 걸리는지 나타냅니다.
        
        **✅ 투자 원칙**  
        짧을수록 좋음. 3일 이상이면 변동성 증가 가능성.
        
        **🎨 색상 기준**
        - 🟢 **초록색 (2일 미만)**: 빠른 청산 가능, 안정적
        - 🟠 **주황색 (2일 ~ 3일)**: 보통 수준
        - 🔴 **빨간색 (3일 이상)**: 청산에 시간이 오래 걸리며, 숏 스퀴즈 가능성 높음
        
        **💡 시사점**  
        3일 이상일 경우, 주가 상승 시 대규모 환매수가 발생하여 변동성을 극대화시킬 수 있습니다.  
        (예: NVDA가 가장 높은 청산 소요일을 보여 잠재적인 숏 스퀴즈 위험 시사)
        
        ---
        
        ### 3. 📊 차트 C: Shares Short (공매도 주식 수 - 누적)
        
        **📌 지표 설명**  
        현재 공매도된 총 주식 수를 나타내는 절대적인 공매도 규모 지표입니다. (백만 주 단위)
        
        **✅ 투자 원칙**  
        클수록 잠재적인 변동성(환매수 또는 추가 매도)이 증가할 수 있습니다.
        
        **🎨 시각화**  
        색상이 진할수록 공매도 규모가 큼을 의미합니다.
        
        **💡 시사점**  
        규모가 크다는 것은 해당 종목이 공매도 투자자들의 핵심 표적 중 하나임을 시사합니다.  
        (예: NVDA가 가장 큰 규모)
        
        ---
        
        ### 4. 📉 차트 D: 전월 대비 공매도 변화율 (MoM Change)
        
        **📌 지표 설명**  
        전월 공매도 잔고 대비 현재 공매도 잔고의 증감률입니다.
        
        **✅ 투자 원칙**  
        **감소(-)**가 **증가(+)**보다 강세 신호. 급격한 증가는 주의 신호.
        
        **🎨 색상 기준**
        - 🔴 **빨간색 (+) 증가**: 공매도 투자자들이 포지션을 늘려 약세 심리 강화  
          (예: META, AMZN, MSFT, COIN)
        - 🟢 **초록색 (-) 감소**: 공매도 포지션이 줄어들어 약세 심리 완화  
          (예: GOOGL, NVDA, TSLA, AAPL)
        
        **💡 시사점**  
        공매도 투자자들의 최신 심리 변화 추세를 파악하는 데 유용합니다.
        
        ---
        
        ### 5. 📊 차트 E: FINRA Daily Short % (FINRA 일일 공매도 비율)
        
        **📌 지표 설명**  
        일일 총 거래량 중 공매도 거래량이 차지하는 비율입니다.  
        매일의 단기적인 공매도 활동성을 나타냅니다.
        
        **✅ 투자 원칙**  
        30% ~ 40%는 정상 수준. 이보다 높으면 단기 하방 압력이 강함을 시사.
        
        **🎨 색상 기준**
        - 🟢 **초록색 (35% 미만)**: 낮은 공매도 거래 수준
        - 🟠 **주황색 (35% ~ 45%)**: 정상 범위
        - 🔴 **빨간색 (45% 이상)**: 공매도 거래가 매우 활발, 단기 하방 압력 강함
        
        **💡 시사점**  
        YF 지표(잔고, 누적)와 달리, 이 지표는 시장의 **단기적인 수급 불균형**을 판단하는 데 활용됩니다.
        
        ---
        
        ### 6. 📊 차트 F: FINRA 10일 평균 vs 최근일
        
        **📌 지표 설명**  
        최근 10일 평균 공매도 비율과 최근일 공매도 비율을 비교하여 단기 추세 변화를 확인합니다.
        
        **🎨 시각화**
        - 🔵 **하늘색**: 10일 이동평균 공매도 비율
        - 🔷 **진한 파란색**: 최근일 공매도 비율
        
        **💡 투자 시사점**
        - **최근일 > 10일 평균**: 단기적으로 공매도 활동이 **증가 추세** (약세 심리 강화)
        - **최근일 < 10일 평균**: 단기적으로 공매도 활동이 **감소 추세** (강세 심리 강화)
        
        **🎯 활용**  
        단기적인 공매도 활동의 모멘텀 변화를 파악하는 데 유용합니다.
      
        """)
    


    
    col1, col2 = st.columns(2)
    
    with col1:
        # 차트 A: Short % of Float
        st.markdown("##### YF Short % of Float")
        st.caption("💡 **유통주식(Float) 대비 공매도 비율** - 낮을수록 좋음 (5% 미만 권장)")
        fig_a = go.Figure()
        colors = ['green' if x < 2 else 'orange' if x < 5 else 'red' for x in df_results['short_percent_float']]
        fig_a.add_trace(go.Bar(
            x=df_results['Ticker'], 
            y=df_results['short_percent_float'],
            marker=dict(color=colors), 
            text=df_results['short_percent_float'].round(2),
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>공매도 비율: %{y:.2f}%<extra></extra>'
        ))
        fig_a.add_hline(y=2, line_dash="dash", line_color="green", annotation_text="매우 건강 (2%)")
        fig_a.add_hline(y=5, line_dash="dash", line_color="orange", annotation_text="건강 (5%)")
        fig_a.update_layout(height=400, template='plotly_white', showlegend=False)
        st.plotly_chart(fig_a, use_container_width=True)
    
    with col2:
        # 차트 B: Days to Cover
        st.markdown("##### Days to Cover")
        st.caption("💡 **공매도 청산 소요일** - 공매도 잔고를 일평균 거래량으로 나눈 값. 3일 이상이면 Short Squeeze 가능")
        fig_b = go.Figure()
        colors_days = ['green' if x < 2 else 'orange' if x < 3 else 'red' for x in df_results['short_ratio_days']]
        fig_b.add_trace(go.Bar(
            x=df_results['Ticker'], 
            y=df_results['short_ratio_days'],
            marker=dict(color=colors_days), 
            text=df_results['short_ratio_days'].round(2),
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>청산 소요일: %{y:.2f}일<extra></extra>'
        ))
        fig_b.add_hline(y=2, line_dash="dash", line_color="green", annotation_text="빠른 청산")
        fig_b.add_hline(y=3, line_dash="dash", line_color="red", annotation_text="Squeeze 가능")
        fig_b.update_layout(height=400, template='plotly_white', showlegend=False)
        st.plotly_chart(fig_b, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        # 차트 C: Shares Short
        st.markdown("##### Shares Short (백만 주)")
        st.caption("💡 **현재 공매도된 총 주식 수** - 절대적인 공매도 규모를 나타냄. 클수록 변동성 증가 가능")
        fig_c = go.Figure()
        fig_c.add_trace(go.Bar(
            x=df_results['Ticker'],
            y=df_results['shares_short_millions'],
            marker=dict(
                color=df_results['shares_short_millions'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="M")
            ),
            text=df_results['shares_short_millions'].round(1),
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>공매도 주식: %{y:.1f}M<extra></extra>'
        ))
        fig_c.update_layout(height=400, template='plotly_white', showlegend=False)
        st.plotly_chart(fig_c, use_container_width=True)
    
    with col4:
        # 차트 D: MoM Change
        st.markdown("##### 전월 대비 공매도 변화율")
        st.caption("💡 **전월 대비 공매도 증감률** - 빨강(+)은 공매도 증가(약세 신호), 초록(-)은 감소(강세 신호)")
        fig_d = go.Figure()
        colors_change = ['red' if x > 0 else 'green' for x in df_results['short_change_pct']]
        fig_d.add_trace(go.Bar(
            x=df_results['Ticker'], 
            y=df_results['short_change_pct'],
            marker=dict(color=colors_change), 
            text=[f"{x:+.1f}%" for x in df_results['short_change_pct']],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>전월 대비: %{y:+.1f}%<extra></extra>'
        ))
        fig_d.add_hline(y=0, line_dash="solid", line_color="black", line_width=2)
        fig_d.update_layout(height=400, template='plotly_white', showlegend=False)
        st.plotly_chart(fig_d, use_container_width=True)
    
    st.markdown("---")
    
    col5, col6 = st.columns(2)
    
    with col5:
        # 차트 E: FINRA Daily
        st.markdown("##### FINRA Daily Short %")
        st.caption("💡 **최근 거래일의 공매도 거래 비율** - 전체 거래량 중 공매도가 차지하는 비중. 30-40%는 정상")
        fig_e = go.Figure()
        colors_finra = ['green' if x < 35 else 'orange' if x < 45 else 'red' for x in df_results['daily_short_ratio']]
        fig_e.add_trace(go.Bar(
            x=df_results['Ticker'], 
            y=df_results['daily_short_ratio'],
            marker=dict(color=colors_finra), 
            text=df_results['daily_short_ratio'].round(1),
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>일일 공매도: %{y:.1f}%<extra></extra>'
        ))
        fig_e.add_hline(y=35, line_dash="dash", line_color="green", annotation_text="낮음 (35%)")
        fig_e.add_hline(y=45, line_dash="dash", line_color="orange", annotation_text="보통 (45%)")
        fig_e.update_layout(height=400, template='plotly_white', showlegend=False)
        st.plotly_chart(fig_e, use_container_width=True)
    
    with col6:
        # 차트 F: FINRA 10일 평균 vs 최근일
        st.markdown("##### FINRA 10일 평균 vs 최근일")
        st.caption("💡 **최근 추세 확인** - 최근일이 10일 평균보다 높으면 공매도 증가 추세, 낮으면 감소 추세")
        fig_f = go.Figure()
        
        fig_f.add_trace(go.Bar(
            x=df_results['Ticker'],
            y=df_results['avg_daily_short_ratio_10d'],
            name='10일 평균',
            marker_color='lightblue',
            text=df_results['avg_daily_short_ratio_10d'].round(1),
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>10일 평균: %{y:.1f}%<extra></extra>'
        ))
        
        fig_f.add_trace(go.Bar(
            x=df_results['Ticker'],
            y=df_results['daily_short_ratio'],
            name='최근일',
            marker_color='darkblue',
            text=df_results['daily_short_ratio'].round(1),
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>최근일: %{y:.1f}%<extra></extra>'
        ))
        
        fig_f.update_layout(
            height=400, 
            template='plotly_white',
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_f, use_container_width=True)

# TAB 3: 공매도 시계열 분석
with tab3:
    st.header("📈 공매도 시계열 분석 (60일)")
    
    if show_timeseries:
        # 시계열 데이터 준비
        timeseries_data = {}
        for ticker in selected_tickers:
            if ticker in df_results['Ticker'].values:
                idx = df_results[df_results['Ticker'] == ticker].index[0]
                hist_data = df_results.loc[idx, 'finra_historical']
                if hist_data is not None and not hist_data.empty:
                    timeseries_data[ticker] = hist_data
        
        if timeseries_data:
            # 차트 A: 전체 종목 추세 비교
            st.subheader("📊 전체 종목 공매도 비율 추세")
            st.caption("💡 **60일간의 일일 공매도 거래 비율 변화** - 추세선이 상승하면 공매도 압력 증가, 하락하면 감소")
            
            fig_ts_all = go.Figure()
            colors_ts = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#E74C3C', '#3498DB']
            
            for idx, (ticker, df_ts) in enumerate(timeseries_data.items()):
                df_ts_sorted = df_ts.sort_values('date')
                
                fig_ts_all.add_trace(go.Scatter(
                    x=pd.to_datetime(df_ts_sorted['date']),
                    y=df_ts_sorted['short_ratio'],
                    mode='lines+markers',
                    name=ticker,
                    line=dict(width=2.5, color=colors_ts[idx % len(colors_ts)]),
                    marker=dict(size=6),
                    hovertemplate='<b>%{fullData.name}</b><br>날짜: %{x|%Y-%m-%d}<br>공매도 비율: %{y:.1f}%<extra></extra>'
                ))
            
            fig_ts_all.add_hline(y=40, line_dash="dash", line_color="gray", annotation_text="정상 범위 (40%)")
            fig_ts_all.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="약세 압력 (50%)")
            
            fig_ts_all.update_layout(
                xaxis_title='날짜',
                yaxis_title='공매도 거래 비율 (%)',
                hovermode='x unified',
                height=600,
                template='plotly_white',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_ts_all, use_container_width=True)
            
            st.markdown("---")
            
            # 차트 B: 개별 종목 상세 시계열 (서브플롯)
            st.subheader("📊 개별 종목 상세 시계열 (7일 이동평균 포함)")
            st.caption("💡 **종목별 공매도 추세 분석** - 빨간 점선(7일 이동평균)이 상승하면 공매도 압력 증가 추세")
            
            # 3x3 그리드
            n_tickers = len(selected_tickers)
            n_cols = 3
            n_rows = (n_tickers + n_cols - 1) // n_cols
            
            fig_ts_individual = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=[ticker for ticker in selected_tickers],
                vertical_spacing=0.10,
                horizontal_spacing=0.08
            )
            
            for idx, ticker in enumerate(selected_tickers):
                row = idx // n_cols + 1
                col = idx % n_cols + 1
                
                if ticker in timeseries_data:
                    df_ts = timeseries_data[ticker]
                    df_ts_sorted = df_ts.sort_values('date')
                    
                    # 공매도 비율 라인
                    fig_ts_individual.add_trace(
                        go.Scatter(
                            x=pd.to_datetime(df_ts_sorted['date']),
                            y=df_ts_sorted['short_ratio'],
                            mode='lines',
                            name=ticker,
                            line=dict(width=2, color=colors_ts[idx % len(colors_ts)]),
                            fill='tozeroy',
                            fillcolor=f'rgba({int(colors_ts[idx % len(colors_ts)][1:3], 16)}, {int(colors_ts[idx % len(colors_ts)][3:5], 16)}, {int(colors_ts[idx % len(colors_ts)][5:7], 16)}, 0.2)',
                            showlegend=False,
                            hovertemplate='%{y:.1f}%<extra></extra>'
                        ),
                        row=row, col=col
                    )
                    
                    # 이동평균선 (7일)
                    if len(df_ts_sorted) >= 7:
                        ma7 = df_ts_sorted['short_ratio'].rolling(window=7).mean()
                        fig_ts_individual.add_trace(
                            go.Scatter(
                                x=pd.to_datetime(df_ts_sorted['date']),
                                y=ma7,
                                mode='lines',
                                name=f'{ticker} MA7',
                                line=dict(width=1.5, color='red', dash='dash'),
                                showlegend=False,
                                hovertemplate='MA7: %{y:.1f}%<extra></extra>'
                            ),
                            row=row, col=col
                        )
            
            fig_ts_individual.update_xaxes(title_text="날짜")
            fig_ts_individual.update_yaxes(title_text="공매도 비율 (%)")
            
            fig_ts_individual.update_layout(
                height=300 * n_rows,
                template='plotly_white',
                showlegend=False
            )
            
            st.plotly_chart(fig_ts_individual, use_container_width=True)
            
            st.markdown("---")
            
            # 차트 C: 거래량 vs 공매도 비율
            if show_correlation:
                st.subheader("📊 거래량 vs 공매도 비율 관계 (최근 30일)")
                st.caption("💡 **거래량이 많을 때 공매도도 증가하는지 확인** - 버블 크기는 공매도 거래량을 나타냄")
                
                all_ts_data = []
                for ticker, df_ts in timeseries_data.items():
                    df_temp = df_ts.copy()
                    df_temp['ticker'] = ticker
                    all_ts_data.append(df_temp)
                
                if all_ts_data:
                    df_all_ts = pd.concat(all_ts_data, ignore_index=True)
                    df_all_ts['date'] = pd.to_datetime(df_all_ts['date'])
                    
                    recent_date = df_all_ts['date'].max() - timedelta(days=30)
                    df_recent = df_all_ts[df_all_ts['date'] >= recent_date]
                    
                    fig_vol_short = px.scatter(
                        df_recent,
                        x='total_volume',
                        y='short_ratio',
                        color='ticker',
                        size='short_volume',
                        hover_data=['date'],
                        labels={
                            'total_volume': '전체 거래량',
                            'short_ratio': '공매도 비율 (%)',
                            'ticker': '종목'
                        },
                        color_discrete_sequence=colors_ts
                    )
                    
                    fig_vol_short.update_layout(height=600, template='plotly_white')
                    st.plotly_chart(fig_vol_short, use_container_width=True)
            
            st.markdown("---")
            
            # 차트 D: 변동성 분석
            if show_volatility:
                st.subheader("📊 공매도 비율 변동성 분석 (Box Plot)")
                st.caption("💡 **공매도 비율의 안정성 확인** - 박스가 작을수록 변동성이 낮아 안정적. 수염이 길면 극단값 존재")
                
                fig_volatility = go.Figure()
                
                for idx, ticker in enumerate(selected_tickers):
                    if ticker in timeseries_data:
                        df_ts = timeseries_data[ticker]
                        
                        fig_volatility.add_trace(go.Box(
                            y=df_ts['short_ratio'],
                            name=ticker,
                            marker_color=colors_ts[idx % len(colors_ts)],
                            boxmean='sd'
                        ))
                
                fig_volatility.update_layout(
                    yaxis_title='공매도 비율 (%)',
                    xaxis_title='종목',
                    height=600,
                    template='plotly_white',
                    showlegend=False
                )
                
                st.plotly_chart(fig_volatility, use_container_width=True)
                
                # 변동성 통계
                volatility_data = []
                for ticker, df_ts in timeseries_data.items():
                    if len(df_ts) > 1:
                        volatility_data.append({
                            'Ticker': ticker,
                            'Avg_Short_Ratio': df_ts['short_ratio'].mean(),
                            'Std_Dev': df_ts['short_ratio'].std(),
                            'Min': df_ts['short_ratio'].min(),
                            'Max': df_ts['short_ratio'].max(),
                            'Range': df_ts['short_ratio'].max() - df_ts['short_ratio'].min()
                        })
                
                if volatility_data:
                    df_volatility = pd.DataFrame(volatility_data)
                    st.markdown("##### 📊 변동성 통계")
                    st.dataframe(df_volatility.round(2), use_container_width=True, hide_index=True)
                    
                    st.info("""
                    **💡 해석:**
                    - **Std_Dev (표준편차)**: 높을수록 변동성이 큼
                    - **Range (범위)**: 최대-최소 차이, 높을수록 불안정
                    - **변동성이 낮고 평균이 40% 미만이면 안정적**
                    """)
        else:
            st.warning("시계열 데이터가 충분하지 않습니다.")
    else:
        st.info("사이드바에서 '시계열 분석 차트'를 활성화하세요.")

# TAB 4: 고급 분석
with tab4:
    st.header("🎯 고급 분석")
    
    # 차트 G: YF vs FINRA 상관관계
    st.subheader("📊 YF Short % vs FINRA Daily % 상관관계")
    st.caption("""
    💡 **두 지표의 관계 분석**
    - **YF Short %**: 공매도 잔고 (누적된 포지션, 월 2회 업데이트)
    - **FINRA Daily %**: 일일 공매도 거래 비율 (신규 거래, 매일 업데이트)
    - 오른쪽 상단에 위치할수록 공매도 압력이 강함
    """)
    fig_correlation = go.Figure()
    
    fig_correlation.add_trace(go.Scatter(
        x=df_results['short_percent_float'],
        y=df_results['daily_short_ratio'],
        mode='markers+text',
        text=df_results['Ticker'],
        textposition='top center',
        marker=dict(
            size=df_results['shares_short_millions'] / 10,
            color=df_results['short_change_pct'],
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title="MoM<br>변화율")
        ),
        hovertemplate='<b>%{text}</b><br>YF Short: %{x:.2f}%<br>FINRA Daily: %{y:.1f}%<extra></extra>'
    ))
    
    fig_correlation.add_hline(y=40, line_dash="dash", line_color="gray", annotation_text="FINRA 정상선 (40%)")
    fig_correlation.add_vline(x=2, line_dash="dash", line_color="gray", annotation_text="YF 매우건강 (2%)")
    fig_correlation.add_vline(x=5, line_dash="dash", line_color="orange", annotation_text="YF 건강선 (5%)")
    
    fig_correlation.update_layout(
        xaxis_title='Yahoo Finance: Short % of Float',
        yaxis_title='FINRA: Daily Short Volume %',
        height=600,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_correlation, use_container_width=True)
    
    st.info("""
    **💡 차트 해석:**
    - **버블 크기**: 공매도 주식 수 (클수록 많음)
    - **색상**: 전월 대비 변화율 (빨강=증가, 초록=감소)
    - **위치**: 오른쪽 상단으로 갈수록 공매도 압력 강함
    """)
    
    st.markdown("---")
    
    # 차트 H: 종합 점수판
    st.subheader("🎯 공매도 종합 점수판")
    st.caption("""
    💡 **4개 지표를 0-100점으로 정규화하여 평균 계산**
    - Short % Score: 공매도 비율 (낮을수록 높은 점수)
    - Days Score: 청산 소요일 (짧을수록 높은 점수)
    - FINRA Score: 일일 공매도 (낮을수록 높은 점수)
    - Change Score: 전월 대비 변화 (감소할수록 높은 점수)
    """)
    
    # 정규화 함수
    def normalize_inverse(values, max_val):
        return np.clip(100 - (values / max_val * 100), 0, 100)
    
    # 정규화
    norm_short_pct = normalize_inverse(df_results['short_percent_float'], 10)
    norm_days = normalize_inverse(df_results['short_ratio_days'], 5)
    norm_finra_daily = normalize_inverse(df_results['daily_short_ratio'], 60)
    norm_change = np.clip(50 - df_results['short_change_pct'] * 2, 0, 100)
    
    # 종합 점수
    comprehensive_score = (norm_short_pct + norm_days + norm_finra_daily + norm_change) / 4
    
    fig_comprehensive = go.Figure()
    
    colors_comp = ['green' if x > 70 else 'orange' if x > 50 else 'red' for x in comprehensive_score]
    
    fig_comprehensive.add_trace(go.Bar(
        x=df_results['Ticker'],
        y=comprehensive_score,
        marker=dict(color=colors_comp),
        text=comprehensive_score.round(1),
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>종합 점수: %{y:.1f}/100<extra></extra>'
    ))
    
    fig_comprehensive.add_hline(y=70, line_dash="dash", line_color="green", annotation_text="우수 (70점)")
    fig_comprehensive.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="보통 (50점)")
    
    fig_comprehensive.update_layout(
        xaxis_title='종목',
        yaxis_title='종합 점수 (점)',
        height=450,
        template='plotly_white',
        showlegend=False
    )
    
    st.plotly_chart(fig_comprehensive, use_container_width=True)
    
    # 점수 상세 테이블
    score_detail = pd.DataFrame({
        'Ticker': df_results['Ticker'],
        'Short%_Score': norm_short_pct.round(1),
        'Days_Score': norm_days.round(1),
        'FINRA_Score': norm_finra_daily.round(1),
        'Change_Score': norm_change.round(1),
        'Total_Score': comprehensive_score.round(1)
    })
    
    st.markdown("##### 📊 공매도 종합 점수 상세")
    
    # 컬럼명에 툴팁 설명 추가
    with st.expander("📖 컬럼 설명 보기", expanded=False):
        st.markdown("""
        - **Short%_Score**: 유통주식 대비 공매도 비율을 점수화 (낮을수록 높은 점수)
        - **Days_Score**: 공매도 청산 소요일을 점수화 (짧을수록 높은 점수)
        - **FINRA_Score**: FINRA 일일 공매도 비율을 점수화 (낮을수록 높은 점수)
        - **Change_Score**: 전월 대비 공매도 변화를 점수화 (감소할수록 높은 점수)
        - **Total_Score**: 위 4개 점수의 평균 (70점 이상이면 우수)
        """)
    
    st.dataframe(score_detail, use_container_width=True, hide_index=True)
    
    st.info("""
    **💡 점수 해석:**
    - **Short%_Score**: 공매도 비율 점수 (낮을수록 높은 점수)
    - **Days_Score**: 청산 소요일 점수 (짧을수록 높은 점수)
    - **FINRA_Score**: 일일 공매도 점수 (낮을수록 높은 점수)
    - **Change_Score**: 전월 대비 변화 점수 (감소할수록 높은 점수)
    - **Total_Score**: 70점 이상이면 공매도 관점에서 우수
    """)
    
    st.markdown("---")
    
    # 데이터 품질 분석
    st.subheader("🔍 데이터 품질 및 신뢰도 분석")
    st.caption("""
    💡 **각 종목의 데이터 완성도와 공매도 신호 평가**
    - ⭐⭐⭐: YF + FINRA 데이터 모두 확보 (최고 신뢰도)
    - ⭐⭐: 한 가지 데이터만 확보
    - ⭐: 데이터 부족
    """)
    
    quality_analysis = []
    
    for idx, row in df_results.iterrows():
        ticker = row['Ticker']
        
        yf_complete = (row.get('short_percent_float', 0) > 0 and row.get('shares_short_millions', 0) > 0)
        finra_complete = row.get('daily_short_ratio', 0) > 0
        
        data_quality = "⭐⭐⭐" if (yf_complete and finra_complete) else "⭐⭐" if (yf_complete or finra_complete) else "⭐"
        
        short_pct = row.get('short_percent_float', 0)
        daily_short = row.get('daily_short_ratio', 0)
        
        if short_pct < 3 and daily_short < 40:
            signal = "💚 매우 긍정"
        elif short_pct < 5 and daily_short < 45:
            signal = "🟢 긍정"
        elif short_pct < 10 and daily_short < 50:
            signal = "🟡 중립"
        else:
            signal = "🔴 약세 압력"
        
        quality_analysis.append({
            'Ticker': ticker,
            'YF_Complete': '✓' if yf_complete else '✗',
            'FINRA_Complete': '✓' if finra_complete else '✗',
            'Data_Quality': data_quality,
            'Short_Signal': signal,
            'Interpretation': f"잔고 {short_pct:.1f}% / 일거래 {daily_short:.0f}%"
        })
    
    df_quality = pd.DataFrame(quality_analysis)
    st.dataframe(df_quality, use_container_width=True, hide_index=True)
    
    st.success("""
    **💡 통합 해석 가이드**
    
    **[최적 투자 신호] 💚**
    - Yahoo: Short % < 3% + FINRA: Daily < 40%
    - 공매도 잔고도 낮고, 일일 공매도 거래도 적음
    - 시장의 강한 신뢰, 매수 추천
    
    **[양호한 신호] 🟢**
    - Yahoo: Short % < 5% + FINRA: Daily < 45%
    - 전반적으로 건강한 수준
    - 적극 매수 고려
    
    **[주의 필요] 🟡**
    - Yahoo: Short % 5-10% + FINRA: Daily 45-50%
    - 약간의 약세 포지션 존재
    - 기술적 분석 병행 필수
    
    **[약세 압력] 🔴**
    - Yahoo: Short % > 10% 또는 FINRA: Daily > 50%
    - 강한 공매도 압력
    - Short Squeeze 가능성은 있으나 위험도 높음
    """)

# TAB 5: 데이터
with tab5:
    st.header("📋 전체 데이터")
    
    # 최종 요약표
    st.subheader("🏆 종합 요약표")
    summary_table = []
    
    for idx, row in df_results.iterrows():
        rank = df_results.index.get_loc(idx) + 1
        
        summary_table.append({
            'Rank': rank,
            'Ticker': row['Ticker'],
            'Price': f"${row['Current_Price']:.2f}",
            'VWAP_Diff': f"{row['Price_vs_VWAP_%']:+.1f}%",
            'Q_Return': f"{row['Quarter_Return_%']:+.1f}%",
            'YF_Short%': f"{row.get('short_percent_float', 0):.2f}%",
            'Days_Cover': f"{row.get('short_ratio_days', 0):.1f}",
            'FINRA_Daily%': f"{row.get('daily_short_ratio', 0):.1f}%" if row.get('daily_short_ratio', 0) > 0 else "N/A",
            'Tech_Score': f"{row['Buy_Signal_Score']}/100",
            'Total_Score': f"{row['Total_Investment_Score']}/120",
            'Signal': '💚' if row['Total_Investment_Score'] >= 90 else '💛' if row['Total_Investment_Score'] >= 75 else '💙'
        })
    
    df_summary = pd.DataFrame(summary_table)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # 전체 데이터
    st.subheader("📊 상세 데이터")
    st.dataframe(df_results, use_container_width=True, hide_index=True)
    
    # CSV 다운로드
    csv = df_results.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        "📥 전체 데이터 CSV 다운로드", 
        csv, 
        f"mag7_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 
        "text/csv",
        use_container_width=True
    )
    
    # 요약 통계
    st.markdown("---")
    st.subheader("📈 요약 통계")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 기술적 분석")
        above_vwap = len(df_results[df_results['Is_Above_VWAP'] == True])
        st.metric("VWAP 위 종목", f"{above_vwap}개")
        st.metric("평균 분기 수익률", f"{df_results['Quarter_Return_%'].mean():+.1f}%")
        st.metric("평균 기술적 점수", f"{df_results['Buy_Signal_Score'].mean():.1f}/100")
    
    with col2:
        st.markdown("##### 공매도 분석")
        low_short = len(df_results[df_results['short_percent_float'] < 5])
        st.metric("낮은 공매도 (<5%)", f"{low_short}개")
        st.metric("평균 공매도 비율", f"{df_results['short_percent_float'].mean():.2f}%")
        st.metric("평균 청산 소요일", f"{df_results['short_ratio_days'].mean():.1f}일")
    
    with col3:
        st.markdown("##### 종합 평가")
        top_score = len(df_results[df_results['Total_Investment_Score'] >= 90])
        st.metric("최우선 매수 (90점↑)", f"{top_score}개")
        strong_buy = len(df_results[(df_results['Total_Investment_Score'] >= 75) & 
                                     (df_results['Total_Investment_Score'] < 90)])
        st.metric("강력 매수 (75-90점)", f"{strong_buy}개")
        st.metric("평균 종합 점수", f"{df_results['Total_Investment_Score'].mean():.1f}/120")
    
    st.markdown("---")
    
    # 투자 가이드
    st.subheader("💡 투자 가이드")
    st.info("""
    **1. 💚 최우선 매수 (90점 이상):**
    - VWAP 위 + 낮은 공매도 비율
    - 즉시 매수 검토, 강한 상승 모멘텀 예상
    
    **2. 💛 강력 매수 (75-90점):**
    - 기술적으로 양호하나 공매도 약간 존재
    - VWAP 근처 조정 시 매수 기회
    
    **3. 💙 눌림목 대기 (60-75점):**
    - 공매도 비율 확인 필수
    - VWAP 돌파 확인 후 매수
    
    **4. ⚠️ 공매도 주의사항:**
    - 10% 이상: Short Squeeze 가능성 주의
    - 5% 미만: 시장의 신뢰 높음
    - Days to Cover 3일 이상: 변동성 증가 가능
    """)

# ==================== TAB 6: AI 분석 추가 ====================
with tab6:
    st.header("🤖 Gemini AI 분석")
    st.markdown("**Google Gemini 2.5 Flash 기반 MAG 7+2 종합 분석**")
    
    if not GEMINI_ENABLED:
        st.error("""
        ❌ **Gemini API가 설정되지 않았습니다.**
        
        `.streamlit/secrets.toml` 파일에 추가:
```toml
        GEMINI_API_KEY = "your-api-key-here"
```
        
        **API 키 발급:** https://aistudio.google.com/app/apikey (무료!)
        """)
        st.stop()
    
    st.markdown("---")
    
    # 분석 유형 선택
    col1, col2 = st.columns([3, 1])
    
    with col1:
        analysis_type = st.selectbox(
            "📊 분석 유형 선택",
            ["종합분석", "공매도분석", "기술적분석", "비교분석"]
        )
    
    with col2:
        st.metric("🔋 API 상태", "활성화")
    
    st.markdown("##### 📋 분석 유형 설명")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.info("""
        **🎯 종합분석**
        - 현재 MAG 7+2 시장 상황
        - Top 3 투자 추천
        - 주의 종목
        - 향후 2주 전망
        
        **🔴 공매도분석**
        - 공매도 상황 평가
        - Short Squeeze 가능성
        - 안전한 종목
        - 투자 전략
        """)
    
    with col_b:
        st.info("""
        **📈 기술적분석**
        - VWAP 기반 분석
        - 강한 모멘텀 종목
        - 진입/청산 타이밍
        - 손절/익절 기준
        
        **📊 비교분석**
        - 빅테크 7 vs 비트코인 2
        - 섹터별 매력도
        - 포트폴리오 구성
        - 리스크 관리
        """)
    
    # 분석 실행
    if st.button("🚀 AI 분석 실행", type="primary", use_container_width=True):
        market_data = get_market_summary(df_results)
        short_data = get_short_summary(df_results)
        technical_data = get_technical_summary(df_results)
        
        analysis_result = analyze_with_gemini(
            analysis_type,
            market_data,
            short_data,
            technical_data
        )
        
        st.markdown("---")
        st.markdown(f"### 📊 {analysis_type} 결과")
        st.caption(f"분석 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 결과 표시
        st.markdown(
            f"""
            <div style='background-color: #f0f2f6; padding: 25px; border-radius: 10px; border-left: 5px solid #2E86AB;'>
            {analysis_result.replace(chr(10), '<br>')}
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("---")
        
        # 다운로드 및 저장
        col_x, col_y = st.columns(2)
        
        with col_x:
            st.download_button(
                "📥 분석 결과 다운로드",
                analysis_result,
                file_name=f"mag7_ai_{analysis_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col_y:
            if st.button("💾 분석 결과 저장", use_container_width=True):
                if 'analysis_history' not in st.session_state:
                    st.session_state.analysis_history = []
                
                st.session_state.analysis_history.append({
                    'timestamp': datetime.now(),
                    'type': analysis_type,
                    'result': analysis_result
                })
                st.success("✅ 저장 완료!")
                st.rerun()
    
    # 히스토리
    if 'analysis_history' in st.session_state and len(st.session_state.analysis_history) > 0:
        st.markdown("---")
        st.markdown("### 📜 분석 히스토리 (최근 5개)")
        
        for idx, item in enumerate(reversed(st.session_state.analysis_history[-5:])):
            with st.expander(f"🕐 {item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - {item['type']}"):
                st.markdown(item['result'])
        
        if st.button("🗑️ 히스토리 전체 삭제"):
            st.session_state.analysis_history = []
            st.success("✅ 삭제 완료!")
            st.rerun()
    
    # 대화형 챗봇
    st.markdown("---")
    st.markdown("### 💬 AI와 대화하기")
    st.caption("💡 MAG 7+2에 대해 자유롭게 질문하세요!")
    
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    
    col_chat1, col_chat2 = st.columns([4, 1])
    
    with col_chat2:
        if st.button("🔄 대화 초기화", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()
    
    # 메시지 표시
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 입력
    if prompt := st.chat_input("질문 입력 (예: NVDA 지금 사도 될까요?)"):
        st.session_state.chat_messages.append({
            "role": "user",
            "content": prompt
        })
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # AI 응답
        context = f"""
당신은 MAG 7+2 전문 애널리스트입니다.

현재 시장 상황:
{get_market_summary(df_results)}

공매도 현황:
{get_short_summary(df_results)}

기술적 분석:
{get_technical_summary(df_results)}

사용자 질문: {prompt}

간결하고 명확하게 답변해주세요.
"""
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 생각 중..."):
                try:
                    response = gemini_model.generate_content(context)
                    answer = response.text
                    st.markdown(answer)
                    
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": answer
                    })
                except Exception as e:
                    st.error(f"❌ 오류: {str(e)}\n무료 할당량을 초과했을 수 있습니다.")
    
    # 사용 팁
    st.markdown("---")
    st.markdown("### 💡 AI 분석 활용 팁")
    
    col_tip1, col_tip2 = st.columns(2)
    
    with col_tip1:
        st.success("""
        **✅ 효과적인 질문 예시**
        - "NVDA가 지금 매수 타이밍인가요?"
        - "공매도 비율이 높은 종목은?"
        - "VWAP 위에 있는 종목 추천해줘"
        - "비트코인 관련주 전망은?"
        """)
    
    with col_tip2:
        st.info("""
        **📊 분석 유형별 활용**
        - **종합분석**: 전반적인 시장 파악
        - **공매도분석**: 리스크 체크
        - **기술적분석**: 매매 타이밍
        - **비교분석**: 포트폴리오 구성
        """)


# 푸터
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "<p>📊 MAG 7+2 종합 분석 대시보드 v3.2 (Expander Edition)</p>"
    "<p>Magnificent Seven + Bitcoin Exposure | Powered by Streamlit</p>"
    "</div>", 
    unsafe_allow_html=True
)
