import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import random
import time
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# UI CONFIGURATION
# ==========================================
st.set_page_config(page_title="OpiumFly Quant Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("🦅 OpiumFly Institutional Quant Dashboard")

# ==========================================
# SHARED BLACK-SCHOLES ENGINE
# ==========================================
def black_scholes(S, K, T, r, sigma, option_type="call"):
    if T <= 0: return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call": return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put": return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# ==========================================
# ENGINE 1: SWARM CONNECTOME
# ==========================================
class FlyBrain:
    def __init__(self, ticker):
        self.ticker = ticker
        self.synaptic_bias = 0.5 
        self.active_option = None
        
    def read_motor_neurons(self):
        if self.active_option is not None: return "HOLD"
        decision = random.random()
        if decision < (0.05 + (self.synaptic_bias * 0.1)): return "BUY_CALL"
        elif decision > (0.95 - ((1 - self.synaptic_bias) * 0.1)): return "BUY_PUT"
        return "HOLD"

    def stimulate_pam(self):
        if self.active_option['type'] == 'call': self.synaptic_bias = min(0.95, self.synaptic_bias + 0.05)
        else: self.synaptic_bias = max(0.05, self.synaptic_bias - 0.05)
            
    def stimulate_ppl101(self):
        if self.active_option['type'] == 'call': self.synaptic_bias = max(0.05, self.synaptic_bias - 0.05)
        else: self.synaptic_bias = min(0.95, self.synaptic_bias + 0.05)

def run_swarm_engine():
    tickers = ['QQQ', 'SPY', 'DIA', 'IWM', 'OEF']
    
    with st.spinner(f"Fetching 60-day market data for Swarm ({', '.join(tickers)})..."):
        data = yf.download(tickers, period="60d", interval="5m", group_by="ticker", progress=False)
        valid_times = data[tickers[0]].dropna().index
    
    portfolio = {"cash": 100000.0, "initial_balance": 100000.0}
    equity_curve, equity_timestamps, trade_pnls = [], [], []
    swarm = {ticker: FlyBrain(ticker) for ticker in tickers}
    
    r, sigma, DTE_years, contracts_per_trade = 0.04, 0.18, 7/365, 5  

    progress_bar = st.progress(0)
    total_steps = len(valid_times)

    with st.spinner("Simulating Swarm Neural Networks..."):
        for i, timestamp in enumerate(valid_times):
            current_equity = portfolio['cash']
            for ticker in tickers:
                try:
                    row = data[ticker].loc[timestamp]
                    if pd.isna(row['Close']): continue
                except KeyError: continue
                    
                current_price = float(row['Close'])
                brain = swarm[ticker]
                
                if brain.active_option is not None:
                    opt = brain.active_option
                    time_elapsed_yrs = (timestamp - opt['entry_time']).total_seconds() / (365 * 24 * 3600)
                    T_remaining = max(0.001, opt['T_start'] - time_elapsed_yrs)
                    current_premium = black_scholes(current_price, opt['strike'], T_remaining, r, sigma, opt['type'])
                    pnl_pct = (current_premium - opt['entry_premium']) / opt['entry_premium']
                    
                    exit_triggered = pnl_pct >= 0.50 or pnl_pct <= -0.40 or time_elapsed_yrs >= (5 / 365)
                        
                    if exit_triggered:
                        revenue = current_premium * 100 * opt['contracts']
                        portfolio['cash'] += revenue
                        net_profit = revenue - opt['cost_basis']
                        trade_pnls.append(net_profit)
                        
                        if net_profit > 0: brain.stimulate_pam()
                        else: brain.stimulate_ppl101()
                        brain.active_option = None
                        
                    current_equity += (current_premium * 100 * opt['contracts']) if brain.active_option else 0

                if brain.active_option is None:
                    action = brain.read_motor_neurons()
                    if action in ["BUY_CALL", "BUY_PUT"]:
                        opt_type = "call" if action == "BUY_CALL" else "put"
                        premium = black_scholes(current_price, current_price, DTE_years, r, sigma, opt_type)
                        cost = premium * 100 * contracts_per_trade
                        if portfolio['cash'] >= cost:
                            portfolio['cash'] -= cost
                            brain.active_option = {'type': opt_type, 'strike': current_price, 'entry_time': timestamp, 'T_start': DTE_years, 'contracts': contracts_per_trade, 'cost_basis': cost, 'entry_premium': premium}

            equity_timestamps.append(timestamp)
            equity_curve.append(current_equity)
            if i % 100 == 0: progress_bar.progress(i / total_steps)
            
    progress_bar.empty()
    return equity_timestamps, equity_curve, trade_pnls, portfolio

# ==========================================
# ENGINE 2 & 3: ML COMPOUNDING
# ==========================================
def run_ml_engine(mode):
    is_aggressive = (mode == "Aggressive")
    ticker = "SPY" if is_aggressive else "QQQ"
    
    with st.spinner(f"Fetching market data for {ticker}..."):
        data = yf.Ticker(ticker).history(period="59d", interval="5m")
        if data.empty: data = yf.Ticker(ticker).history(period="1mo", interval="5m")
        if data.empty:
            st.error("Could not retrieve market data from Yahoo Finance.")
            return None, None, None, None

    with st.spinner(f"Calculating {'Aggressive' if is_aggressive else 'Defensive'} quantitative features..."):
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['STD_20'] = data['Close'].rolling(window=20).std()
        
        band_mult = 1.2 if is_aggressive else 1.5
        data['Lower_Band'] = data['SMA_20'] - (band_mult * data['STD_20'])
        data['Upper_Band'] = data['SMA_20'] + (band_mult * data['STD_20'])
        
        if not is_aggressive: data['EMA_100'] = data['Close'].ewm(span=100, adjust=False).mean()
        if is_aggressive:
            data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
            data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()
        
        data['Z_Score'] = (data['Close'] - data['SMA_20']) / data['STD_20']
        data['ROC_5'] = data['Close'].pct_change(5) * 100 
        if not is_aggressive: data['Price_EMA_Dist'] = (data['Close'] - data['EMA_100']) / data['EMA_100']
        
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        data['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
        
        data['Log_Ret'] = np.log(data['Close'] / data['Close'].shift(1))
        data['Volatility'] = data['Log_Ret'].rolling(window=78).std() * np.sqrt(252 * 78) 
        data['Vol_SMA'] = data['Volatility'].rolling(window=390).mean()
        
        HURDLE = 0.0008 if is_aggressive else 0.0010
        data['Future_Price'] = data['Close'].shift(-6)
        data['Call_Target'] = ((data['Future_Price'] - data['Close']) / data['Close'] > HURDLE).astype(int)
        data['Put_Target']  = ((data['Close'] - data['Future_Price']) / data['Close'] > HURDLE).astype(int)
        data = data.dropna()

    data['Date'] = data.index.date
    unique_dates = data['Date'].unique()
    cycles = [unique_dates[i:i+5] for i in range(0, len(unique_dates), 5)]
    if len(cycles) < 4:
        st.error("Not enough data chunks downloaded.")
        return None, None, None, None
        
    train_dates = np.concatenate(cycles[:3])
    trade_cycles = cycles[3:]
    train_data = data[data['Date'].isin(train_dates)]
    
    with st.spinner("Training Random Forest AI..."):
        features = ['RSI', 'Z_Score', 'Volatility', 'ROC_5']
        if not is_aggressive: features.insert(3, 'Price_EMA_Dist')
        
        rf_call = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42, min_samples_leaf=1 if is_aggressive else 5)
        rf_call.fit(train_data[features], train_data['Call_Target'])
        rf_put = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42, min_samples_leaf=1 if is_aggressive else 5)
        rf_put.fit(train_data[features], train_data['Put_Target'])

    portfolio = {"cash": 100000.0, "initial_balance": 100000.0}
    active_option = None
    equity_timestamps, equity_curve, trade_pnls = [], [], []
    
    CONFIDENCE = 0.51 if is_aggressive else 0.55
    HARD_STOP = 0.45 if is_aggressive else 0.20
    TRAIL_ACT = 0.40 if is_aggressive else 0.10
    TRAIL_STEP = 0.15 if is_aggressive else 0.08
    MAX_PROFIT = 1.20 if is_aggressive else 0.50

    progress_bar = st.progress(0)
    total_steps = len(trade_cycles)

    with st.spinner("Running 60-Day Forward Simulation..."):
        for cycle_num, cycle_dates in enumerate(trade_cycles, 1):
            cycle_data = data[data['Date'].isin(cycle_dates)]
            
            for i in range(len(cycle_data)):
                index = cycle_data.index[i]
                row = cycle_data.iloc[i]
                is_last_candle = (i == len(cycle_data) - 1)
                
                current_price = float(row['Close'])
                r, sigma = 0.04, float(row['Volatility']) if pd.notna(row['Volatility']) else (0.15 if is_aggressive else 0.18)
                
                if active_option is not None:
                    T_rem = active_option['T_start'] - ((index - active_option['entry_time']).total_seconds() / (365 * 24 * 3600))
                    premium = black_scholes(current_price, active_option['strike'], max(0.001, T_rem), r, sigma, active_option['type'])
                    if premium > active_option['peak']: active_option['peak'] = premium
                    pnl_pct = (premium - active_option['entry']) / active_option['entry']
                    
                    exit_triggered = False
                    if is_last_candle or pnl_pct >= MAX_PROFIT or premium <= active_option['entry'] * (1 - HARD_STOP):
                        exit_triggered = True
                    elif active_option['peak'] >= active_option['entry'] * (1 + TRAIL_ACT):
                        if premium <= active_option['peak'] * (1 - TRAIL_STEP): exit_triggered = True
                        
                    if exit_triggered:
                        revenue = premium * 100 * active_option['contracts']
                        pnl = revenue - active_option['cost_basis']
                        trade_pnls.append(pnl)
                        portfolio["cash"] += revenue
                        active_option = None
                        
                if active_option is None and not is_last_candle:
                    DTE = 7/365 if is_aggressive else 14/365
                    feats = pd.DataFrame([[row[f] for f in features]], columns=features)
                    vol_safe = row['Volatility'] < (row['Vol_SMA'] * (1.75 if is_aggressive else 1.20))
                    
                    bullish = (row['EMA_50'] > row['EMA_200']) if is_aggressive else True
                    bearish = (row['EMA_50'] < row['EMA_200']) if is_aggressive else True
                    rsi_call = 45 if is_aggressive else 40
                    rsi_put = 55 if is_aggressive else 60
                    
                    if bullish and current_price < row['Lower_Band'] and row['RSI'] < rsi_call and vol_safe:
                        prob = rf_call.predict_proba(feats)[0][1]
                        if prob >= CONFIDENCE:
                            alloc = min(0.25 if is_aggressive else 0.10, max(0.05 if is_aggressive else 0.02, (prob - 0.45) * (1.0 if is_aggressive else 0.5)))
                            cost_per = black_scholes(current_price, current_price, DTE, r, sigma, 'call') * 100
                            contracts = int((portfolio['cash'] * alloc) / cost_per)
                            if contracts > 0:
                                cost = cost_per * contracts
                                portfolio['cash'] -= cost
                                active_option = {'type':'call', 'strike':current_price, 'entry_time':index, 'T_start':DTE, 'contracts':contracts, 'cost_basis':cost, 'entry':cost_per/100, 'peak':cost_per/100}

                    elif bearish and current_price > row['Upper_Band'] and row['RSI'] > rsi_put and vol_safe:
                        prob = rf_put.predict_proba(feats)[0][1]
                        if prob >= CONFIDENCE:
                            alloc = min(0.25 if is_aggressive else 0.10, max(0.05 if is_aggressive else 0.02, (prob - 0.45) * (1.0 if is_aggressive else 0.5)))
                            cost_per = black_scholes(current_price, current_price, DTE, r, sigma, 'put') * 100
                            contracts = int((portfolio['cash'] * alloc) / cost_per)
                            if contracts > 0:
                                cost = cost_per * contracts
                                portfolio['cash'] -= cost
                                active_option = {'type':'put', 'strike':current_price, 'entry_time':index, 'T_start':DTE, 'contracts':contracts, 'cost_basis':cost, 'entry':cost_per/100, 'peak':cost_per/100}

                curr_val = portfolio['cash']
                if active_option is not None:
                    T_rem = DTE - ((index - active_option['entry_time']).total_seconds() / (365 * 24 * 3600))
                    curr_val += black_scholes(current_price, active_option['strike'], max(0.001, T_rem), r, sigma, active_option['type']) * 100 * active_option['contracts']
                equity_curve.append(curr_val)
                equity_timestamps.append(index)
                
            progress_bar.progress(cycle_num / total_steps)

    progress_bar.empty()
    return equity_timestamps, equity_curve, trade_pnls, portfolio

# ==========================================
# STREAMLIT UI RENDERER
# ==========================================
with st.sidebar:
    st.header("⚙️ Simulation Settings")
    engine_choice = st.radio("Select Trading Engine:", [
        "Swarm Intelligence (Multi-Index)",
        "QQQ Safe Compounding (V7)",
        "SPY Aggressive Growth (V10)"
    ])
    
    st.markdown("---")
    st.markdown("**Initial Capital:** $100,000")
    st.markdown("**Data Granularity:** 5m Candles")
    run_button = st.button("🚀 Run Backtest", use_container_width=True, type="primary")

if run_button:
    if engine_choice == "Swarm Intelligence (Multi-Index)":
        timestamps, curve, trade_pnls, port = run_swarm_engine()
    elif engine_choice == "QQQ Safe Compounding (V7)":
        timestamps, curve, trade_pnls, port = run_ml_engine("Safe")
    else:
        timestamps, curve, trade_pnls, port = run_ml_engine("Aggressive")
        
    if timestamps and curve:
        # 1. Statistical Calculations
        net_profit = port['cash'] - port['initial_balance']
        roi = (net_profit / port['initial_balance']) * 100
        
        eq_series = pd.Series(curve)
        rolling_max = eq_series.cummax()
        drawdowns = (eq_series - rolling_max) / rolling_max * 100
        max_dd = drawdowns.min() if not drawdowns.empty else 0.0
        
        wins = [x for x in trade_pnls if x > 0]
        losses = [x for x in trade_pnls if x <= 0]
        
        total_trades = len(trade_pnls)
        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
        
        # 2. Key Performance Indicators Layout
        st.subheader(f"🏁 Performance Audit: {engine_choice.split(' ')[0]}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Net Return", f"${net_profit:+,.2f}", f"{roi:+.2f}%")
        col2.metric("Maximum Drawdown", f"{max_dd:.2f}%")
        col3.metric("Win Rate", f"{win_rate:.1f}%", f"{len(wins)}W / {len(losses)}L")
        col4.metric("Profit Factor", f"{profit_factor:.2f}x")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ending Capital", f"${port['cash']:,.2f}")
        c2.metric("Total Trades Executed", total_trades)
        c3.metric("Average Winning Trade", f"${avg_win:,.2f}")
        c4.metric("Average Losing Trade", f"${avg_loss:,.2f}")
        
        st.markdown("---")
        
        # 3. Dedicated Visualizations
        plt.style.use('dark_background')
        chart_color = '#00ffcc' if "Safe" in engine_choice or "Swarm" in engine_choice else '#ffcc00'
        
        # GRAPH 1: Cumulative Equity Curve
        st.subheader("1. Portfolio Equity Curve")
        fig1, ax1 = plt.subplots(figsize=(14, 4))
        fig1.patch.set_facecolor('#0e1117')
        ax1.set_facecolor('#0e1117')
        ax1.plot(timestamps, curve, color=chart_color, linewidth=2)
        ax1.axhline(y=port['initial_balance'], color='red', linestyle='--', alpha=0.5, label='Initial Balance')
        ax1.tick_params(colors='white')
        for spine in ax1.spines.values(): spine.set_edgecolor('#555555')
        ax1.grid(True, linestyle='--', alpha=0.2, color='#555555')
        ax1.legend(facecolor='#0e1117', edgecolor='#555555', labelcolor='white')
        fig1.autofmt_xdate()
        st.pyplot(fig1)

        # GRAPH 2: Underwater Drawdown Curve
        st.subheader("2. Underwater Curve (Risk of Ruin)")
        fig2, ax2 = plt.subplots(figsize=(14, 3))
        fig2.patch.set_facecolor('#0e1117')
        ax2.set_facecolor('#0e1117')
        ax2.fill_between(timestamps, drawdowns, 0, color='#ff3333', alpha=0.4)
        ax2.plot(timestamps, drawdowns, color='#ff3333', linewidth=1)
        ax2.set_ylabel("Drawdown %", color='white')
        ax2.tick_params(colors='white')
        for spine in ax2.spines.values(): spine.set_edgecolor('#555555')
        ax2.grid(True, linestyle='--', alpha=0.2, color='#555555')
        fig2.autofmt_xdate()
        st.pyplot(fig2)
        
        # GRAPH 3: Chronological Trade PnL Distribution
        st.subheader("3. Sequential Trade Analysis (PnL)")
        if trade_pnls:
            fig3, ax3 = plt.subplots(figsize=(14, 4))
            fig3.patch.set_facecolor('#0e1117')
            ax3.set_facecolor('#0e1117')
            bar_colors = ['#00ffcc' if pnl > 0 else '#ff3333' for pnl in trade_pnls]
            ax3.bar(range(1, len(trade_pnls) + 1), trade_pnls, color=bar_colors, alpha=0.8)
            ax3.axhline(y=0, color='#ffffff', linewidth=1, alpha=0.5)
            ax3.set_xlabel("Sequential Trade ID", color='white')
            ax3.set_ylabel("Realized PnL (USD)", color='white')
            ax3.tick_params(colors='white')
            for spine in ax3.spines.values(): spine.set_edgecolor('#555555')
            ax3.grid(True, axis='y', linestyle='--', alpha=0.2, color='#555555')
            st.pyplot(fig3)
        else:
            st.warning("No trades were executed during this period to graph.")
            
else:
    st.info("👈 Select a strategy from the sidebar and execute the backtest.")