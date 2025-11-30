import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objs as go
from datetime import date, timedelta

# ================================
# PARAMÈTRES GLOBAUX
# ================================

# Liste (approximative) des principaux titres du SMI sur SIX.
# Tu peux l'ajuster facilement dans la sidebar.
SMI_TICKERS = [
    "NESN.SW",  # Nestlé
    "NOVN.SW",  # Novartis
    "ROG.SW",   # Roche
    "UBSG.SW",  # UBS
    "ZURN.SW",  # Zurich Insurance
    "SIKA.SW",  # Sika
    "GIVN.SW",  # Givaudan
    "ADEN.SW",  # Adecco
    "ALCC.SW",  # Alcon (si erreur, essaie ALCN.SW dans la sidebar)
    "HOLN.SW",  # Holcim
    "LONN.SW",  # Lonza
    "SREN.SW",  # Swiss Re
    "SGSN.SW",  # SGS
    "LOGN.SW",  # Logitech
    "UHR.SW",   # Swatch Group
    "ABBN.SW",  # ABB
]

DATA_PERIOD_YEARS = 2  # historique utilisé pour les indicateurs


# ================================
# FONCTIONS UTILITAIRES
# ================================

def safe_float(x):
    """Convertit en float si possible, sinon retourne None."""
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


# ================================
# INDICATEURS TECHNIQUES
# ================================

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(close: pd.Series):
    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    macd = ema12 - ema26
    signal = ema(macd, 9)
    hist = macd - signal
    return macd, signal, hist


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["EMA20"] = ema(df["Close"], 20)
    df["EMA50"] = ema(df["Close"], 50)
    df["EMA200"] = ema(df["Close"], 200)

    df["RSI14"] = compute_rsi(df["Close"], 14)

    macd, signal, hist = compute_macd(df["Close"])
    df["MACD"] = macd
    df["MACD_signal"] = signal
    df["MACD_hist"] = hist

    df["Vol_Moy20"] = df["Volume"].rolling(window=20).mean()

    return df


# ================================
# TÉLÉCHARGEMENT DES DONNÉES
# ================================

@st.cache_data
def download_data(tickers, years: int = DATA_PERIOD_YEARS) -> dict:
    end = date.today()
    start = end - timedelta(days=years * 365)

    data: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval="1d",
                progress=False,
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                df = df.dropna()
                df = add_indicators(df)
                data[ticker] = df
        except Exception as e:
            st.warning(f"Erreur lors du téléchargement de {ticker} : {e}")

    return data


# ================================
# SCORING COURT / LONG TERME
# ================================

def score_short_term(df: pd.DataFrame) -> float:
    """Score pour horizon ~1 mois."""
    if df is None or df.empty:
        return 0.0

    last = df.iloc[-1]

    score = 0.0

    # RSI
    rsi = safe_float(last.get("RSI14"))
    if rsi is not None:
        if rsi < 30:
            score += 1      # très survendu
        if 30 <= rsi < 40:
            score += 3      # zone de rebond
        if 40 <= rsi <= 60:
            score += 1      # neutre/ok

    # MACD
    macd = safe_float(last.get("MACD"))
    macd_sig = safe_float(last.get("MACD_signal"))
    if macd is not None and macd_sig is not None:
        if macd > macd_sig:
            score += 3
        if macd > 0:
            score += 1

    # Position vs EMA
    close = safe_float(last.get("Close"))
    ema20 = safe_float(last.get("EMA20"))
    ema50 = safe_float(last.get("EMA50"))

    if close is not None and ema20 is not None and close > ema20:
        score += 2
    if close is not None and ema50 is not None and close > ema50:
        score += 1

    # Volume
    vol = safe_float(last.get("Volume"))
    vol_moy = safe_float(last.get("Vol_Moy20"))
    if vol is not None and vol_moy is not None and vol > vol_moy * 1.2:
        score += 2

    return float(score)


def score_long_term(df: pd.DataFrame) -> float:
    """Score pour horizon ~1 an."""
    if df is None or df.empty:
        return 0.0

    last = df.iloc[-1]
    score = 0.0

    close = safe_float(last.get("Close"))
    ema50 = safe_float(last.get("EMA50"))
    ema200 = safe_float(last.get("EMA200"))
    rsi = safe_float(last.get("RSI14"))
    macd = safe_float(last.get("MACD"))

    # Tendance structurelle
    if close is not None and ema200 is not None and close > ema200:
        score += 3

    if ema50 is not None and ema200 is not None and ema50 > ema200:
        score += 3

    # RSI
    if rsi is not None and 40 <= rsi <= 65:
        score += 2

    # MACD
    if macd is not None and macd > 0:
        score += 2

    # Liquidité minimum
    vol_moy = safe_float(last.get("Vol_Moy20"))
    if vol_moy is not None:
        score += 1

    return float(score)


def build_scores(data: dict) -> pd.DataFrame:
    rows = []

    for ticker, df in data.items():
        if df is None or df.empty:
            continue

        last = df.iloc[-1]
        row = {
            "Ticker": ticker,
            "Dernier_Prix": safe_float(last.get("Close")),
            "RSI14": safe_float(last.get("RSI14")),
            "MACD": safe_float(last.get("MACD")),
            "MACD_signal": safe_float(last.get("MACD_signal")),
            "EMA20": safe_float(last.get("EMA20")),
            "EMA50": safe_float(last.get("EMA50")),
            "EMA200": safe_float(last.get("EMA200")),
            "Volume": safe_float(last.get("Volume")),
            "Volume_Moy20": safe_float(last.get("Vol_Moy20")),
            "Score_Court_Terme": score_short_term(df),
            "Score_Long_Terme": score_long_term(df),
        }
        rows.append(row)

    if len(rows) == 0:
        return pd.DataFrame()

    scores_df = pd.DataFrame(rows)
    scores_df = scores_df.sort_values("Ticker").reset_index(drop=True)
    return scores_df


# ================================
# GRAPHIQUES
# ================================

def plot_candles_with_indicators(df: pd.DataFrame, ticker: str):
    if df is None or df.empty:
        st.write("Pas de données pour cet instrument.")
        return

    df = df.copy().tail(180)

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Prix"
    ))

    if "EMA20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], mode="lines", name="EMA20"))
    if "EMA50" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], mode="lines", name="EMA50"))
    if "EMA200" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA200"], mode="lines", name="EMA200"))

    fig.update_layout(
        title=f"Prix & moyennes mobiles - {ticker}",
        xaxis_title="Date",
        yaxis_title="Prix",
        xaxis_rangeslider_visible=False,
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    if "RSI14" in df.columns:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=df.index,
            y=df["RSI14"],
            mode="lines",
            name="RSI14"
        ))
        fig_rsi.add_hline(y=30, line_dash="dash")
        fig_rsi.add_hline(y=70, line_dash="dash")
        fig_rsi.update_layout(
            title="RSI14",
            xaxis_title="Date",
            yaxis_title="RSI",
            height=250
        )
        st.plotly_chart(fig_rsi, use_container_width=True)


# ================================
# DASHBOARD STREAMLIT
# ================================

def main():
    st.set_page_config(page_title="Dashboard SMI (court / long terme)", layout="wide")

    st.title("📈 Dashboard d'analyse technique – SMI")
    st.caption("⚠️ Outil pédagogique – pas un conseil d’investissement.")

    # --- SIDEBAR ---
    st.sidebar.header("Paramètres")

    tickers_str = st.sidebar.text_input(
        "Liste des tickers (SMI, séparés par des virgules)",
        value=",".join(SMI_TICKERS)
    )
    tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]

    years = st.sidebar.slider("Nombre d'années d'historique", 1, 5, DATA_PERIOD_YEARS)

    st.sidebar.write("Univers analysé :")
    st.sidebar.write(tickers)

    # --- DONNÉES ---
    st.info("Téléchargement des données en cours...")
    data = download_data(tickers, years=years)
    st.success(f"Données téléchargées pour {len(data)} ticker(s).")

    if not isinstance(data, dict) or len(data) == 0:
        st.error("Aucune donnée disponible. Vérifie les tickers.")
        return

    scores_df = build_scores(data)
    if scores_df.empty:
        st.error("Impossible de calculer les scores.")
        return

    # --- TOP 3 COURT TERME ---
    st.subheader("🏃‍♂️ Top 3 court terme (horizon ~1 mois)")
    top_short = scores_df.sort_values("Score_Court_Terme", ascending=False).head(3)
    st.dataframe(
        top_short[["Ticker", "Dernier_Prix", "Score_Court_Terme", "RSI14", "MACD", "EMA20", "EMA50"]],
        use_container_width=True
    )

    # --- TOP 3 LONG TERME ---
    st.subheader("📅 Top 3 long terme (horizon ~1 an)")
    top_long = scores_df.sort_values("Score_Long_Terme", ascending=False).head(3)
    st.dataframe(
        top_long[["Ticker", "Dernier_Prix", "Score_Long_Terme", "RSI14", "EMA50", "EMA200"]],
        use_container_width=True
    )

    # --- DÉTAIL D'UNE ACTION ---
    st.subheader("🔍 Détail d'une action")

    all_tickers_available = list(data.keys())
    selected_ticker = st.selectbox("Choisir un ticker", all_tickers_available)

    df_sel = data.get(selected_ticker)
    if df_sel is None or df_sel.empty:
        st.warning("Pas de données pour ce ticker.")
        return

    col1, col2 = st.columns(2)
    with col1:
        last_price = safe_float(df_sel["Close"].iloc[-1])
        if last_price is not None:
            st.metric("Prix actuel", f"{last_price:.2f}")
        else:
            st.metric("Prix actuel", "N/A")
    with col2:
        st.write("Derniers indicateurs :")
        st.dataframe(
            scores_df[scores_df["Ticker"] == selected_ticker].T,
            use_container_width=True
        )

    plot_candles_with_indicators(df_sel, selected_ticker)

    st.caption("Ceci reste un outil d'analyse, pas une garantie de performance 😉")


if __name__ == "__main__":
    main()
