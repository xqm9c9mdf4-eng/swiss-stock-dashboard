import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objs as go
import requests
from datetime import date, timedelta

# ================================
# PARAMÈTRES GLOBAUX / FALLBACK
# ================================

# Fallback : principales valeurs du SMI si on n'arrive pas à charger le SPI
SMI_TICKERS = [
    "NESN.SW",  # Nestlé
    "NOVN.SW",  # Novartis
    "ROG.SW",   # Roche
    "UBSG.SW",  # UBS
    "ZURN.SW",  # Zurich
    "SIKA.SW",  # Sika
    "ABBN.SW",  # ABB
    "SREN.SW",  # Swiss Re
    "LOGN.SW",  # Logitech
    "HOLN.SW",  # Holcim
]

DATA_PERIOD_YEARS = 2  # historique utilisé pour les indicateurs
SPI_WIKI_URL = "https://en.wikipedia.org/wiki/Swiss_Performance_Index"


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


@st.cache_data
def get_spi_tickers_from_web() -> list[str]:
    """
    Récupère la liste des composants du SPI depuis Wikipédia
    et les transforme en tickers Yahoo Finance (xxx.SW).
    Si échec, renvoie la liste SMI_TICKERS.
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        }
        resp = requests.get(SPI_WIKI_URL, headers=headers, timeout=10)
        resp.raise_for_status()

        tables = pd.read_html(resp.text)
        for tbl in tables:
            if "Symbol" in tbl.columns:
                symbols = (
                    tbl["Symbol"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .str.replace(" ", "", regex=False)
                    .str.upper()
                    .unique()
                )
                tickers = [s + ".SW" for s in symbols]
                tickers = sorted({t for t in tickers if t})
                if len(tickers) > 0:
                    return tickers

        st.warning(
            "Impossible de trouver la colonne 'Symbol' pour le SPI sur Wikipédia. "
            "Utilisation de la liste SMI par défaut."
        )
        return SMI_TICKERS.copy()

    except Exception as e:
        st.warning(
            f"Impossible de charger la liste SPI en ligne : {e}. "
            "Utilisation de la liste SMI par défaut."
        )
        return SMI_TICKERS.copy()


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


def compute_signal_and_levels(
    close: float | None,
    ema50: float | None,
    ema200: float | None,
    rsi: float | None,
    macd: float | None,
    score_ct: float,
    score_lt: float,
) -> tuple[str, float | None, float | None]:
    """
    Retourne (signal, stop_loss, target_price)
    Règles simples :
      - BUY si scores élevés + structure haussière propre
      - SELL si scores faibles + RSI haut
      - sinon HOLD
    Stop-loss = sous un support technique (EMA50/EMA200) avec petite marge
    Target    = R:R ≈ 2:1 par rapport au risque
    """
    # -------- SIGNAL --------
    signal = "HOLD"

    if (
        score_lt >= 9
        and score_ct >= 8
        and rsi is not None
        and 40 <= rsi <= 65
        and macd is not None
        and macd > 0
        and close is not None
        and ema50 is not None
        and ema200 is not None
        and close > ema50 > ema200
    ):
        signal = "BUY"
    elif (
        score_lt <= 4
        and score_ct <= 4
        and rsi is not None
        and rsi > 65
    ):
        signal = "SELL"

    # -------- NIVEAUX PRIX --------
    if close is None:
        return signal, None, None

    # Support technique = plus bas de EMA50/EMA200 si dispo, sinon -10 %
    supports = [v for v in [ema50, ema200] if v is not None]
    if supports:
        technical_support = min(supports)
    else:
        technical_support = close * 0.9

    stop_loss = round(technical_support * 0.97, 2)  # petite marge sous le support

    # cible = RR 2:1
    risk_per_share = close - stop_loss
    if risk_per_share <= 0:
        target_price = None
    else:
        target_price = round(close + 2 * risk_per_share, 2)

    return signal, stop_loss, target_price


def build_scores(data: dict) -> pd.DataFrame:
    rows = []

    for ticker, df in data.items():
        if df is None or df.empty:
            continue

        last = df.iloc[-1]
        close = safe_float(last.get("Close"))
        rsi = safe_float(last.get("RSI14"))
        macd = safe_float(last.get("MACD"))
        ema20 = safe_float(last.get("EMA20"))
        ema50 = safe_float(last.get("EMA50"))
        ema200 = safe_float(last.get("EMA200"))
        volume = safe_float(last.get("Volume"))
        vol_moy20 = safe_float(last.get("Vol_Moy20"))

        score_ct = score_short_term(df)
        score_lt = score_long_term(df)
        score_total = score_ct + score_lt

        signal, stop_loss, target_price = compute_signal_and_levels(
            close, ema50, ema200, rsi, macd, score_ct, score_lt
        )

        row = {
            "Ticker": ticker,
            "Dernier_Prix": close,
            "RSI14": rsi,
            "MACD": macd,
            "MACD_signal": safe_float(last.get("MACD_signal")),
            "EMA20": ema20,
            "EMA50": ema50,
            "EMA200": ema200,
            "Volume": volume,
            "Volume_Moy20": vol_moy20,
            "Score_Court_Terme": score_ct,
            "Score_Long_Terme": score_lt,
            "Score_Total": score_total,
            "Signal": signal,
            "Stop_Loss": stop_loss,
            "Cible_Prix": target_price,
        }
        rows.append(row)

    if len(rows) == 0:
        return pd.DataFrame()

    scores_df = pd.DataFrame(rows)
    scores_df = scores_df.sort_values("Score_Total", ascending=False).reset_index(drop=True)
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


def plot_score_radar(score_ct: float, score_lt: float, ticker: str):
    """Petit graphique comparatif Court vs Long terme pour 1 action."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Court terme", "Long terme"],
        y=[score_ct, score_lt],
        text=[f"{score_ct:.1f}", f"{score_lt:.1f}"],
        textposition="auto",
        name="Scores"
    ))
    fig.update_layout(
        title=f"Scores court / long terme – {ticker}",
        yaxis_title="Score",
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_global_ranking(scores_df: pd.DataFrame, n: int = 20):
    """Bar chart des meilleurs scores totaux."""
    top = scores_df.sort_values("Score_Total", ascending=False).head(n)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top["Ticker"],
        y=top["Score_Total"],
        text=[f"{v:.1f}" for v in top["Score_Total"]],
        textposition="auto",
        name="Score total"
    ))
    fig.update_layout(
        title=f"Top {n} – Score global (court + long terme)",
        xaxis_title="Ticker",
        yaxis_title="Score total",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


# ================================
# DASHBOARD STREAMLIT
# ================================

def main():
    st.set_page_config(page_title="Dashboard actions suisses (SPI)", layout="wide")

    st.title("📈 Dashboard d'analyse technique – Actions suisses (SPI)")
    st.caption("⚠️ Outil pédagogique – pas un conseil d’investissement.")

    # --- SIDEBAR ---
    st.sidebar.header("Paramètres")

    default_tickers = get_spi_tickers_from_web()
    default_str = ",".join(default_tickers)

    tickers_str = st.sidebar.text_area(
        "Liste des tickers (séparés par des virgules)",
        value=default_str,
        height=150,
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

    # --- SCORING GLOBAL GRAPHIQUE ---
    plot_global_ranking(scores_df, n=20)

    # --- TOP 3 COURT TERME ---
    st.subheader("🏃‍♂️ Top 3 court terme (horizon ~1 mois)")
    top_short = scores_df.sort_values("Score_Court_Terme", ascending=False).head(3)
    st.dataframe(
        top_short[
            [
                "Ticker",
                "Dernier_Prix",
                "Score_Court_Terme",
                "Signal",
                "RSI14",
                "MACD",
                "EMA20",
                "EMA50",
                "Stop_Loss",
                "Cible_Prix",
            ]
        ],
        use_container_width=True,
    )

    # --- TOP 3 LONG TERME ---
    st.subheader("📅 Top 3 long terme (horizon ~1 an)")
    top_long = scores_df.sort_values("Score_Long_Terme", ascending=False).head(3)
    st.dataframe(
        top_long[
            [
                "Ticker",
                "Dernier_Prix",
                "Score_Long_Terme",
                "Signal",
                "RSI14",
                "EMA50",
                "EMA200",
                "Stop_Loss",
                "Cible_Prix",
            ]
        ],
        use_container_width=True,
    )

    # --- DÉTAIL D'UNE ACTION ---
    st.subheader("🔍 Détail d'une action")

    all_tickers_available = list(data.keys())
    selected_ticker = st.selectbox("Choisir un ticker", all_tickers_available)

    df_sel = data.get(selected_ticker)
    if df_sel is None or df_sel.empty:
        st.warning("Pas de données pour ce ticker.")
        return

    selected_row = scores_df[scores_df["Ticker"] == selected_ticker].iloc[0]

    col1, col2, col3 = st.columns(3)
    with col1:
        last_price = selected_row["Dernier_Prix"]
        if last_price is not None:
            st.metric("Prix actuel", f"{last_price:.2f}")
        else:
            st.metric("Prix actuel", "N/A")

    with col2:
        st.metric("Signal", selected_row["Signal"])

    with col3:
        sl = selected_row["Stop_Loss"]
        tp = selected_row["Cible_Prix"]
        st.write(f"Stop-loss : **{sl:.2f}**" if sl is not None else "Stop-loss : N/A")
        st.write(f"Cible : **{tp:.2f}**" if tp is not None else "Cible : N/A")

    st.write("Données détaillées :")
    st.dataframe(
        selected_row.to_frame("Valeur"),
        use_container_width=True,
    )

    plot_score_radar(
        selected_row["Score_Court_Terme"],
        selected_row["Score_Long_Terme"],
        selected_ticker,
    )

    plot_candles_with_indicators(df_sel, selected_ticker)

    st.caption("Ceci reste un outil d'analyse, pas une garantie de performance 😉")


if __name__ == "__main__":
    main()
