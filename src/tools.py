from langchain.tools import tool
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

@tool
def search_ticket(name) -> str:
    """Here, a ticket will be searched by its name, and a list of the first three possible matching tickets will be returned."""
    corresponding_ticket = yf.Search(name).quotes

    return str(corresponding_ticket[:3])

@tool
def get_ticker_values(ticket_and_period: str) -> str:
    """Gets ticker price. Input format: 'TICKER PERIOD' (ex: 'BTC-USD 5d', 'XP 3d', 'ITUB4.SA 1mo').
    Period: number + d/mo/y (ex: 5d, 3mo, 1y)."""
    parts = ticket_and_period.strip().strip("'\"").split()
    ticket = parts[0]
    period = parts[1] if len(parts) > 1 else "5d"

    result = yf.download(ticket, period=period.lower(), threads=False)
    result.columns = result.columns.droplevel(1)
    result_final = result[["Close", "Volume"]].ffill().dropna().to_string()


    return result_final

@tool
def calculate_indicaters(ticket_and_period: str) -> str:
    """Calculates RSI, moving averages, and volatility for a ticker. Input format: 'TICKER PERIOD' (ex: 'BTC-USD 1y', 'XP 6mo', 'ITUB4.SA 2y'). Period: number + d/mo/y. Default period and minimun is 1 year if not specified."""
    parts = ticket_and_period.strip().strip("'\"").split()
    ticket = parts[0]
    period = parts[1] if len(parts) > 1 else "1y"

    result = yf.download(ticket, period=period.lower(), threads=False)
    result.columns = result.columns.droplevel(1)
    result_final = result.ffill().dropna()


    # Calc for RSI
    diference: list[float] = np.diff(result['Close'])
    highs: list[float] = np.where(diference > 0, diference, 0)
    highs = np.insert(highs, 0, 0)
    lows: list[float] = np.where(diference < 0, np.abs(diference), 0)
    lows = np.insert(lows, 0, 0)
    mean_gain = pd.Series(highs).rolling(window=14).mean()
    mean_lost = pd.Series(lows).rolling(window=14).mean()
    RS = mean_gain / mean_lost
    RSI = 100 - (100 / (1 + RS))

    #Calc for Averages
    result['MME80'] = result['Close'].ewm(span=80, min_periods=80, adjust=False).mean()
    averages = result['MME80'].values

    #Calc for Volatilitys
    result['log_returns'] = np.log(result['Close'] / result['Close'].shift(1))
    result['VOL80'] = result['log_returns'].rolling(window=80).std() * (365**0.5)
    volatilitys = result['VOL80'].values

    df_final = pd.DataFrame({
        'RSI': RSI,
        'Averages': averages,
        'Volatilitys': volatilitys 
    })

    df_final = df_final.dropna().tail(5).to_string()

    return df_final

@tool
def calculate_indicators(ticket_and_period: str) -> str:
    """Trains an XGBoost classifier and predicts if the price will rise or fall tomorrow. Input format: 'TICKER PERIOD' (ex: 'BTC-USD 10y', 'XP 5y', 'ITUB4.SA 10y'). Period: number + d/mo/y. Default period and minimun is 10 years if not specified."""
    parts = ticket_and_period.strip().strip("'\"").split()
    ticket = parts[0]
    period = parts[1] if len(parts) > 1 else "10y"
    
    #Download ticker
    result = yf.download(ticket, period=period.lower(), threads=False)
    result.columns = result.columns.droplevel(1)
    result_final = result.ffill().dropna()


    # Calc for RSI
    diference: list[float] = np.diff(result['Close'])
    highs: list[float] = np.where(diference > 0, diference, 0)
    highs = np.insert(highs, 0, 0)
    lows: list[float] = np.where(diference < 0, np.abs(diference), 0)
    lows = np.insert(lows, 0, 0)
    mean_gain = pd.Series(highs).rolling(window=14).mean()
    mean_lost = pd.Series(lows).rolling(window=14).mean()
    RS = mean_gain / mean_lost
    RSI = 100 - (100 / (1 + RS))

    #Calc for Averages
    result['MME80'] = result['Close'].ewm(span=80, min_periods=80, adjust=False).mean()
    averages = result['MME80'].values

    #Calc for Volatilitys
    result['log_returns'] = np.log(result['Close'] / result['Close'].shift(1))
    result['VOL80'] = result['log_returns'].rolling(window=80).std() * (365**0.5)
    volatilitys = result['VOL80'].values

    returning = (result['Close'].shift(-1) - result['Close']) / result['Close'] * 100
    target = np.where(returning > 0, 1, 0)

    df_final = pd.DataFrame({
        'RSI': RSI,
        'Averages': averages,
        'Volatilitys': volatilitys,
        'Target': target
    })


    #Treino
    X = df_final.drop('Target', axis=1).values
    Y = df_final['Target'].values

    x_training = X[:-1, :]
    y_training = Y[:-1]
    history  = []

    model = XGBClassifier(n_estimators=100, learning_rate=0.1)
    model.fit(x_training, y_training)

    y_pred = model.predict(x_training)
    accuracy = accuracy_score(y_training, y_pred)
    history.append(accuracy)

    
    # Teste
    ultimo_dia = X[-1:, :]
    predicao = model.predict(ultimo_dia)[0]
    confianca = model.predict_proba(ultimo_dia)[0][1]

    direcao = "ALTA" if predicao == 1 else "BAIXA"
    return f"Tendência de {direcao} com {confianca:.1%} de confiança."