import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import ta
import numpy as np
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import warnings
from yahooquery import Ticker

# Ignore warnings from ta library
warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────────────────────────
#  Dash App Initialization
# ────────────────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server  # Expose Flask server for deployment platforms

# ────────────────────────────────────────────────────────────────────────────────
#  Layout
# ────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
#  Refreshed “single‑card” layout
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
#  Refreshed “single‑card” layout  (spinner fixed)
# ─────────────────────────────────────────────────────────────────────────────
app.layout = dbc.Container(fluid=True, children=[

    # ── Title row ────────────────────────────────────────────────────────────
    dbc.Row(
        dbc.Col(
            html.H1("Advanced Stock Analysis App",
                    className="text-center mt-4 mb-3 text-primary"),
            width=12)
    ),

    # ── Main card holding sidebar + content ──────────────────────────────────
    dbc.Card(className="shadow-sm mb-4", children=[
        dbc.CardBody([
            dbc.Row([

                # ░░ Sidebar ░░──────────────────────────────────────────────
                dbc.Col(width=3, children=[
                    html.H4("Inputs", className="mb-3"),

                    dbc.Label("Ticker (no .SR for Saudi):"),
                    dbc.Input(id="stock-symbol", type="text", value="1303",
                              placeholder="e.g. AAPL or 1303", debounce=True,
                              className="mb-3"),

                    dbc.Label("Time Period:"),
                    dcc.Dropdown(
                        id="time-period",
                        options=[
                            {"label": "1 Year",  "value": "1y"},
                            {"label": "18 Months", "value": "18mo"},
                            {"label": "2 Years", "value": "2y"},
                            {"label": "30 Months", "value": "30mo"},
                            {"label": "3 Years", "value": "3y"},
                            {"label": "Max",    "value": "max"}
                        ],
                        value="18mo",
                        className="mb-3"
                    ),

                    dbc.Label("SMA Short:"),
                    dbc.Input(id="sma-short", type="number", value=7,
                              min=1, debounce=True, className="mb-3"),

                    dbc.Label("SMA Long:"),
                    dbc.Input(id="sma-long", type="number", value=10,
                              min=1, debounce=True, className="mb-3"),

                    dbc.Label("RSI Threshold:"),
                    dbc.Input(id="rsi-threshold", type="number", value=40,
                              min=0, max=100, debounce=True, className="mb-3"),

                    dbc.Label("ADL Short:"),
                    dbc.Input(id="adl-short", type="number", value=13,
                              min=1, debounce=True, className="mb-3"),

                    dbc.Label("ADL Long:"),
                    dbc.Input(id="adl-long", type="number", value=30,
                              min=1, debounce=True, className="mb-4"),

                    dbc.Button("Analyze Stock", id="submit-button",
                               color="primary", className="w-100"),

                    # Fixed spinner (use spinnerClassName instead of className)
                    dbc.Spinner(
                        html.Div(id="loading-output"),
                        size="sm",
                        color="primary",
                        spinnerClassName="mt-3"   # margin‑top
                    ),
                ]),

                # ░░ Content ░░─────────────────────────────────────────────
                dbc.Col(width=9, children=[
                    dbc.Tabs(id="output-tabs", active_tab="tab-price-chart",
                             className="mb-3", children=[

                        dbc.Tab(label="Price Chart",
                                tab_id="tab-price-chart",
                                children=dcc.Graph(id="stock-graph",
                                                   style={"height": "70vh"})),

                        dbc.Tab(label="Performance Metrics",
                                tab_id="tab-metrics",
                                children=html.Div(id="performance-metrics",
                                                  className="p-3")),

                        dbc.Tab(label="Trade Details",
                                tab_id="tab-trades",
                                children=html.Div(id="trade-details",
                                                  className="p-3")),
                    ])
                ])
            ])
        ])
    ])
])

# ────────────────────────────────────────────────────────────────────────────────
#  Helper: fetch data via yahooquery
# ────────────────────────────────────────────────────────────────────────────────

def fetch_stock_data(symbol: str, period_code: str) -> pd.DataFrame:
    """Download price history using yahooquery and standardize column names."""
    today = datetime.today()

    tq = Ticker(symbol)

    # Direct "period" if supported -------------------------------------------------
    if period_code in {"1y", "2y", "3y", "max"}:
        df = tq.history(period=period_code)
    else:
        # Map special codes to start‑date offsets ----------------------------------
        offsets = {
            "18mo": today - timedelta(days=18 * 30),
            "30mo": today - timedelta(days=30 * 30)
        }
        start_date = offsets.get(period_code, today - timedelta(days=18 * 30))
        df = tq.history(start=start_date.strftime("%Y-%m-%d"), end=today.strftime("%Y-%m-%d"))

    if df.empty:
        return df

    # yahooquery returns a MultiIndex (symbol, date) -------------------------------
    df = df.reset_index()
    if "symbol" in df.columns:
        df = df.drop(columns=["symbol"])
    df.set_index("date", inplace=True)
    df.index = pd.to_datetime(df.index)

    df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adjclose": "Adj Close",
        "volume": "Volume",
    }, inplace=True)

    return df

# ────────────────────────────────────────────────────────────────────────────────
#  Callbacks
# ────────────────────────────────────────────────────────────────────────────────

@app.callback(
    [Output("stock-graph", "figure"),
     Output("performance-metrics", "children"),
     Output("trade-details", "children"),
     Output("loading-output", "children")],
    Input("submit-button", "n_clicks"),
    State("stock-symbol", "value"),
    State("time-period", "value"),
    State("sma-short", "value"),
    State("sma-long", "value"),
    State("rsi-threshold", "value"),
    State("adl-short", "value"),
    State("adl-long", "value")
)
def update_output(n_clicks, stock_symbol, time_period,
                  sma_short, sma_long, rsi_threshold, adl_short, adl_long):
    if n_clicks is None:
        return dash.no_update, "", "", ""

    try:
        # Append ".SR" to numeric symbols for Saudi exchange ------------------
        if stock_symbol.isdigit():
            stock_symbol += ".SR"

        # ── Fetch data ---------------------------------------------------------
        df = fetch_stock_data(stock_symbol, time_period)

        if df.empty:
            msg = html.P("No data found for the provided stock symbol and time period.",
                          className="text-danger")
            return dash.no_update, msg, "", ""

        # ── Process data & strategy evaluation --------------------------------
        result = process_stock(df.copy(), stock_symbol, sma_short, sma_long,
                               rsi_threshold, adl_short, adl_long)
        df = result["df"]
        buy_signals  = df[df["Buy Signal"]]
        sell_signals = df[df["Sell Signal"]]
        fig = go.Figure()

        # 1) Close price  – solid blue
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["Close"],
                mode="lines", name="Close Price",
                line=dict(color="blue", width=2)
            )
        )

        # 2) SMA‑Short  – dashed yellow
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["SMA_Short"],
                mode="lines", name=f"SMA Short ({sma_short})",
                line=dict(color="yellow", dash="dash", width=1.5)
            )
        )

        # 3) SMA‑Long  – dotted green
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["SMA_Long"],
                mode="lines", name=f"SMA Long ({sma_long})",
                line=dict(color="green", dash="dot", width=1.5)
            )
        )

        # 4) Buy signals  – green triangles pointing up
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index, y=buy_signals["Close"],
                mode="markers", name="Buy Signal",
                marker=dict(symbol="triangle-up", size=10, color="green")
            )
        )

        # 5) Sell signals  – red triangles pointing down
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index, y=sell_signals["Close"],
                mode="markers", name="Sell Signal",
                marker=dict(symbol="triangle-down", size=10, color="red")
            )
        )

        fig.update_layout(
            title={"text": "Price with Buy and Sell Signals", "y": 0.98,
                   "x": 0.5, "xanchor": "center", "yanchor": "top"},
            xaxis_title="Date", yaxis_title="Price", template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                         xanchor="center", x=0.5),
            margin=dict(l=40, r=40, t=120, b=40)
        )

        # ── Performance metrics -----------------------------------------------
# Prepare performance metrics
        # Calculate average percentage profit from all trades
        if result['trades']:
            trades_df = pd.DataFrame(result['trades'])
            average_profit_percentage = trades_df['Profit Percentage'].mean()
            average_profit_percentage_str = f"{average_profit_percentage:.2f}%"
        else:
            average_profit_percentage_str = "N/A"

        metrics = [
            html.H5("Performance Metrics", className="card-title"),
            html.Ul([
                html.Li(f"Initial Investment: {100000:,.2f} SAR"),
                html.Li(f"Final Portfolio Value: {result['final_value']:,.2f} SAR"),
                html.Li(f"Total Return: {result['final_value'] - 100000:,.2f} SAR"),
                html.Li(f"Percentage Return: {result['percentage_return']:.2f}%"),
                html.Li(f"Number of Trades: {result['number_of_trades']}"),
                html.Li(f"Average Days Held per Trade: {result['average_days_held']:.2f} days"),
                html.Li(f"Average Profit per Trade: {average_profit_percentage_str}"),
                html.Li(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}"),
                html.Li(f"Win Rate: {result['win_rate']*100:.2f}%"),
            ], className="list-unstyled mt-3"),
        ]

        # Prepare trade details
        if result['trades']:
            trades_df = pd.DataFrame(result['trades'])
            # Format columns for better readability
            trades_df['Buy Price'] = trades_df['Buy Price'].map('{:,.2f} SAR'.format)
            trades_df['Sell Price'] = trades_df['Sell Price'].map('{:,.2f} SAR'.format)
            trades_df['Amount Invested'] = trades_df['Amount Invested'].map('{:,.2f} SAR'.format)
            trades_df['Profit'] = trades_df['Profit'].map('{:,.2f} SAR'.format)
            trades_df['Profit Percentage'] = trades_df['Profit Percentage'].map('{:.2f}%'.format)
            trades_df['Days Held'] = trades_df['Days Held'].astype(int)

            trades_table = dbc.Table.from_dataframe(trades_df, striped=True, bordered=True, hover=True, responsive=True, className="mt-3")
            trade_details = [
                html.H5("Trade Details", className="card-title"),
                trades_table
            ]
        else:
            trade_details = [html.P("No trades were made with the given parameters.", className="mt-3")]

        return fig, metrics, trade_details, ""
    except Exception as e:
        return dash.no_update, html.P(f"An error occurred: {str(e)}", className="text-danger mt-3"), "", ""

# Function to process stock data
def process_stock(df, stock_symbol, sma_short, sma_long, rsi_threshold, adl_short, adl_long, initial_investment=100000):
    # Calculate indicators
    df['SMA_Short'] = df['Close'].rolling(window=int(sma_short)).mean()
    df['SMA_Long'] = df['Close'].rolling(window=int(sma_long)).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['ADL'] = ta.volume.AccDistIndexIndicator(
        df['High'], df['Low'], df['Close'], df['Volume']
    ).acc_dist_index()
    df['ADL_Short_SMA'] = df['ADL'].rolling(window=int(adl_short)).mean()
    df['ADL_Long_SMA'] = df['ADL'].rolling(window=int(adl_long)).mean()

    # Signal generation
    df['Signal'] = df.apply(
        lambda row: -1 if row['Close'] >= row['SMA_Short']
                        and row['SMA_Short'] > row['SMA_Long']
                        and row['ADL_Short_SMA'] > row['ADL_Long_SMA']
                        and row['RSI'] >= int(rsi_threshold)
                        and row['MACD'] > row['MACD_Signal'] else (
            1 if row['Close'] < row['SMA_Short']
                    and row['SMA_Short'] < row['SMA_Long'] else 0
        ), axis=1
    )

    # Initialize portfolio variables
    position = 0
    cash = initial_investment
    portfolio_value = initial_investment
    buy_price = None
    trade_start = None
    trades = []
    number_of_trades = 0
    portfolio_values = []
    amount_invested = 0

    for index, row in df.iterrows():
        if row['Signal'] == 1 and position == 0:
            # Buy
            buy_price = row['Close']
            position = cash / buy_price
            amount_invested = cash  # Amount invested in this trade
            cash = 0
            trade_start = index
            number_of_trades += 1
        elif row['Signal'] == -1 and position > 0:
            # Sell
            sell_price = row['Close']
            cash = position * sell_price
            profit = cash - amount_invested  # Profit from this trade
            profit_percentage = ((sell_price - buy_price) / buy_price) * 100
            days_held = (index - trade_start).days

            trades.append({
                'Sell Date': index.date(),
                'Buy Price': buy_price,
                'Sell Price': sell_price,
                'Days Held': days_held,
                'Amount Invested': amount_invested,
                'Profit': profit,
                'Profit Percentage': profit_percentage
            })

            position = 0
            buy_price = None
            amount_invested = 0
        else:
            # Update portfolio value if holding a position
            if position > 0:
                portfolio_value = position * row['Close']
            else:
                portfolio_value = cash

        portfolio_values.append(portfolio_value)

    # Final calculations
    final_value = portfolio_values[-1] if portfolio_values else initial_investment
    total_return = final_value - initial_investment
    percentage_return = (total_return / initial_investment) * 100

    # Calculate daily returns
    portfolio_values_array = np.array(portfolio_values)
    daily_returns = pd.Series(portfolio_values_array).pct_change().fillna(0)

    # Assuming risk-free rate is zero
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0

    # Win rate
    profitable_trades = [t for t in trades if t['Profit'] > 0]
    win_rate = len(profitable_trades) / number_of_trades if number_of_trades > 0 else 0

    average_days_held = sum([t['Days Held'] for t in trades]) / number_of_trades if number_of_trades > 0 else 0

    # Add buy/sell signals for plotting
    df['Buy Signal'] = df['Signal'] == 1
    df['Sell Signal'] = df['Signal'] == -1

    return {
        'final_value': final_value,
        'percentage_return': percentage_return,
        'number_of_trades': number_of_trades,
        'average_days_held': average_days_held,
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate,
        'trades': trades,
        'portfolio_values': portfolio_values,
        'df': df
    }

if __name__ == '__main__':
    app.run_server(debug=True)
    
