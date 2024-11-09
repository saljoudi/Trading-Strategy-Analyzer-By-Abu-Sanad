import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import yfinance as yf
import ta
import plotly.graph_objs as go
import numpy as np

# Initialize the Dash app with a Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server  # Expose the Flask server

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Advanced Trading Strategy Analyzer", className="text-center text-primary"), className="mb-4 mt-4")
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Input Parameters", className="card-title"),
                    dbc.Label("Ticker Symbol (without .SR for Saudi stocks):"),
                    dcc.Input(id='ticker-input', type='text', value='1303', className="mb-3", style={'width': '100%'}),
                    dbc.Label("Period:"),
                    dcc.Dropdown(
                        id='period-input',
                        options=[
                            {'label': '1 Year', 'value': '1y'},
                            {'label': '2 Years', 'value': '2y'},
                            {'label': '5 Years', 'value': '5y'},
                            {'label': 'All', 'value': 'max'}
                        ],
                        value='1y',
                        className="mb-3",
                        style={'width': '100%'}
                    ),
                    dbc.Label("Short SMA Period:"),
                    dcc.Input(id='sma-short-input', type='number', value=7, className="mb-3", style={'width': '100%'}),
                    dbc.Label("Long SMA Period:"),
                    dcc.Input(id='sma-long-input', type='number', value=10, className="mb-3", style={'width': '100%'}),
                    dbc.Label("RSI Threshold:"),
                    dcc.Input(id='rsi-threshold-input', type='number', value=40, className="mb-3", style={'width': '100%'}),
                    dbc.Label("Short ADL SMA Period:"),
                    dcc.Input(id='adl-short-input', type='number', value=19, className="mb-3", style={'width': '100%'}),
                    dbc.Label("Long ADL SMA Period:"),
                    dcc.Input(id='adl-long-input', type='number', value=25, className="mb-3", style={'width': '100%'}),
                    dbc.Button("Analyze", id="analyze-button", color="primary", className="mt-3", style={'width': '100%'})
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Trading Strategy Graph", className="card-title"),
                    dcc.Graph(id='trading-graph')
                ])
            ])
        ], width=9)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Best Strategy Summary", className="card-title"),
                    html.Pre(id='summary-output', style={'whiteSpace': 'pre-wrap', 'font-family': 'monospace'})
                ])
            ])
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Best Trades Details", className="card-title"),
                    html.Div(id='trades-table')
                ])
            ])
        ], width=12)
    ])
], fluid=True)


@app.callback(
    [Output('trading-graph', 'figure'),
     Output('summary-output', 'children'),
     Output('trades-table', 'children')],
    [Input('analyze-button', 'n_clicks')],
    [Input('ticker-input', 'value'),
     Input('period-input', 'value'),
     Input('sma-short-input', 'value'),
     Input('sma-long-input', 'value'),
     Input('rsi-threshold-input', 'value'),
     Input('adl-short-input', 'value'),
     Input('adl-long-input', 'value')]
)
def update_graph(n_clicks, ticker_input, period, sma_short, sma_long, rsi_threshold, adl_short, adl_long):
    # Check if the ticker is numeric (Saudi stock symbol)
    if ticker_input.isdigit():
        ticker = f"{ticker_input}.SR"
    else:
        ticker = ticker_input

    # Download the data for the ticker
    df = yf.download(ticker, period=period)
    df.index = pd.to_datetime(df.index)

    if df.empty:
        return {}, "No data available for the ticker.", ""

    # Calculate indicators
    df['SMA_Short'] = df['Close'].rolling(window=sma_short).mean()
    df['SMA_Long'] = df['Close'].rolling(window=sma_long).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['MACD_Signal'] = ta.trend.MACD(df['Close']).macd_signal()
    df['ADL'] = ta.volume.AccDistIndexIndicator(df['High'], df['Low'], df['Close'], df['Volume']).acc_dist_index()
    df['ADL_Short_SMA'] = df['ADL'].rolling(window=adl_short).mean()
    df['ADL_Long_SMA'] = df['ADL'].rolling(window=adl_long).mean()

    # Signal generation
    df['Signal'] = df.apply(
        lambda row: -1 if row['Close'] >= row['SMA_Short'] and row['SMA_Short'] > row['SMA_Long'] and row['ADL_Short_SMA'] > row['ADL_Long_SMA'] and row['RSI'] >= rsi_threshold and row['MACD'] > row['MACD_Signal'] else (
            1 if row['Close'] < row['SMA_Short'] and row['SMA_Short'] < row['SMA_Long'] else 0
        ), axis=1
    )

    # Initialize trading simulation variables
    initial_investment = 100000
    portfolio_value = initial_investment
    position = 0  # 0 for no position, 1 for holding shares
    number_of_shares = 0
    trades = []
    trade_returns = []
    number_of_trades = 0
    df['Portfolio Value'] = initial_investment

    # Simulate trading
    for index, row in df.iterrows():
        if row['Signal'] == 1 and position == 0:
            # Buy signal
            buy_price = row['Close']
            number_of_shares = portfolio_value / buy_price
            position = 1
            trade_start = index
            number_of_trades += 1
        elif row['Signal'] == -1 and position == 1:
            # Sell signal
            sell_price = row['Close']
            profit = number_of_shares * (sell_price - buy_price)
            portfolio_value = number_of_shares * sell_price
            days_held = (index - trade_start).days

            profit_percentage = ((sell_price - buy_price) / buy_price) * 100
            trade_returns.append((sell_price - buy_price) / buy_price)

            trades.append({
                'Sell Date': index.date().strftime('%Y-%m-%d'),
                'Buy Price': f"{buy_price:.2f} SAR",
                'Sell Price': f"{sell_price:.2f} SAR",
                'Days Held': days_held,
                'Profit': f"{profit:,.2f} SAR",
                'Profit Percentage': f"{profit_percentage:.2f}%"
            })

            position = 0
            number_of_shares = 0
        # Update portfolio value
        if position == 1:
            portfolio_value = number_of_shares * row['Close']
        # Record portfolio value
        df.loc[index, 'Portfolio Value'] = portfolio_value

    # Handle the case where we still hold a position at the end
    if position == 1:
        sell_price = df.iloc[-1]['Close']
        profit = number_of_shares * (sell_price - buy_price)
        portfolio_value = number_of_shares * sell_price
        days_held = (df.index[-1] - trade_start).days

        profit_percentage = ((sell_price - buy_price) / buy_price) * 100
        trade_returns.append((sell_price - buy_price) / buy_price)

        trades.append({
            'Sell Date': df.index[-1].date().strftime('%Y-%m-%d'),
            'Buy Price': f"{buy_price:.2f} SAR",
            'Sell Price': f"{sell_price:.2f} SAR",
            'Days Held': days_held,
            'Profit': f"{profit:,.2f} SAR",
            'Profit Percentage': f"{profit_percentage:.2f}%"
        })

        position = 0
        number_of_shares = 0

    final_value = portfolio_value
    total_return = final_value - initial_investment
    percentage_return = (total_return / initial_investment) * 100

    # Calculate daily returns and Sharpe Ratio
    df['Daily Return'] = df['Portfolio Value'].pct_change()
    mean_daily_return = df['Daily Return'].mean()
    std_daily_return = df['Daily Return'].std()
    if std_daily_return != 0:
        sharpe_ratio = (mean_daily_return / std_daily_return) * np.sqrt(252)
    else:
        sharpe_ratio = 0

    # Calculate average trade return percentage
    if trade_returns:
        avg_trade_return_percentage = (np.mean(trade_returns)) * 100
    else:
        avg_trade_return_percentage = 0

    # Create the plot with enhanced visuals
    fig = go.Figure()

    # Add the Closing Price and SMA lines
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_Short'], mode='lines', name=f'SMA Short ({sma_short})', line=dict(color='orange', dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_Long'], mode='lines', name=f'SMA Long ({sma_long})', line=dict(color='green', dash='dot')))

    # Highlight Buy and Sell signals
    buy_signals = df[df['Signal'] == 1]
    sell_signals = df[df['Signal'] == -1]

    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', name='Buy Signal',
                             marker=dict(color='green', size=12, symbol='triangle-up')))
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', name='Sell Signal',
                             marker=dict(color='red', size=12, symbol='triangle-down')))

    fig.update_layout(title=f'Trading Strategy for {ticker}', xaxis_title='Date', yaxis_title='Price', template='plotly_white')

    # Prepare the summary text
    average_days_held = sum([t['Days Held'] for t in trades]) / number_of_trades if number_of_trades > 0 else 0

    summary_text = (
        f"Ticker: {ticker}\n"
        f"Initial Investment: {initial_investment:,.2f} SAR\n"
        f"Final Portfolio Value: {final_value:,.2f} SAR\n"
        f"Total Return: {total_return:,.2f} SAR\n"
        f"Percentage Return: {percentage_return:.2f}%\n"
        f"Number of Trades: {number_of_trades}\n"
        f"Average Days Held per Trade: {average_days_held:.2f} days\n"
        f"Average Trade Return: {avg_trade_return_percentage:.2f}%\n"
        f"Sharpe Ratio: {sharpe_ratio:.2f}"
    )

    # Create the trades table
    trades_df = pd.DataFrame(trades)
    trades_table = dbc.Table.from_dataframe(trades_df, striped=True, bordered=True, hover=True)

    return fig, summary_text, trades_table


if __name__ == '__main__':
    app.run_server(debug=True)
