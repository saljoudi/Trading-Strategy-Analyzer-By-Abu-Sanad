import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import yfinance as yf
import ta
import plotly.graph_objs as go
import numpy as np  # Added for numerical computations

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
                            {'label': '6 Months', 'value': '6mo'},
                            {'label': '1 Year', 'value': '1y'},
                            {'label': '18 Months', 'value': '18mo'},
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
    Input('analyze-button', 'n_clicks'),
    State('ticker-input', 'value'),
    State('period-input', 'value'),
    State('sma-short-input', 'value'),
    State('sma-long-input', 'value'),
    State('rsi-threshold-input', 'value'),
    State('adl-short-input', 'value'),
    State('adl-long-input', 'value')
)
def update_graph(n_clicks, ticker_input, period, sma_short, sma_long, rsi_threshold, adl_short, adl_long):
    if n_clicks is None:
        # Prevent update if the function is called without a button click
        return dash.no_update, dash.no_update, dash.no_update

    # Ensure numpy is imported for calculations
    import numpy as np

    # Check if the ticker is numeric (Saudi stock symbol)
    if ticker_input.isdigit():
        ticker = f"{ticker_input}.SR"
    else:
        ticker = ticker_input.upper()

    # Download the data for the ticker
    try:
        df = yf.download(ticker, period=period)
        if df.empty:
            return {}, "No data found for the ticker symbol.", html.Div()
    except Exception as e:
        return {}, f"Error downloading data: {e}", html.Div()

    df.index = pd.to_datetime(df.index)

    # Calculate indicators
    df['SMA_Short'] = df['Close'].rolling(window=sma_short).mean()
    df['SMA_Long'] = df['Close'].rolling(window=sma_long).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    adl = ta.volume.AccDistIndexIndicator(df['High'], df['Low'], df['Close'], df['Volume'])
    df['ADL'] = adl.acc_dist_index()
    df['ADL_Short_SMA'] = df['ADL'].rolling(window=adl_short).mean()
    df['ADL_Long_SMA'] = df['ADL'].rolling(window=adl_long).mean()

    # Signal generation
    df['Signal'] = df.apply(
        lambda row: -1 if row['Close'] >= row['SMA_Short'] and row['SMA_Short'] > row['SMA_Long'] and row['ADL_Short_SMA'] > row['ADL_Long_SMA'] and row['RSI'] >= rsi_threshold and row['MACD'] > row['MACD_Signal'] else (
            1 if row['Close'] < row['SMA_Short'] and row['SMA_Short'] < row['SMA_Long'] else 0
        ), axis=1
    )

    # Simulate trading
    initial_investment = 100000
    portfolio = initial_investment
    trades = []
    portfolio_values = []
    position = 0  # 0 means no position, 1 means holding stock
    number_of_trades = 0
    number_of_shares = 0
    buy_price = None

    for index, row in df.iterrows():
        if row['Signal'] == 1 and position == 0:
            # Buy signal and not holding position
            buy_price = row['Close']
            number_of_shares = portfolio / buy_price
            trade_start = index
            number_of_trades += 1
            position = 1  # Now we have a position
        elif row['Signal'] == -1 and position == 1:
            # Sell signal and holding position
            sell_price = row['Close']
            sell_value = number_of_shares * sell_price
            profit = sell_value - (number_of_shares * buy_price)
            portfolio = sell_value
            days_held = (index - trade_start).days
            profit_percentage = (sell_price - buy_price) / buy_price

            trades.append({
                'Sell Date': index.date().strftime('%Y-%m-%d'),
                'Buy Price': buy_price,
                'Sell Price': sell_price,
                'Days Held': days_held,
                'Profit': profit,
                'Profit Percentage': profit_percentage
            })

            buy_price = None
            number_of_shares = 0
            position = 0  # No position after selling
        else:
            # Update portfolio value
            if position == 1:
                # Holding position
                portfolio = number_of_shares * row['Close']

        # Record portfolio value
        portfolio_values.append(portfolio)

    # If still holding a position at the end, sell at the last price
    if position == 1:
        sell_price = df.iloc[-1]['Close']
        sell_value = number_of_shares * sell_price
        profit = sell_value - (number_of_shares * buy_price)
        portfolio = sell_value
        days_held = (df.index[-1] - trade_start).days
        profit_percentage = (sell_price - buy_price) / buy_price

        trades.append({
            'Sell Date': df.index[-1].date().strftime('%Y-%m-%d'),
            'Buy Price': buy_price,
            'Sell Price': sell_price,
            'Days Held': days_held,
            'Profit': profit,
            'Profit Percentage': profit_percentage
        })

        buy_price = None
        number_of_shares = 0
        position = 0  # No position after selling

        # Update last portfolio value
        portfolio_values[-1] = portfolio

    final_value = portfolio
    total_return = final_value - initial_investment
    percentage_return = (total_return / initial_investment) * 100

    # Create a DataFrame for portfolio values to calculate Sharpe ratio
    portfolio_df = pd.DataFrame({
        'Date': df.index,
        'Portfolio Value': portfolio_values
    })
    portfolio_df.set_index('Date', inplace=True)
    portfolio_df['Daily Return'] = portfolio_df['Portfolio Value'].pct_change()

    # Calculate Sharpe Ratio
    mean_daily_return = portfolio_df['Daily Return'].mean()
    std_daily_return = portfolio_df['Daily Return'].std()
    sharpe_ratio = (mean_daily_return / std_daily_return) * np.sqrt(252) if std_daily_return != 0 else 0

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
    trades_df = pd.DataFrame(trades)

    if not trades_df.empty:
        average_profit_percentage = trades_df['Profit Percentage'].mean() * 100  # Converted to percentage
        average_days_held = trades_df['Days Held'].mean()
        average_profit = trades_df['Profit'].mean()
        std_profit_percentage = trades_df['Profit Percentage'].std()
        
        summary_text = (
            f"Ticker: {ticker}\n"
            f"Initial Investment: {initial_investment:,.2f} SAR\n"
            f"Final Portfolio Value: {final_value:,.2f} SAR\n"
            f"Total Return: {total_return:,.2f} SAR\n"
            f"Percentage Return: {percentage_return:.2f}%\n"
            f"Number of Trades: {number_of_trades}\n"
            f"Average Days Held per Trade: {average_days_held:.2f} days\n"
            f"Average Profit per Trade: {average_profit:,.2f} SAR\n"
            f"Average Profit Percentage per Trade: {average_profit_percentage:.2f}%\n"
            f"Sharpe Ratio: {sharpe_ratio:.2f}"
        )

        # Create the trades table
        trades_df_display = trades_df.copy()
        trades_df_display['Buy Price'] = trades_df_display['Buy Price'].apply(lambda x: f"{x:.2f} SAR")
        trades_df_display['Sell Price'] = trades_df_display['Sell Price'].apply(lambda x: f"{x:.2f} SAR")
        trades_df_display['Profit'] = trades_df_display['Profit'].apply(lambda x: f"{x:,.2f} SAR")
        trades_df_display['Profit Percentage'] = trades_df_display['Profit Percentage'].apply(lambda x: f"{x * 100:.2f}%")
        trades_table = dbc.Table.from_dataframe(trades_df_display, striped=True, bordered=True, hover=True)
    else:
        summary_text = (
            f"Ticker: {ticker}\n"
            f"No trades were made during this period."
        )
        trades_table = html.Div("No trades were made during this period.")

    return fig, summary_text, trades_table

if __name__ == '__main__':
    app.run_server(debug=True)
