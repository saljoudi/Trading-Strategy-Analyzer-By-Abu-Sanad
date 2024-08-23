#!/usr/bin/env python
# coding: utf-8

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import yfinance as yf
import ta
import plotly.graph_objs as go
import warnings
warnings.filterwarnings("ignore")


# Initialize the Dash app with a Bootstrap theme for a professional look
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Trading Strategy Analyzer", className="text-center"), className="mb-4 mt-4")
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Input Parameters", className="card-title"),
                    dbc.Label("Ticker Symbol (without .SR):"),
                    dcc.Input(id='ticker-input', type='text', value='1303', className="mb-3", style={'width': '100%'}),
                    dbc.Label("Period:"),
                    dcc.Input(id='period-input', type='text', value='1y', className="mb-3", style={'width': '100%'}),
                    dbc.Label("Short SMA Period:"),
                    dcc.Input(id='sma-short-input', type='number', value=17, className="mb-3", style={'width': '100%'}),
                    dbc.Label("Long SMA Period:"),
                    dcc.Input(id='sma-long-input', type='number', value=10, className="mb-3", style={'width': '100%'}),
                    dbc.Label("RSI Threshold:"),
                    dcc.Input(id='rsi-threshold-input', type='number', value=45, className="mb-3", style={'width': '100%'}),
                    dbc.Label("Short ADL SMA Period:"),
                    dcc.Input(id='adl-short-input', type='number', value=7, className="mb-3", style={'width': '100%'}),
                    dbc.Label("Long ADL SMA Period:"),
                    dcc.Input(id='adl-long-input', type='number', value=15, className="mb-3", style={'width': '100%'}),
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
                    html.H4("Strategy Summary", className="card-title"),
                    html.Pre(id='summary-output', style={'whiteSpace': 'pre-wrap', 'font-family': 'monospace'})
                ])
            ])
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Trades Details", className="card-title"),
                    dbc.Table(id='trades-table', bordered=True, hover=True, responsive=True, striped=True)
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
    ticker = f"{ticker_input}.SR"

    # Download the data for the ticker
    df = yf.download(ticker, period=period)

    # Calculate the indicators based on the parameters provided
    df['SMA_Short'] = df['Close'].rolling(window=sma_short).mean()
    df['SMA_Long'] = df['Close'].rolling(window=sma_long).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['MACD_Signal'] = ta.trend.MACD(df['Close']).macd_signal()
    df['ADL'] = ta.volume.AccDistIndexIndicator(df['High'], df['Low'], df['Close'], df['Volume']).acc_dist_index()
    df['ADL_Short_SMA'] = df['ADL'].rolling(window=adl_short).mean()
    df['ADL_Long_SMA'] = df['ADL'].rolling(window=adl_long).mean()

    # Signal generation based on the current parameters
    df['Signal'] = 0  # Default to no signal
    df['Signal'] = df.apply(
        lambda row: 1 if row['SMA_Short'] > row['SMA_Long'] and row['ADL_Short_SMA'] > row['ADL_Long_SMA'] and row['RSI'] >= rsi_threshold and row['MACD'] > row['MACD_Signal'] else (
            -1 if row['SMA_Short'] < row['SMA_Long'] and row['ADL_Short_SMA'] < row['ADL_Long_SMA'] and row['RSI'] < rsi_threshold and row['MACD'] <= row['MACD_Signal'] else 0
        ), axis=1
    )

    # Simulate trading with the generated signals
    initial_investment = 100000  # Example initial investment
    portfolio = initial_investment
    trades = []
    buy_price = None
    trade_start = None
    number_of_trades = 0

    for index, row in df.iterrows():
        if row['Signal'] == 1 and buy_price is None:
            buy_price = row['Close']
            trade_start = index
            number_of_trades += 1
        elif row['Signal'] == -1 and buy_price is not None:
            sell_price = row['Close']
            profit = (sell_price - buy_price) * (portfolio / buy_price)
            portfolio += profit
            days_held = (index - trade_start).days

            trades.append({
                'Sell Date': index.strftime('%Y-%m-%d'),
                'Buy Price': f"{buy_price:,.2f} SAR",
                'Sell Price': f"{sell_price:,.2f} SAR",
                'Days Held': days_held,
                'Profit': f"{profit:,.2f} SAR",
                'Profit Percentage': f"{(profit / (portfolio - profit)) * 100:.2f}%"
            })

            buy_price = None  # Reset after trade

    final_value = portfolio
    total_return = final_value - initial_investment
    percentage_return = (total_return / initial_investment) * 100

    # Create the plot
    fig = go.Figure()

    # Add the Closing Price and SMA lines
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Closing Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_Short'], mode='lines', name=f'SMA Short ({sma_short})'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_Long'], mode='lines', name=f'SMA Long ({sma_long})'))

    # Add the Buy and Sell signals
    buy_signals = df[df['Signal'] == 1]
    sell_signals = df[df['Signal'] == -1]

    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', name='Buy Signal', marker=dict(color='green', symbol='triangle-up')))
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', name='Sell Signal', marker=dict(color='red', symbol='triangle-down')))

    fig.update_layout(title=f'Trading Strategy for {ticker}', xaxis_title='Date', yaxis_title='Price', template='plotly_white')

    # Prepare the summary text
    summary_text = (
        f"Ticker: {ticker}\n"
        f"Initial Investment: {initial_investment:,.2f} SAR\n"
        f"Final Portfolio Value: {final_value:,.2f} SAR\n"
        f"Total Return: {total_return:,.2f} SAR\n"
        f"Percentage Return: {percentage_return:.2f}%\n"
        f"Number of Trades: {number_of_trades}\n"
        f"Average Days Held per Trade: {sum([t['Days Held'] for t in trades]) / number_of_trades if number_of_trades > 0 else 0:.2f} days"
    )

    # Prepare the trades table
    trades_table_header = [html.Thead(html.Tr([html.Th(col) for col in trades[0].keys()]))]
    trades_table_body = [html.Tbody([html.Tr([html.Td(trade[col]) for col in trade.keys()]) for trade in trades])]

    return fig, summary_text, trades_table_header + trades_table_body


if __name__ == '__main__':
    app.run_server(debug=True)
