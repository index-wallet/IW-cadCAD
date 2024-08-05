import pickle

import pandas as pd

import numpy as np

import networkx as nx

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from util.utils import get_latest_sim

def load_simulation_data(filepath):
    """Load the sim data from a pickle file"""

    print(f"Loading simulation data from: {filepath}")
    
    with open(filepath, 'rb') as file:
        df = pickle.load(file)
    
    if isinstance(df, pd.Series):
        df = df.to_frame().T
    
    print(f"Loaded DataFrame with shape: {df.shape}")
    
    return df

def calc_color(wallet):
    """ Calculate the color of a node based on the wallet ratios """

    sum_wallet = np.sum(wallet)

    if sum_wallet == 0:
        return 0
    ratios = [currency / sum_wallet for currency in wallet]

    ## Need to come back to this, pretty sure we're doing it different than the original.
    ## May also want to change in general to something else
    return ratios[0]  ## Return the ratio of the first currency as a single value

def create_network_from_grid(row):
    """Create a networkx graph from the grid data in a row of the DataFrame"""
    G = nx.Graph()
    grid = row['grid']
    best_vendors = row['best_vendors']
    inherited_assessments = row['inherited_assessments']
    pricing_assessments = row['pricing_assessments']
    
    for node, agent in grid.nodes.data("agent"):
        G.add_node(node, 
                   best_vendors=best_vendors.get(node, []),
                   inherited_assessment=inherited_assessments.get(node, []),
                   pricing_assessment=pricing_assessments.get(node, []),
                   wallet=agent.wallet,
                   price=agent.price,
                   demand=agent.demand,
                   color=calc_color(agent.wallet))
    
    grid_size = int(np.sqrt(len(grid.nodes)))
    for i in range(grid_size):
        for j in range(grid_size):
            if i < grid_size - 1:
                G.add_edge((i, j), (i+1, j))
            if j < grid_size - 1:
                G.add_edge((i, j), (i, j+1))
    
    return G

def create_time_evolving_network(df):
    """Create a list of networkx graphs from the DataFrame"""
    return [create_network_from_grid(row) for _, row in df.iterrows()]

def safe_log10(x):
    """Return the log10 of x, but return -10 if x is 0"""
    return np.log10(max(abs(x), 1e-10))

def create_network_trace(G, layout='grid', pos=None):
    """Create plotly traces for the networkx graph"""
    if layout == 'grid':
        pos = {node: node for node in G.nodes()}
    elif layout == 'force' and pos is None:
        pos = nx.spring_layout(G, k=1/np.sqrt(len(G.nodes())), iterations=50)

    node_x, node_y = zip(*[pos[node] for node in G.nodes()])

    total_wallet_values = [sum(G.nodes[node]['wallet']) for node in G.nodes()]
    max_wallet_value, min_wallet_value = max(total_wallet_values), min(total_wallet_values)
    node_sizes = [5 + 25 * (value - min_wallet_value) / (max_wallet_value - min_wallet_value) 
                  if max_wallet_value != min_wallet_value else 15 for value in total_wallet_values]

    node_colors = [G.nodes[node]['color'] for node in G.nodes()]

    hover_texts = [
        f"Node: {node}<br>"
        f"Best Vendors: {G.nodes[node]['best_vendors']}<br>"
        f"Wallet: {G.nodes[node]['wallet']}<br>"
        f"Currency Valuation: {G.nodes[node]['inherited_assessment']}<br>"
        f"Price: {G.nodes[node]['price']}<br>"
        f"Demand: {G.nodes[node]['demand']}"
        for node in G.nodes()
    ]

    edge_x, edge_y = [], []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Wallet Ratio")
        ),
        text=hover_texts,
        hoverinfo='text',
        hoverlabel=dict(namelength=-1)
    )

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    return edge_trace, node_trace

def calc_average_valuations(row):
    """Calculate the average valuation of all nodes in a row"""
    asmts = row['pricing_assessments']
    valuation = np.array([0.0, 0.0])
    for _, asmt in asmts.items():
        valuation += asmt
    valuation /= 100
    return valuation

def pre_calculate_force_layouts(networks):
    """Pre-calculate force-directed layouts for all networks"""
    force_layouts = []
    for G in networks:
        pos = nx.spring_layout(G, k=1/np.sqrt(len(G.nodes())), iterations=50)
        force_layouts.append(pos)
    return force_layouts


df = load_simulation_data(get_latest_sim())
networks = create_time_evolving_network(df)
num_timesteps = len(df) - 1

# Pre-calculate force-directed layouts
force_layouts = pre_calculate_force_layouts(networks)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


app.layout = html.Div([
    dcc.Graph(id='network-graph', style={'height': '80vh'}),
    html.Div([
        dcc.Slider(
            id='time-slider',
            min=0,
            max=num_timesteps,
            step=1,
            value=0,
            marks={i: {'label': str(i), 'style': {'transform': 'rotate(-45deg)'}} for i in range(0, num_timesteps + 1, 5)},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
    ], style={'width': '95%', 'margin': '20px auto', 'padding-top': '20px'}),
    html.Div([
        dbc.Button('Play/Pause', id='play-pause-button', n_clicks=0, color="primary", className="mr-2"),
        dbc.RadioItems(
            id='layout-toggle',
            options=[
                {'label': 'Grid Layout', 'value': 'grid'},
                {'label': 'Force-Directed Layout', 'value': 'force'}
            ],
            value='grid',
            inline=True,
            style={'display': 'inline-block', 'margin-left': '20px'}
        ),
    ], style={'textAlign': 'center', 'margin': '20px 0'}),
    dcc.Interval(
        id='interval-component',
        interval=500,  # in milliseconds
        n_intervals=0,
        disabled=True
    ),
])

@app.callback(
    Output('network-graph', 'figure'),
    [Input('time-slider', 'value'),
     Input('layout-toggle', 'value')]
)
def update_graph(time_step, layout):
    """Update the graph based on the time step and layout selection"""
    G = networks[time_step]
    
    fig = make_subplots(rows=2, cols=2, 
                        column_widths=[0.7, 0.3],
                        row_heights=[0.7, 0.3],
                        specs=[[{"type": "scatter", "rowspan": 2}, {"type": "scatter"}],
                               [None, {"type": "scatter"}]],
                        vertical_spacing=0.08,
                        horizontal_spacing=0.08)
    
    if layout == 'force':
        edge_trace, node_trace = create_network_trace(G, layout='force', pos=force_layouts[time_step])
    else:
        edge_trace, node_trace = create_network_trace(G, layout='grid')
    
    fig.add_trace(edge_trace, row=1, col=1)
    fig.add_trace(node_trace, row=1, col=1)
    
    scatter_x = [G.nodes[node]['pricing_assessment'][0] for node in G.nodes()]
    scatter_y = [G.nodes[node]['pricing_assessment'][1] for node in G.nodes()]
    
    scatter_trace = go.Scatter(
        x=scatter_x,
        y=scatter_y,
        mode='markers',
        marker=dict(
            size=8,
            color=node_trace.marker.color,
            colorscale='Viridis',
            showscale=False
        ),
        text=[f"Node: {node}<br>Assessment: {G.nodes[node]['pricing_assessment']}" for node in G.nodes()],
        hoverinfo='text',
        name='Pricing Assessments'
    )
    fig.add_trace(scatter_trace, row=1, col=2)
    
    wallet_data = {}
    for t in range(time_step + 1):
        for node in networks[t].nodes():
            if node not in wallet_data:
                wallet_data[node] = []
            wallet_data[node].append(sum(networks[t].nodes[node]['wallet']))
    
    for node, values in wallet_data.items():
        wallet_trace = go.Scatter(
            x=list(range(time_step + 1)),
            y=values,
            mode='lines',
            name=f'Node {node}',
            line=dict(width=1),
            showlegend=False
        )
        fig.add_trace(wallet_trace, row=2, col=2)
    
    avg_asmt = calc_average_valuations(df.iloc[time_step])
    
    fig.update_layout(
        title=f"Time Step: {time_step} | Average Valuations: Red, Blue = {avg_asmt}",
        showlegend=False,
        hovermode='closest',
        height=800,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor='rgba(227, 234, 255, 0.8)',  # Light blue background for plot areas
        paper_bgcolor='white'  # White background for the overall figure
    )
    
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)
    
    fig.update_xaxes(title_text="Red Currency Assessment", row=1, col=2, 
                     type="log", 
                     exponentformat="power",
                     showexponent="all")
    fig.update_yaxes(title_text="Blue Currency Assessment", row=1, col=2, 
                     type="log", 
                     exponentformat="power",
                     showexponent="all")
    
    fig.update_xaxes(title_text="Time Step", row=2, col=2, 
                     range=[0, num_timesteps],
                     dtick=10)
    
    all_wallet_values = [value for values in wallet_data.values() for value in values]
    min_wallet, max_wallet = min(all_wallet_values), max(all_wallet_values)
    fig.update_yaxes(title_text="Total Wallet Value", row=2, col=2, 
                     type="log",
                     exponentformat="power",
                     showexponent="all",
                     range=[safe_log10(min_wallet*0.9), safe_log10(max_wallet*1.1)])
    
    return fig

@app.callback(
    Output('interval-component', 'disabled'),
    [Input('play-pause-button', 'n_clicks')],
    [State('interval-component', 'disabled')]
)
def toggle_interval(n_clicks, current_state):
    """Toggle the interval component on and off"""
    if n_clicks is None:
        raise PreventUpdate
    return not current_state

@app.callback(
    Output('time-slider', 'value'),
    [Input('interval-component', 'n_intervals')],
    [State('time-slider', 'value'),
     State('time-slider', 'max')]
)
def update_slider(n, current_value, max_value):
    if current_value >= max_value:
        return 0
    return current_value + 1

if __name__ == '__main__':
    app.run_server(debug=True)