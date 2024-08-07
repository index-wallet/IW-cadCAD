## built-in
from itertools import combinations
import pickle
import multiprocessing

## third-party
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
from scipy.spatial import cKDTree

## custom
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

def calculate_node_color(wallet, currency_1, currency_2):
    """Calculate the color of a node based on the ratio of two selected currencies"""
    sum_selected = wallet[currency_1] + wallet[currency_2]

    if sum_selected == 0:
        return 0 
    
    ratio = wallet[currency_1] / sum_selected

    return ratio

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
                   demand=agent.demand)
    
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

def create_network_trace(G, layout='grid', pos=None, currency_1=0, currency_2=1):
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

    node_colors = [calculate_node_color(G.nodes[node]['wallet'], currency_1, currency_2) for node in G.nodes()]

    custom_colorscale = [
        [0, 'rgb(165,0,38)'],    ## Dark red
        [0.25, 'rgb(215,48,39)'],  ## Red
        [0.5, 'rgb(244,109,67)'],  ## Light red
        [0.75, 'rgb(69,117,180)'],  ## Light blue
        [1, 'rgb(49,54,149)']    ## Dark blue
    ]
    
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
            colorscale=custom_colorscale,
            showscale=True,
            colorbar=dict(
                title=f"Ratio of Currency {currency_1} to {currency_2}",
                tickmode='array',
                tickvals=[0, 0.25, 0.5, 0.75, 1],
                ticktext=['0 (All Currency 1)', '0.25', '0.5 (Equal)', '0.75', '1 (All Currency 2)']
            )
        ),
        text=hover_texts,
        hoverinfo='text',
        hoverlabel=dict(
            bgcolor='white',
            font_size=12,
            font_family="Arial",
            font_color='black',
            bordercolor='black',
            namelength=-1
        )
    )

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    return edge_trace, node_trace, node_colors

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

def create_improved_centrality_layout(G, k=0.1, iterations=50):
    """Create an improved centrality-based layout with reduced overlap and centered mass"""
    
    ## We're doing magic math here, because this can take a while
    centrality = nx.degree_centrality(G)
    nodes = list(G.nodes())
    n = len(nodes)
    
    ## Initial layout
    centrality_values = np.array([centrality[node] for node in nodes])
    max_centrality = np.max(centrality_values)
    angles = 2 * np.pi * np.random.random(n)
    r = 1 - (centrality_values / max_centrality)
    layout = np.column_stack((r * np.cos(angles), r * np.sin(angles)))
    
    for _ in range(iterations):
        ## Use KDTree for efficient nearest neighbor computations
        tree = cKDTree(layout)
        distances, indices = tree.query(layout, k=n, distance_upper_bound=np.inf)
        
        ## Compute displacements
        displacements = np.zeros_like(layout)
        for i in range(n):
            delta = layout[i] - layout[indices[i, 1:]]
            distance = np.maximum(0.01, np.linalg.norm(delta, axis=1))
            displacements[i] = np.sum(k * delta / distance[:, np.newaxis]**2, axis=0)
        
        layout += displacements
        
        center = layout.mean(axis=0)
        layout -= center
        
        ## Normalize to keep nodes within a unit circle
        max_distance = np.max(np.linalg.norm(layout, axis=1))
        layout /= max_distance
    
    return dict(zip(nodes, layout))

def pre_calculate_centrality_layouts(networks):
    """Pre-calculate centrality-based layouts for all networks in parallel"""
    with multiprocessing.Pool() as pool:
        centrality_layouts = pool.map(create_improved_centrality_layout, networks)
    return centrality_layouts

def get_currency_pairs(df):
    """Get all possible currency pairs from the DataFrame"""
    sample_wallet = df.iloc[0]['grid'].nodes[list(df.iloc[0]['grid'].nodes)[0]]['agent'].wallet
    currencies = range(len(sample_wallet))
    return list(combinations(currencies, 2))

def load_data_and_prepare_layouts():
    """Load the simulation data and pre-calculate layouts"""
    df = load_simulation_data(get_latest_sim())
    networks = create_time_evolving_network(df)
    num_timesteps = len(df) - 1
    currency_pairs = get_currency_pairs(df)

    print("Data loaded and networks created")


    print("Pre-calculating layouts...")

    print("Pre-calculating force-directed layouts...")
    force_layouts = pre_calculate_force_layouts(networks)

    print("Pre-calculating centrality-based layouts...")
    centrality_layouts = pre_calculate_centrality_layouts(networks)

    print("Layouts pre-calculated")

    return df, networks, num_timesteps, currency_pairs, force_layouts, centrality_layouts

def create_dash_app(df, networks, num_timesteps, currency_pairs, force_layouts, centrality_layouts):
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    app.layout = html.Div([
        html.Div([
            html.Div([
                dbc.Button('Play/Pause', id='play-pause-button', n_clicks=0, color="primary", className="mb-2 w-100"),
                dcc.Dropdown(
                    id='layout-toggle',
                    options=[
                        {'label': 'Grid Layout', 'value': 'grid'},
                        {'label': 'Force-Directed Layout', 'value': 'force'},
                        {'label': 'Centrality-Based Layout', 'value': 'centrality'}
                    ],
                    value='grid',
                    clearable=False,
                    searchable=False,
                    style={'width': '100%'}
                ),
                dcc.Dropdown(
                    id='currency-pair-dropdown',
                    options=[{'label': f'Currency {pair[0]} vs Currency {pair[1]}', 'value': f'{pair[0]},{pair[1]}'} for pair in currency_pairs],
                    value=f'{currency_pairs[0][0]},{currency_pairs[0][1]}',
                    clearable=False,
                    searchable=False,
                    style={
                        'width': '100%', 
                        'marginTop': '10px',
                        'fontSize': '12px'
                    }
                ),
            ], style={'width': '200px', 'position': 'absolute', 'left': '10px', 'top': '50px', 'zIndex': '1000'}),
            dcc.Graph(id='network-graph', style={'height': '80vh', 'width': 'calc(100% - 220px)', 'marginLeft': '220px'}),
        ], style={'position': 'relative', 'height': '80vh'}),
        
        html.Div([
            dcc.Slider(
                id='time-slider',
                min=0,
                max=num_timesteps,
                step=1,
                value=0,
                marks={i: {'label': str(i)} for i in range(0, num_timesteps + 1, 5)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], style={'width': '95%', 'margin': '20px auto', 'paddingTop': '20px'}),
        
        dcc.Interval(
            id='interval-component',
            interval=1000,
            n_intervals=0,
            disabled=True
        ),
    ])
    
    app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .rc-slider-track {
                background-color: #007bff;
            }
            .rc-slider-handle {
                border-color: #007bff;
                background-color: #007bff;
            }
            .rc-slider-handle:hover {
                border-color: #0056b3;
            }
            .rc-slider-handle-active:active {
                border-color: #0056b3;
            }
            .rc-slider-mark-text {
                color: #495057;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''
    @app.callback(
    Output('network-graph', 'figure'),
    [Input('time-slider', 'value'),
    Input('layout-toggle', 'value'),
    Input('currency-pair-dropdown', 'value')]
    )
    def update_graph(time_step, layout, currency_pair):
        """Update the graph based on the time step, layout selection, and currency pair"""
        
        G = networks[time_step]
        
        currency_1, currency_2 = map(int, currency_pair.split(','))
        
        fig = make_subplots(rows=2, cols=2, 
                            column_widths=[0.7, 0.3],
                            row_heights=[0.7, 0.3],
                            specs=[[{"type": "scatter", "rowspan": 2}, {"type": "scatter"}],
                                [None, {"type": "scatter"}]],
                            vertical_spacing=0.08,
                            horizontal_spacing=0.08,
                            subplot_titles=["", f"Currency Assessments (Step {time_step})", ""])
        
        if layout == 'force':
            edge_trace, node_trace, node_colors = create_network_trace(G, layout='force', pos=force_layouts[time_step], currency_1=currency_1, currency_2=currency_2)
        elif layout == 'centrality':
            edge_trace, node_trace, node_colors = create_network_trace(G, layout='force', pos=centrality_layouts[time_step], currency_1=currency_1, currency_2=currency_2)
        else:
            edge_trace, node_trace, node_colors = create_network_trace(G, layout='grid', currency_1=currency_1, currency_2=currency_2)
        
        fig.add_trace(edge_trace, row=1, col=1)
        fig.add_trace(node_trace, row=1, col=1)
        
        scatter_x = [G.nodes[node]['pricing_assessment'][currency_1] for node in G.nodes()]
        scatter_y = [G.nodes[node]['pricing_assessment'][currency_2] for node in G.nodes()]
        
        scatter_trace = go.Scatter(
            x=scatter_x,
            y=scatter_y,
            mode='markers',
            marker=dict(
                size=8,
                color=node_colors,
                colorscale=node_trace.marker.colorscale,
                showscale=False
            ),
            text=[f"Node: {node}<br>Assessment: {G.nodes[node]['pricing_assessment']}" for node in G.nodes()],
            hoverinfo='text',
            hoverlabel=dict(
                bgcolor='white',
                font_size=12,
                font_family="Arial",
                font_color='black',
                bordercolor='black',
            ),
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
                hoverinfo='text',
                hoverlabel=dict(
                    bgcolor='white',
                    font_size=12,
                    font_family="Arial",
                    font_color='black',
                    bordercolor='black',
                ),
                text=[f"Node: {node}<br>Time Step: {t}<br>Total Wallet: {v:.4f}" for t, v in enumerate(values)],
                showlegend=False
            )
            fig.add_trace(wallet_trace, row=2, col=2)
        
        avg_asmt = calc_average_valuations(df.iloc[time_step])
        
        currency_valuations = [f'Currency {i}: {v:.2f}' for i, v in enumerate(avg_asmt)]
        title = f"Time Step: {time_step} | Average Valuations: {', '.join(currency_valuations)}"
        
        fig.update_layout(
            title=title,
            showlegend=False,
            hovermode='closest',
            height=800,
            margin=dict(l=20, r=20, t=50, b=20),
            plot_bgcolor='rgba(227, 234, 255, 0.8)',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)
        
        fig.update_xaxes(title_text=f"Currency {currency_1} Assessment", row=1, col=2, 
                        type="log", 
                        exponentformat="power",
                        showexponent="all")
        fig.update_yaxes(title_text=f"Currency {currency_2} Assessment", row=1, col=2, 
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
         State('time-slider', 'max'),
         State('interval-component', 'disabled')]
    )
    def update_slider(n, current_value, max_value, is_disabled):
        if is_disabled:
            raise PreventUpdate
        if current_value >= max_value:
            return 0
        return current_value + 1

    return app

if __name__ == '__main__':
    multiprocessing.freeze_support()

    print("Loading data and preparing layouts...")
    df, networks, num_timesteps, currency_pairs, force_layouts, centrality_layouts = load_data_and_prepare_layouts()

    print("Creating Dash app...")
    app = create_dash_app(df, networks, num_timesteps, currency_pairs, force_layouts, centrality_layouts)

    print("Running the app...")
    app.run(debug=True, port=8050)