## Some notes

## Agent and Node are basically interchangeable here. Node is the internal word and Agent is what it appears in the UI.
## The same goes for assessment and valuation. Assessment is the internal word and valuation is what it appears in the UI.

## built-in
from itertools import combinations
from functools import partial
from typing import List, Tuple, Dict, Any
import pickle
import re
import multiprocessing
import random
import logging
import sys
import os

## third-party
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor

## global
previous_node = None

## adds the parent directory to the path so util can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

## custom
from util.utils import get_latest_sim

def load_simulation_data(filepath:str) -> pd.DataFrame:
    """Load the sim data from a pickle file"""

    logging.info(f"Loading simulation data from: {filepath}")
    
    with open(filepath, 'rb') as file:
        df = pickle.load(file)
    
    if isinstance(df, pd.Series):
        df = df.to_frame().T
    
    logging.info(f"Loaded DataFrame with shape: {df.shape}")

    return df

def calculate_node_color(wallet:list, currency_1:int, currency_2:int) -> float:
    """Calculate the color of a node based on the ratio of two selected currencies"""
    sum_selected = wallet[currency_1] + wallet[currency_2]

    if sum_selected == 0:
        return 0 
    
    ratio = wallet[currency_1] / sum_selected

    return ratio

def calculate_wallet_value(wallet:list, valuation:list) -> float:
    """Calculate the total value of a wallet based on the valuation of each currency"""
    return sum(amount * value for amount, value in zip(wallet, valuation))

def create_network_from_grid(row:pd.Series) -> nx.DiGraph:
    """Create a networkx graph from the grid data in a row of the DataFrame"""
    graph = nx.DiGraph()
    grid = row['grid']
    best_vendors = row['best_vendors']
    inherited_assessments = row['inherited_assessments']
    pricing_assessments = row['pricing_assessments']
    edges = row['edges']
    
    for node, agent in grid.nodes.data("agent"):
        graph.add_node(node, 
                   best_vendors=best_vendors.get(node, []),
                   inherited_assessment=inherited_assessments.get(node, []),
                   pricing_assessment=pricing_assessments.get(node, []),
                   wallet=agent.wallet,
                   price=agent.price,
                   demand=agent.demand,
                   type=agent.type)
    
    graph.add_edges_from(edges)
    
    return graph


def create_time_evolving_network(df:pd.DataFrame) -> list:
    """Create a list of networkx graphs from the DataFrame"""
    return [create_network_from_grid(row) for _, row in df.iterrows()]

def safe_log10(x:float) -> float:
    """Return the log10 of x, but return -10 if x is 0"""
    return np.log10(max(abs(x), 1e-10))

def create_network_trace(graph:nx.DiGraph, next_graph:nx.DiGraph | None, layout:str='grid', pos=None, currency_1:int=0, currency_2:int=1, selected_node=None, prev_best_vendors=None, time_step=0):
    """Create plotly traces for the networkx graph"""

    if layout == 'grid':
        pos = {node: node for node in graph.nodes()}
    elif layout == 'force' and pos is None:
        pos = nx.spring_layout(graph, k=1/np.sqrt(len(graph.nodes())), iterations=50)

    assert pos is not None, "Positional layout is required for force-directed and centrality-based layouts"

    node_x, node_y = zip(*[pos[node] for node in graph.nodes()])

    total_wallet_values = [calculate_wallet_value(graph.nodes[node]['wallet'], graph.nodes[node]['inherited_assessment']) for node in graph.nodes()]

    ## Use logarithmic scaling for node sizes otherwise it'll be pretty extreme
    log_values = np.log1p(total_wallet_values) 
    min_log_value, max_log_value = min(log_values), max(log_values)
    
    min_size, max_size = 10, 30 
    node_sizes = [min_size + (max_size - min_size) * (log_value - min_log_value) / (max_log_value - min_log_value)
                  for log_value in log_values]

    node_colors = [calculate_node_color(graph.nodes[node]['wallet'], currency_1, currency_2) for node in graph.nodes()]

    custom_colorscale = [
        [0, 'rgb(165,0,38)'],    ## Dark red
        [0.25, 'rgb(215,48,39)'],  ## Red
        [0.5, 'rgb(244,109,67)'],  ## Light red
        [0.75, 'rgb(69,117,180)'],  ## Light blue
        [1, 'rgb(49,54,149)']    ## Dark blue
    ]

    def interpolate_color(val):
        """Interpolate color based on value and custom colorscale"""
        for i in range(len(custom_colorscale) - 1):
            if custom_colorscale[i][0] <= val <= custom_colorscale[i+1][0]:
                t = (val - custom_colorscale[i][0]) / (custom_colorscale[i+1][0] - custom_colorscale[i][0])
                r1, g1, b1 = map(int, custom_colorscale[i][1][4:-1].split(','))
                r2, g2, b2 = map(int, custom_colorscale[i+1][1][4:-1].split(','))
                r = int(r1 * (1-t) + r2 * t)
                g = int(g1 * (1-t) + g2 * t)
                b = int(b1 * (1-t) + b2 * t)
                return f'rgb({r},{g},{b})'
        return custom_colorscale[-1][1]  ## Return last color if val > 1
    
    def format_float(value):
        """Format a float to 4 decimal places"""
        return f"{float(value):.4f}"

    def format_demand(demand):
        """Format demand string with line breaks after each full node-value pair"""
        formatted = str(demand)
        result = []
        paren_count = 0
        number_pattern = re.compile(r'\d+\.\d+')
        
        for char in formatted:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            
            result.append(char)
            
            if paren_count == 0 and char == ',':
                result.append('<br>')
        
        joined_result = ''.join(result).rstrip(',<br>') + ')'
        
        return number_pattern.sub(lambda m: format_float(float(m.group())), joined_result)

    hover_texts = [
        f"Agent: {node}<br>"
        f"Best Vendors: {graph.nodes[node]['best_vendors']}<br>"
        f"Wallet: {graph.nodes[node]['wallet']}<br>"
        f"Wallet Value: {calculate_wallet_value(graph.nodes[node]['wallet'], graph.nodes[node]['inherited_assessment']):.4e}<br>"
        f"Currency Valuation: {graph.nodes[node]['inherited_assessment']}<br>"
        f"Price: {format_float(graph.nodes[node]['price'])}<br>"
        f"Demand:<br>{format_demand(graph.nodes[node]['demand'])}<br>"
        f"Prosumer Group (vendor|customer|both): {graph.nodes[node]['type']}"
        for node in graph.nodes()
    ]

    edge_x, edge_y = [], []
    for edge in graph.edges():
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
                title=f"Ratio of Currency {currency_1+1} to {currency_2+1}",
                tickmode='array',
                tickvals=[0, 0.25, 0.5, 0.75, 1],
                ticktext=[
                    f'0 (All Currency {currency_1+1})', 
                    '0.25', 
                    '0.5 (Equal)', 
                    '0.75', 
                    f'1 (All Currency {currency_2+1})'
                ]
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
        mode='lines+markers',
        marker=dict(
            size=3,
            color='#888',
            symbol='triangle-right'
        )
    )

    selected_circle = None
    if selected_node is not None:
        selected_x, selected_y = pos[selected_node]
        selected_size = node_sizes[list(graph.nodes()).index(selected_node)]
        selected_circle = go.Scatter(
            x=[selected_x],
            y=[selected_y],
            mode='markers',
            marker=dict(
                size=selected_size * 1.5,
                color='rgba(0,0,0,0)', 
                line=dict(color='red', width=2)
            ),
            hoverinfo='none',
            showlegend=False
        )

    shapes = []
    annotations = []

    ## only odd time steps and last step has best vendors (so excluding the first 2 steps)
    display_transactions = time_step % 2 == 1 and prev_best_vendors is not None and any(prev_best_vendors.values())

    ## next graph will only not be a thing at the last time step which should be even so 'technically' shouldn't trigger anyways
    if display_transactions and next_graph:
        
        transaction_sizes = {}
        
        ## using wallet valuation differences (so next time step - current time step) to calculate transaction sizes
        for node in graph.nodes():
            current_wallet = np.array(graph.nodes[node]['wallet'])
            current_valuation = np.array(graph.nodes[node]['inherited_assessment'])
            next_wallet = np.array(next_graph.nodes[node]['wallet'])
            next_valuation = np.array(next_graph.nodes[node]['inherited_assessment'])
            
            current_wealth = current_wallet * current_valuation
            next_wealth = next_wallet * next_valuation
            
            wealth_diff = current_wealth - next_wealth
            
            for vendor in prev_best_vendors.get(node, []): ## type: ignore
                vendor_current_wallet = np.array(graph.nodes[vendor]['wallet'])
                vendor_current_valuation = np.array(graph.nodes[vendor]['inherited_assessment'])
                vendor_next_wallet = np.array(next_graph.nodes[vendor]['wallet'])
                vendor_next_valuation = np.array(next_graph.nodes[vendor]['inherited_assessment'])
                
                vendor_current_wealth = vendor_current_wallet * vendor_current_valuation
                vendor_next_wealth = vendor_next_wallet * vendor_next_valuation
                
                vendor_wealth_diff = vendor_next_wealth - vendor_current_wealth
                
                transaction_size = np.sum(np.minimum(wealth_diff, vendor_wealth_diff))
                transaction_sizes[(node, vendor)] = transaction_size

        max_transaction = max(abs(size) for size in transaction_sizes.values()) if transaction_sizes else 1
        min_width, max_width = 1, 10

        for (node, vendor), transaction_size in transaction_sizes.items():
            x0, y0 = pos[node]
            x1, y1 = pos[vendor]
            
            ## Calculate midpoint and control point for the curve
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            offset = 0.2  
            perp_x = (y0 - y1) * offset
            perp_y = (x1 - x0) * offset
            ctrl_x = mid_x + perp_x
            ctrl_y = mid_y + perp_y
            
            ## Get the color of the originating node
            node_index = list(graph.nodes()).index(node)
            node_color_value = node_colors[node_index]
            node_color = interpolate_color(node_color_value)
            
            ## Calculate arrow width based on transaction size
            normalized_size = abs(transaction_size) / max_transaction
            arrow_width = min_width + (max_width - min_width) * normalized_size
            
            ## Determine arrow direction
            if transaction_size > 0:
                start_x, start_y = x0, y0
                end_x, end_y = x1, y1
            else:
                start_x, start_y = x1, y1
                end_x, end_y = x0, y0
            
            ## Create a curved path using a quadratic Bezier curve
            shapes.append(dict(
                type="path",
                path=f"M {start_x},{start_y} Q {ctrl_x},{ctrl_y} {end_x},{end_y}",
                line=dict(width=arrow_width, color=node_color)
            ))
            
            ## Add a Unicode triangle at the end of the line
            annotations.append(dict(
                x=end_x,
                y=end_y,
                xref="x",
                yref="y",
                text="▶",
                showarrow=False,
                font=dict(size=10 + arrow_width, color=node_color),
                textangle=np.degrees(np.arctan2(end_y-start_y, end_x-start_x)), 
                ax=5,
                ay=5
            ))

    return edge_trace, node_trace, node_colors, selected_circle, shapes, annotations

def calc_average_valuations(row:pd.Series, currency_1:int, currency_2:int) -> np.ndarray:
    """Calculate the average valuation of all nodes in a row for two specified currencies"""
    asmts = row['pricing_assessments']
    valuation = np.array([0.0, 0.0])
    for _, asmt in asmts.items():
        valuation += np.array([asmt[currency_1], asmt[currency_2]])
    valuation /= len(asmts)
    return valuation

def pre_calculate_force_layouts(networks:List[nx.DiGraph]):
    """Pre-calculate force-directed layouts for all networks using a fixed seed"""
    random.seed(42)
    np.random.seed(42) 

    force_layouts = []
    for graph in networks:
        pos = nx.spring_layout(graph, k=1/np.sqrt(len(graph.nodes())), iterations=50, seed=42)
        force_layouts.append(pos)
    return force_layouts

def create_centrality_layout(graph:nx.DiGraph, k:float=0.1, iterations:int=50, central_force:float=0.2):
    """Create a centrality-based layout for a networkx graph"""
    random.seed(42)
    np.random.seed(42)
    
    ## Eigenvector centrality with fallback to degree centrality
    try:
        centrality = nx.eigenvector_centrality(graph)
    except:
        centrality = nx.degree_centrality(graph)
    
    nodes = list(graph.nodes())
    n = len(nodes)
    
    ## Initial layout
    centrality_values = np.array([centrality[node] for node in nodes])
    max_centrality = np.max(centrality_values)
    min_centrality = np.min(centrality_values)
    
    ## Normalization
    if max_centrality > min_centrality:
        normalized_centrality = (centrality_values - min_centrality) / (max_centrality - min_centrality)
    else:
        normalized_centrality = np.ones(n) * 0.5  ## If all centralities are equal
    
    ## Initial positioning
    angles = 2 * np.pi * np.random.random(n)
    r = 1 - normalized_centrality
    r = np.power(r, 0.5)
    layout = np.column_stack((r * np.cos(angles), r * np.sin(angles)))
    
    for _ in range(iterations):
        tree = cKDTree(layout)
        distances, indices = tree.query(layout, k=min(n, 10), distance_upper_bound=np.inf)
        
        displacements = np.zeros_like(layout)
        for i in range(n):
            delta = layout[i] - layout[indices[i, 1:]]
            distance = np.maximum(0.01, np.linalg.norm(delta, axis=1))
            repulsion_factor = k * (1 + 5 * normalized_centrality[i]) 
            displacements[i] = np.sum(repulsion_factor * delta / distance[:, np.newaxis]**2, axis=0)
        
        central_attraction = central_force * normalized_centrality[:, np.newaxis] * - layout # type: ignore
        
        layout += displacements + central_attraction
        
        ## Keep nodes within the unit circle
        distances_from_center = np.maximum(0.01, np.linalg.norm(layout, axis=1))
        layout /= distances_from_center[:, np.newaxis]
    
    ## Final adjustments to prevent overlap
    final_repulsion_iterations = 10
    for _ in range(final_repulsion_iterations):
        tree = cKDTree(layout)
        distances, indices = tree.query(layout, k=2)  ## Only consider the nearest neighbor
        
        displacements = np.zeros_like(layout)
        for i in range(n):
            if distances[i, 1] < 0.05: 
                delta = layout[i] - layout[indices[i, 1]]
                norm = np.linalg.norm(delta)
                if norm > 0:
                    displacements[i] = 0.1 * delta / norm
                else:
                    displacements[i] = 0.1 * np.random.random(2) - 0.05
        
        layout += displacements
    
    layout = np.nan_to_num(layout, nan=0.0, posinf=1.0, neginf=-1.0)
    
    layout *= 1.8
    
    return dict(zip(nodes, layout))

def pre_calculate_centrality_layouts(networks:List[nx.DiGraph]):
    """Pre-calculate centrality-based layouts for all networks in parallel"""
    with multiprocessing.Pool() as pool:
        centrality_layouts = pool.map(create_centrality_layout, networks)
    return centrality_layouts

def pre_calculate_node_metrics(networks:List[nx.DiGraph]):
    """Pre-calculate node metrics for all networks"""
    node_metrics = {}
    for t, graph in enumerate(networks):
        for node in graph.nodes():
            if node not in node_metrics:
                node_metrics[node] = {
                    'wallet': [],
                    'wallet_value': [],
                    'price': [],
                    'inherited_assessment': [],
                    'pricing_assessment': [],
                    'demand': []
                }
            
            node_data = graph.nodes[node]
            node_metrics[node]['wallet'].append(node_data['wallet'])
            node_metrics[node]['wallet_value'].append(calculate_wallet_value(node_data['wallet'], node_data['inherited_assessment']))
            node_metrics[node]['price'].append(node_data['price'])
            node_metrics[node]['inherited_assessment'].append(node_data['inherited_assessment'])
            node_metrics[node]['pricing_assessment'].append(node_data['pricing_assessment'])
            node_metrics[node]['demand'].append(node_data['demand'])
    
    return node_metrics

def fast_process_time_step(time_step, networks, node_metrics, currency_pairs):
    """Process a single time step and return traces for the wallet, currency scatter, and node metrics"""
    graph = networks[time_step]
    nodes = list(graph.nodes())
    time_range = np.arange(time_step + 1)
    
    ## Wallet value traces
    wallet_values = np.array([[node_metrics[node]['wallet_value'][t] for t in range(time_step + 1)] for node in nodes])
    wallet_traces = [
        dict(
            x=time_range,
            y=values,
            mode='lines',
            name=f'Agent {node}',
            line=dict(width=1),
            hoverinfo='text',
            hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial", font_color="black"),
            text=[f"Agent: {node}<br>Time Step: {t}<br>Total Wallet Value: {v:.4e}" for t, v in enumerate(values)],
            showlegend=False,
            visible=False
        ) for node, values in zip(nodes, wallet_values)
    ]
    
    ## Currency scatter traces
    currency_scatter_traces = {}
    for currency_pair in currency_pairs:
        currency_1, currency_2 = currency_pair
        scatter_x = np.array([node_metrics[node]['pricing_assessment'][time_step][currency_1] for node in nodes])
        scatter_y = np.array([node_metrics[node]['pricing_assessment'][time_step][currency_2] for node in nodes])
        currency_scatter_traces[currency_pair] = dict(
            x=scatter_x,
            y=scatter_y,
            mode='markers',
            marker=dict(size=8, showscale=False),
            text=[f"Agent: {node}<br>Time Step: {time_step}<br>Currency {currency_1+1} Valuation: {x:.4e}<br>Currency {currency_2+1} Valuation: {y:.4e}" for node, x, y in zip(nodes, scatter_x, scatter_y)],
            hoverinfo='text',
            hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial", font_color="black"),
            name='Pricing Valuations',
            visible=False
        )
    
    ## Node metric traces
    node_metric_traces = {}
    num_currencies = len(node_metrics[nodes[0]]['wallet'][0])
    for node in nodes:
        node_traces = {}
        wallet_data = np.array(node_metrics[node]['wallet'][:time_step+1]).T
        currency_valuation_data = np.array(node_metrics[node]['inherited_assessment'][:time_step+1]).T
        
        for i in range(num_currencies):
            node_traces[f'wallet_{i}'] = dict(
                x=time_range, 
                y=wallet_data[i], 
                mode='lines', 
                name=f'Wallet Currency {i+1}', 
                visible=False,
                hoverinfo='text',
                hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial", font_color="black"),
                text=[f"Agent: {node}<br>Time Step: {t}<br>Wallet Currency {i+1}: {v}" for t, v in enumerate(wallet_data[i])]
            )
            node_traces[f'cv_{i}'] = dict(
                x=time_range, 
                y=currency_valuation_data[i], 
                mode='lines', 
                name=f'Currency Valuation {i+1}', 
                visible=False,
                hoverinfo='text',
                hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial", font_color="black"),
                text = [f"Agent: {node}<br>Time Step: {t}<br>Currency Valuation {i+1}: {v:.4e}" for t, v in enumerate(currency_valuation_data[i])]
            )
        
        price_data = np.array(node_metrics[node]['price'][:time_step+1])
        node_traces['price'] = dict(
            x=time_range, 
            y=price_data, 
            mode='lines', 
            name='Price', 
            visible=False,
            hoverinfo='text',
            hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial", font_color="black"),
            text=[f"Agent: {node}<br>Time Step: {t}<br>Price: {v:.4f}" for t, v in enumerate(price_data)]
        )
        
        node_metric_traces[node] = node_traces
    
    return time_step, wallet_traces, currency_scatter_traces, node_metric_traces

def pre_calculate_traces(networks, node_metrics, num_timesteps, currency_pairs):
    """Pre-calculate all traces for the entire simulation in parallel, a bit slow (10 secs or so) but reduces latency in the app"""

    all_traces = {
        'wallet_traces': {},
        'currency_scatter_traces': {},
        'node_metric_traces': {}
    }
    
    process_func = partial(fast_process_time_step, networks=networks, node_metrics=node_metrics, 
                           currency_pairs=currency_pairs)
    
    with ProcessPoolExecutor() as executor:
        results = executor.map(process_func, range(num_timesteps + 1))
    
    for time_step, wallet_traces, currency_scatter_traces, node_metric_traces in results:
        all_traces['wallet_traces'][time_step] = wallet_traces
        all_traces['currency_scatter_traces'][time_step] = currency_scatter_traces
        all_traces['node_metric_traces'].update({(time_step, node): traces for node, traces in node_metric_traces.items()})
    
    return all_traces

def get_currency_pairs(df:pd.DataFrame) -> List[Tuple[int, int]]:
    """Get all possible currency pairs from the DataFrame"""
    sample_wallet = df.iloc[0]['grid'].nodes[list(df.iloc[0]['grid'].nodes)[0]]['agent'].wallet
    currencies = range(len(sample_wallet))
    return list(combinations(currencies, 2))

def load_data_and_prepare_layouts() -> Tuple[pd.DataFrame,
                                                List[nx.DiGraph],
                                                int,
                                                List[Tuple[int, int]],
                                                List[Dict[int, Tuple[float, float]]],
                                                List[Dict[int, Tuple[float, float]]],
                                                Dict[str, Any]]:
    
    """Load the simulation data and pre-calculate layouts and metrics"""
    df = load_simulation_data(get_latest_sim())
    networks = create_time_evolving_network(df)
    num_timesteps = len(df) - 1
    currency_pairs = get_currency_pairs(df)

    logging.info("Data loaded and networks created")

    logging.info("Pre-calculating visualizations...")

    logging.info("Pre-calculating force-directed layouts...")
    force_layouts = pre_calculate_force_layouts(networks)

    logging.info("Pre-calculating centrality-based layouts...")
    centrality_layouts = pre_calculate_centrality_layouts(networks)

    logging.info("Pre-calculating node metrics...")
    node_metrics = pre_calculate_node_metrics(networks)

    return df, networks, num_timesteps, currency_pairs, force_layouts, centrality_layouts, node_metrics

def create_dash_app(df:pd.DataFrame,
                     networks:List[nx.DiGraph],
                     num_timesteps:int, 
                     currency_pairs:List[Tuple[int, int]], 
                     force_layouts:List[Dict[int, Tuple[float, float]]], 
                     centrality_layouts:List[Dict[int, Tuple[float, float]]], 
                     all_traces:Dict[str, Any]) -> Dash:
    
    """Create the Dash app with the given parameters"""
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    num_currencies = len(df.iloc[0]['grid'].nodes[list(df.iloc[0]['grid'].nodes)[0]]['agent'].wallet)

    ## Custom CSS for upward expanding dropdown
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                #attribute-dropdown .Select-menu-outer,
                #wallet-lines-dropdown .Select-menu-outer {
                    top: auto !important;
                    bottom: 100% !important;
                    border-bottom-left-radius: 0px !important;
                    border-bottom-right-radius: 0px !important;
                    border-top-left-radius: 4px !important;
                    border-top-right-radius: 4px !important;
                }
                #attribute-dropdown .Select-menu,
                #wallet-lines-dropdown .Select-menu {
                    max-height: 300px !important;
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

    app.layout = html.Div([
        html.Div([
            html.Div([
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
                    options=[{'label': f'Currency {pair[0]+1} vs Currency {pair[1]+1}', 'value': f'{pair[0]},{pair[1]}'} for pair in currency_pairs],
                    value=f'{currency_pairs[0][0]},{currency_pairs[0][1]}',
                    clearable=False,
                    searchable=False,
                    style={
                        'width': '100%',
                        'marginTop': '10px',
                        'fontSize': '12px'
                    }
                ),
                dcc.RadioItems(
                    id='show-transactions',
                    options=[
                        {'label': 'Show Transactions', 'value': 'show'},
                        {'label': 'Hide Transactions', 'value': 'hide'}
                    ],
                    value='hide',
                    inline=True,
                    style={
                        'marginTop': '10px',
                        'fontSize': '12px'
                    }
                ),
            ], style={'width': '200px', 'position': 'absolute', 'left': '10px', 'top': '50px', 'zIndex': '1000'}),
            dcc.Graph(id='network-graph', style={'height': '80vh', 'width': 'calc(100% - 220px)', 'marginLeft': '220px'}),
        ], style={'position': 'relative', 'height': '80vh'}),
        html.Div([
            dcc.Dropdown(
                id='wallet-lines-dropdown',
                options=[{'label': '1', 'value': 1}],
                value=1,
                clearable=False,
                searchable=False,
                style={
                    'width': '100px',
                    'fontSize': '12px'
                }
            ),
            dcc.Dropdown(
                id='attribute-dropdown',
                options=[
                    {'label': 'All', 'value': 'all'},
                    {'label': 'Price', 'value': 'price'},
                    {'label': 'All Currencies', 'value': 'all_currencies'},
                    {'label': 'All Valuations', 'value': 'all_valuations'}
                ] +
                [{'label': f'Currency Valuation {i+1}', 'value': f'currency_valuation_{i}'} for i in range(num_currencies)] +
                [{'label': f'Wallet Currency {i+1}', 'value': f'wallet_currency_{i}'} for i in range(num_currencies)],
                value='all',
                clearable=False,
                searchable=False,
                style={
                    'width': '200px',
                    'fontSize': '12px'
                }
            ),
        ], style={'position': 'absolute', 'right': '85px', 'top': 'calc(80vh + 35px)', 'display': 'flex', 'gap': '10px'}),    
        html.Div([
            html.Div([
                dbc.Button('Play/Pause', id='play-pause-button', n_clicks=0, color="primary", className="mb-2", style={'width': '120px'}),
            ], style={'position': 'absolute', 'left': '10px', 'top': '-40px'}),
            dcc.Slider(
                id='time-slider',
                min=0,
                max=num_timesteps,
                step=1,
                value=0,
                marks={i: {'label': str(i)} for i in range(0, num_timesteps + 1, 5)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], style={'width': '95%', 'margin': '60px auto 20px auto', 'paddingTop': '20px', 'position': 'relative'}),
        dcc.Interval(
            id='interval-component',
            interval=250,
            n_intervals=0,
            disabled=True
        ),
    ])
        
    @app.callback(
        Output('network-graph', 'figure'),
        [Input('time-slider', 'value'),
        Input('layout-toggle', 'value'),
        Input('currency-pair-dropdown', 'value'),
        Input('attribute-dropdown', 'value'),
        Input('wallet-lines-dropdown', 'value'),
        Input('network-graph', 'clickData'),
        Input('show-transactions', 'value')]
    )
    def update_graph(time_step:int, layout:str, currency_pair:str, selected_attribute:str, num_wallet_lines:int, clickData:dict, show_transactions:str) -> go.Figure:
        
        global previous_node

        graph = networks[time_step]
        next_graph = networks[time_step - 1] if time_step > 0 else None
        next_graph = networks[time_step + 1] if time_step < num_timesteps else None

        ## Sometimes if the user clicks weirdly it'll return None which will crash the entire simulation
        ## We can just revert to the previous node in that case
        if clickData:
            try:
                selected_node = eval(clickData['points'][0]['text'].split('<br>')[0].split(': ')[1])
                previous_node = selected_node
            except:
                selected_node = previous_node
        else:
            random.seed(42)
            selected_node = random.choice(list(graph.nodes()))
            previous_node = selected_node

        currency_1, currency_2 = map(int, currency_pair.split(','))
            
        prev_best_vendors = {node: next_graph.nodes[node]['best_vendors'] for node in next_graph.nodes()} if next_graph else None

        fig = make_subplots(rows=3, cols=2, 
                            column_widths=[0.7, 0.3],
                            row_heights=[0.33, 0.33, 0.33],
                            specs=[
                                [{"type": "scattergl", "rowspan": 3}, {"type": "scattergl"}],
                                [None, {"type": "scattergl"}],
                                [None, {"type": "scattergl"}]
                            ],
                            vertical_spacing=0.12,
                            horizontal_spacing=0.08, 
                            subplot_titles=[
                                "", 
                                f"Currency {currency_1+1} vs {currency_2+1} Valuations",
                                f"Agent Metrics for Agent {selected_node}",
                                "Total Wallet Value Over Time"
                            ])
        
        ## Network graph (left side, full height)
        if layout == 'force':
            edge_trace, node_trace, node_colors, selected_circle, shapes, annotations = create_network_trace(
                graph, layout='force', pos=force_layouts[time_step], currency_1=currency_1, currency_2=currency_2, 
                selected_node=selected_node, prev_best_vendors=prev_best_vendors, time_step=time_step, next_graph=next_graph
            )
        elif layout == 'centrality':
            edge_trace, node_trace, node_colors, selected_circle, shapes, annotations = create_network_trace(
                graph, layout='force', pos=centrality_layouts[time_step], currency_1=currency_1, currency_2=currency_2, 
                selected_node=selected_node, prev_best_vendors=prev_best_vendors, time_step=time_step, next_graph=next_graph
            )
        else:
            edge_trace, node_trace, node_colors, selected_circle, shapes, annotations = create_network_trace(
                graph, layout='grid', currency_1=currency_1, currency_2=currency_2, 
                selected_node=selected_node, prev_best_vendors=prev_best_vendors, time_step=time_step, next_graph=next_graph
            )
            
        fig.add_trace(edge_trace, row=1, col=1)
        fig.add_trace(node_trace, row=1, col=1)
        if selected_circle:
            fig.add_trace(selected_circle, row=1, col=1)

        if(show_transactions == 'show'):
            fig.update_layout(shapes=shapes, annotations=annotations)

        if layout == 'grid':
            grid_size = int(np.sqrt(len(graph.nodes())))
            x_range = [-0.5, grid_size - 0.5]
            y_range = [-0.5, grid_size - 0.5]
        else:  ## For force-directed and centrality-based layouts
            x_coords = node_trace.x
            y_coords = node_trace.y
            x_min, x_max = min(x_coords), max(x_coords) # type: ignore (Tricky cause Plotly is a bit weird with types)
            y_min, y_max = min(y_coords), max(y_coords) # type: ignore (Tricky cause Plotly is a bit weird with types)
            x_range = [x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min)]
            y_range = [y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min)]

        fig.update_xaxes(range=x_range, row=1, col=1)
        fig.update_yaxes(range=y_range, row=1, col=1)

        ## What we're doing here below is a bit tricky. Since we're pre-calculating all the traces for all time steps to reduce latency, we need to render based of the arguments passed in.
        ## Previously I had done by rendering it twice, once for actually pre-calculating the traces and then again for rendering the graph. But that was too slow as the bottleneck was the rendering.
        ## This seems to work a lot better but it could be a bit confusing, personally I think it looks fine. You just need to think about it as assembling all the arguments and then rendering the graph based on those arguments.
        ## It's a bit like a puzzle, but instead of once piece at a time, you're assembling the whole puzzle at once. 

        ## Also we do dynamically throw things into the dictionary here based off of some user interactions that are not pre-calculated.
                
        ## Currency scatter plot (top right)
        scatter_trace_dict = all_traces['currency_scatter_traces'][time_step][(currency_1, currency_2)]
        scatter_trace_dict['marker'] = dict(
            size=8,
            color=node_colors,
            colorscale=node_trace.marker.colorscale, # type: ignore (this exists you dumb linter)
            showscale=False
        )
        scatter_trace_dict['visible'] = True
        scatter_trace = go.Scattergl(**scatter_trace_dict)
        fig.add_trace(scatter_trace, row=1, col=2)
        
        ## Node metrics graph (middle right)
        node_traces = all_traces['node_metric_traces'][(time_step, selected_node)]
        
        ## Attribute selection
        if selected_attribute == 'all':
            for trace_key, trace_dict in node_traces.items():
                trace_dict['visible'] = True
                trace = go.Scattergl(**trace_dict)
                fig.add_trace(trace, row=2, col=2)
            y_axis_title = "All Values"

        elif selected_attribute == 'price':
            node_traces['price']['visible'] = True
            trace = go.Scattergl(**node_traces['price'])
            fig.add_trace(trace, row=2, col=2)
            y_axis_title = "Price"

        elif selected_attribute == 'all_currencies':
            for i in range(num_currencies):
                node_traces[f'wallet_{i}']['visible'] = True
                trace = go.Scattergl(**node_traces[f'wallet_{i}'])
                fig.add_trace(trace, row=2, col=2)
            y_axis_title = "All Currencies"

        elif selected_attribute == 'all_valuations':
            for i in range(num_currencies):
                node_traces[f'cv_{i}']['visible'] = True
                trace = go.Scattergl(**node_traces[f'cv_{i}'])
                fig.add_trace(trace, row=2, col=2)
            y_axis_title = "All Valuations"

        elif selected_attribute.startswith('currency_valuation_'):
            currency_index = int(selected_attribute.split('_')[-1])
            node_traces[f'cv_{currency_index}']['visible'] = True
            trace = go.Scattergl(**node_traces[f'cv_{currency_index}'])
            fig.add_trace(trace, row=2, col=2)
            y_axis_title = f"Currency Valuation {currency_index+1}"

        elif selected_attribute.startswith('wallet_currency_'):
            currency_index = int(selected_attribute.split('_')[-1])
            node_traces[f'wallet_{currency_index}']['visible'] = True
            trace = go.Scattergl(**node_traces[f'wallet_{currency_index}'])
            fig.add_trace(trace, row=2, col=2)
            y_axis_title = f"Wallet Currency {currency_index+1}"
        
        ## Wallet value line graph (bottom right)
        sorted_traces = sorted(all_traces['wallet_traces'][time_step], 
                            key=lambda trace_dict: max(trace_dict['y'][:time_step+1]), 
                            reverse=True)
        
        max_lines = len(networks[time_step].nodes())
        if num_wallet_lines is None or num_wallet_lines > max_lines:
            num_wallet_lines = max(1, max_lines // 10) 
        
        visible_wallet_values = []
        for i, trace_dict in enumerate(sorted_traces):
            if i < num_wallet_lines:
                trace_dict['visible'] = True
                trace = go.Scattergl(**trace_dict)
                fig.add_trace(trace, row=3, col=2)
                visible_wallet_values.extend(trace_dict['y'][:time_step+1])
            else:
                break
        
        ## Calculate y-axis range based on the visible wallet values
        if visible_wallet_values:
            min_visible_wallet, max_visible_wallet = min(visible_wallet_values), max(visible_wallet_values)
            fig.update_yaxes(title_text="Total Wallet<br>Value", row=3, col=2, 
                            type="log", exponentformat="power", showexponent="all",
                            range=[safe_log10(min_visible_wallet*0.9), safe_log10(max_visible_wallet*1.1)],
                            title_standoff=2)
        else:
            fig.update_yaxes(title_text="Total Wallet<br>Value", row=3, col=2, 
                            type="log", exponentformat="power", showexponent="all",
                            title_standoff=2)
        
        num_agents = len(graph.nodes())
        title = f"Simulation of an Index Wallet Marketplace with {num_agents} agents | Current Time Step: {time_step}/{num_timesteps}"
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=24),
                xanchor='left',
                x=0.011
            ),
            showlegend=False,
            hovermode='closest',
            height=800,
            margin=dict(l=20, r=20, t=80, b=20),
            plot_bgcolor='rgba(227, 234, 255, 0.8)',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)

        fig.update_xaxes(title_text=f"Currency {currency_1+1} Valuation", row=1, col=2, 
                            type="log", exponentformat="power", showexponent="all",
                            title_standoff=2)
        fig.update_yaxes(title_text=f"Currency {currency_2+1}<br>Valuation", row=1, col=2, 
                        type="log", exponentformat="power", showexponent="all",
                        title_standoff=2)
        
        fig.update_xaxes(title_text="Time Step", row=2, col=2, title_standoff=2)
        fig.update_yaxes(title_text=y_axis_title, row=2, col=2, title_standoff=2, type="log", exponentformat="power", showexponent="all")
        
        fig.update_xaxes(title_text="Time Step", row=3, col=2, 
                        range=[0, num_timesteps], dtick=10,
                        title_standoff=2)
        
        fig.update_layout(
            font=dict(size=9),
            xaxis_title_font=dict(size=9),
            yaxis_title_font=dict(size=9),
            uirevision='constant'
        )
        
        ## If it's part of the network graph arrow annotation, don't update it
        ## Also it does have the annotations attribute but plotly has jack for type hints so it's a bit annoying
        for i in range(len(fig.layout.annotations)): # type: ignore
            if("▶" not in fig.layout.annotations[i].text): # type: ignore
                fig.layout.annotations[i].text += f" (Step {time_step})"  # type: ignore

        for row in range(1, 4):
            for col in range(1, 3):
                fig.update_xaxes(title_font=dict(size=14), row=row, col=col)
                fig.update_yaxes(title_font=dict(size=14), row=row, col=col)

        return fig


    @app.callback(
        Output('time-slider', 'value'),
        [Input('interval-component', 'n_intervals')],
        [State('time-slider', 'value'),
         State('time-slider', 'max'),
         State('interval-component', 'disabled')]
    )
    def update_slider(n_intervals:int, current_value:int, max_value:int, is_disabled:bool) -> int:
        """Update the time slider value"""
        if is_disabled:
            raise PreventUpdate
        if current_value >= max_value:
            return 0
        return current_value + 1

    @app.callback(
        [Output('wallet-lines-dropdown', 'options'),
        Output('wallet-lines-dropdown', 'value')],
        [Input('time-slider', 'value')],
        [State('wallet-lines-dropdown', 'value')] 
    )
    def update_wallet_lines_dropdown(time_step:int, current_value:int) -> Tuple[List[Dict[str, Any]], int]:
        num_nodes = len(networks[time_step].nodes())
        
        ## Generate options based on the number of nodes
        options = [{'label': '1', 'value': 1}]
        options.extend([{'label': str(i), 'value': i} for i in range(5, num_nodes + 1, 5) if i < num_nodes])
        options.append({'label': 'All', 'value': num_nodes})
        
        ## If no valid options are found, ensure '1' is always present
        if len(options) == 0:
            options = [{'label': '1', 'value': 1}]
        
        ## Ensure the dropdown starts with a valid value
        if current_value in [opt['value'] for opt in options]:
            return options, current_value  ## Maintain the current value if it's still valid
        else:
            ## Default to '1' if no other valid option is available
            return options, 1

    return app

def prepare_and_get_dash_app(is_debug:bool=True) -> Dash:
    """Prepare the data and create the Dash app"""

    ## Setup logging
    logging.basicConfig(level=logging.DEBUG, 
                        format='[%(asctime)s] [%(levelname)s] [%(filename)s] %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S')
    
    if not any(isinstance(handler, logging.StreamHandler) for handler in logging.getLogger('').handlers):
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(filename)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    if is_debug:
        multiprocessing.freeze_support()

    df, networks, num_timesteps, currency_pairs, force_layouts, centrality_layouts, node_metrics = load_data_and_prepare_layouts()

    logging.info("Pre-calculating traces... (this may take some time but makes the app more responsive and faster)")
    all_traces = pre_calculate_traces(networks, node_metrics, num_timesteps, currency_pairs)

    logging.info("Creating Dash app...") 
    app:Dash = create_dash_app(df, networks, num_timesteps, currency_pairs, force_layouts, centrality_layouts, all_traces)

    ## This callback is used to toggle the play/pause button
    ## It's in javascript because the python one would not work
    app.clientside_callback(
        """
        function(n_intervals, n_clicks) {
            if (n_intervals === 0 && n_clicks === 0) {
                return true;  // Start disabled
            }
            if (n_clicks === null) {
                return true;  // Keep disabled if button hasn't been clicked
            }
            // Toggle based on number of clicks (odd: enabled, even: disabled)
            return n_clicks % 2 === 0;
        }
        """,
        Output('interval-component', 'disabled'),
        Input('interval-component', 'n_intervals'),
        Input('play-pause-button', 'n_clicks')
    )

    ## This callback is used to add a spacebar listener to the play/pause button
    app.clientside_callback(
        """
        function(n_clicks) {
            if (!window.spacebarListenerAdded) {
                document.addEventListener('keydown', function(event) {
                    if (event.code === 'Space' || event.key === ' ') {
                        event.preventDefault();
                        document.getElementById('play-pause-button').click();
                    }
                });
                window.spacebarListenerAdded = true;
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output('play-pause-button', 'n_clicks'),
        Input('play-pause-button', 'n_clicks')
    )

    logging.info("Dash app created successfully... launching server")

    return app    

if __name__ == '__main__':

    ## Main guard is REQUIRED for multiprocessing to work in Windows

    app = prepare_and_get_dash_app()

    app.run(debug=True, port="8050")