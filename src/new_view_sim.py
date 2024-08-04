import pickle

import pandas as pd

import numpy as np

import networkx as nx

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from util.utils import get_latest_sim

pio.renderers.default = "browser"

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

def create_network_trace(G, layout='grid'):
    """Create plotly traces for the networkx graph"""

    if layout == 'grid':
        pos = {node: node for node in G.nodes()}

    elif layout == 'force':
        pos = nx.spring_layout(G, k=1/np.sqrt(len(G.nodes())), iterations=50)
        
        ## Normalize positions to [0, 1]
        x_coords = [pos[node][0] for node in G.nodes()]
        y_coords = [pos[node][1] for node in G.nodes()]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        x_range = x_max - x_min
        y_range = y_max - y_min
        pos = {node: ((pos[node][0] - x_min) / x_range, (pos[node][1] - y_min) / y_range) for node in G.nodes()}

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]

    total_wallet_values = [sum(G.nodes[node]['wallet']) for node in G.nodes()]
    max_wallet_value = max(total_wallet_values)
    min_wallet_value = min(total_wallet_values)

    node_sizes = [5 + 25 * (value - min_wallet_value) / (max_wallet_value - min_wallet_value) 
                  if max_wallet_value != min_wallet_value else 15 for value in total_wallet_values]

    node_colors = [G.nodes[node]['color'] for node in G.nodes()]

    hover_texts = []

    for node in G.nodes():
        hover_text = f"Node: {node}<br>"
        hover_text += f"Best Vendors: {G.nodes[node]['best_vendors']}<br>"
        hover_text += f"Wallet: {G.nodes[node]['wallet']}<br>"
        hover_text += f"Currency Valuation: {G.nodes[node]['inherited_assessment']}<br>"
        hover_text += f"Price: {G.nodes[node]['price']}<br>"
        hover_text += f"Demand: {G.nodes[node]['demand']}"
        hover_texts.append(hover_text)

    edge_x, edge_y = [], []

    for edge in G.edges():
        x0, y0 = pos[edge[0]][0], pos[edge[0]][1]
        x1, y1 = pos[edge[1]][0], pos[edge[1]][1]
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

def create_interactive_time_evolving_network(networks, df):
    """Create an interactive plotly figure of the time-evolving networks"""

    frames = []
    
    fig = make_subplots(rows=2, cols=2, 
                        column_widths=[0.7, 0.3],
                        row_heights=[0.6, 0.4],
                        specs=[[{"type": "scatter", "rowspan": 2}, {"type": "scatter"}],
                               [None, {"type": "scatter"}]])
    
    wallet_data = {}


    def update_layout(layout):
        """Update the visibility of the traces based on the layout"""
        return [
            {'visible': [layout == 'grid', layout == 'grid', layout == 'force', layout == 'force', True] + [True] * len(wallet_data)},
            {'title': f"{layout.capitalize()} Layout"}
        ]

    for t, G in enumerate(networks):
        grid_edge_trace, grid_node_trace = create_network_trace(G, layout='grid')
        force_edge_trace, force_node_trace = create_network_trace(G, layout='force')
        
        scatter_x = [G.nodes[node]['pricing_assessment'][0] for node in G.nodes()]
        scatter_y = [G.nodes[node]['pricing_assessment'][1] for node in G.nodes()]
        
        log_x = [safe_log10(x) for x in scatter_x]
        log_y = [safe_log10(y) for y in scatter_y]
        
        log_x_range = [min(log_x) - 0.1, max(log_x) + 0.1]
        log_y_range = [min(log_y) - 0.1, max(log_y) + 0.1]
        
        scatter_trace = go.Scatter(
            x=scatter_x,
            y=scatter_y,
            mode='markers',
            marker=dict(
                size=8,
                color=grid_node_trace.marker.color,
                colorscale='Viridis',
                showscale=False
            ),
            text=[f"Node: {node}<br>Assessment: {G.nodes[node]['pricing_assessment']}" for node in G.nodes()],
            hoverinfo='text',
            name='Pricing Assessments'
        )
        
        for node in G.nodes():
            if node not in wallet_data:
                wallet_data[node] = []
            wallet_data[node].append(sum(G.nodes[node]['wallet']))
        
        wallet_traces = []
        for node, values in wallet_data.items():
            wallet_trace = go.Scatter(
                x=list(range(t+1)),
                y=values,
                mode='lines',
                name=f'Node {node}',
                line=dict(width=1),
                showlegend=False
            )
            wallet_traces.append(wallet_trace)
        
        avg_asmt = calc_average_valuations(df.iloc[t])
        
        frames.append(go.Frame(
            data=[grid_edge_trace, grid_node_trace, force_edge_trace, force_node_trace, scatter_trace] + wallet_traces,
            name=f't{t}',
            layout=go.Layout(
                title=f"Average Valuations: Red, Blue = {avg_asmt}",
                xaxis2=dict(range=log_x_range, type="log"),
                yaxis2=dict(range=log_y_range, type="log")
            )
        ))

    fig.add_trace(grid_edge_trace, row=1, col=1)
    fig.add_trace(grid_node_trace, row=1, col=1)
    fig.add_trace(force_edge_trace, row=1, col=1)
    fig.add_trace(force_node_trace, row=1, col=1)
    fig.add_trace(scatter_trace, row=1, col=2)
    
    for wallet_trace in wallet_traces:
        fig.add_trace(wallet_trace, row=2, col=2)

    ## Set the initial layout to grid
    for i in range(len(fig.data)):
        if i < 2: # Grid traces
            fig.data[i].visible = True
        elif i < 4:  # Force traces
            fig.data[i].visible = False
        else:  # Other traces (scatter, wallet)
            fig.data[i].visible = True

    updatemenus = [
        dict(
            type="buttons",
            direction="left",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None, {"frame": {"duration": 500, "redraw": True},
                                       "fromcurrent": True,
                                       "transition": {"duration": 300, "easing": "quadratic-in-out"}}]),
                     dict(label="Pause",
                          method="animate",
                          args=[[None], {"frame": {"duration": 0, "redraw": False},
                                         "mode": "immediate",
                                         "transition": {"duration": 0}}])],
        ),
        dict(
            buttons=[
                dict(label="Grid Layout",
                     method="update",
                     args=update_layout('grid')),
                dict(label="Force-Directed Layout",
                     method="update",
                     args=update_layout('force'))
            ],
            direction="down",
            showactive=True,
            xanchor="left",
            y=0.9,
            x=-0.13,
            yanchor="top"
        )
    ]

    fig.update_layout(
        title=f"Average Valuations: Red, Blue = {calc_average_valuations(df.iloc[0])}",
        showlegend=False,
        hovermode='closest',
        height=1000,
        updatemenus=updatemenus,
        sliders=[dict(
            active=0,
            yanchor="top",
            xanchor="left",
            currentvalue=dict(
                font=dict(size=12),
                prefix="Time step: ",
                visible=True,
                xanchor="right"
            ),
            transition=dict(duration=300, easing="cubic-in-out"),
            pad=dict(b=10, t=50),
            len=0.95, 
            x=0,
            y=0,
            steps=[dict(
                method='animate',
                args=[[f't{k}'],
                    dict(mode='immediate',
                        frame=dict(duration=500, redraw=True),
                        transition=dict(duration=300, easing="quadratic-in-out"))
                    ],
                label=f'{k}'
            ) for k in range(len(frames))]
        )]
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
                     range=[0, len(networks)-1],
                     dtick=10)
    
    all_wallet_values = [value for values in wallet_data.values() for value in values]
    min_wallet, max_wallet = min(all_wallet_values), max(all_wallet_values)
    fig.update_yaxes(title_text="Total Wallet Value", row=2, col=2, 
                     type="log",
                     exponentformat="power",
                     showexponent="all",
                     range=[safe_log10(min_wallet*0.9), safe_log10(max_wallet*1.1)])

    fig.frames = frames

    return fig

def calc_average_valuations(row):
    """Calculate the average valuation of all nodes in a row"""
    asmts = row['pricing_assessments']
    valuation = np.array([0.0, 0.0])
    for _, asmt in asmts.items():
        valuation += asmt
    valuation /= 100
    return valuation

if __name__ == "__main__":
    df = load_simulation_data(get_latest_sim())
    networks = create_time_evolving_network(df)
    print(f"Created {len(networks)} networks")
    
    fig = create_interactive_time_evolving_network(networks, df)
    
    num_timesteps = len(df) - 1
    print(f"Number of time steps in the simulation: {num_timesteps}")
    
    print("Displaying figure")
    fig.show()