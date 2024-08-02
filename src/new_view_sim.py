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
    """ Load simulation data from a pickle file """

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
    return ratios[0]  ## Return the ratio of the first currency as a single value

def create_network_from_grid(row):
    """ Create a networkx graph from the grid data in a DataFrame"""
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
    """ Create a list of networkx graphs from a DataFrame of simulation data """
    return [create_network_from_grid(row) for _, row in df.iterrows()]

def create_interactive_time_evolving_network(networks, df):
    """ Create an interactive plotly figure showing the time evolving networks """
    frames = []
    
    fig = make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3], 
                        specs=[[{"type": "scatter"}, {"type": "table"}]])
    
    for t, G in enumerate(networks):
        pos = {node: node for node in G.nodes()}
        node_x = [pos[node][1] for node in G.nodes()]
        node_y = [pos[node][0] for node in G.nodes()]
        node_sizes = [10 for _ in G.nodes()]
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
            x0, y0 = pos[edge[0]][1], pos[edge[0]][0]
            x1, y1 = pos[edge[1]][1], pos[edge[1]][0]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale='Viridis',  # Choose a colorscale
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
        
        avg_asmt = calc_average_valuations(df.iloc[t])
        
        # Create simplified table data for the current timestep
        table_data = create_simplified_table_data(G)
        
        table_trace = go.Table(
            header=dict(values=["Node", "Wallet Ratio"],
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=table_data,
                       fill_color='lavender',
                       align='left'),
            visible=True
        )
        
        frames.append(go.Frame(
            data=[edge_trace, node_trace, table_trace],
            name=f't{t}',
            layout=go.Layout(
                title=f"Average Valuations: Red, Blue = {avg_asmt}"
            )
        ))

    fig.add_trace(frames[0].data[0], row=1, col=1)
    fig.add_trace(frames[0].data[1], row=1, col=1)
    fig.add_trace(frames[0].data[2], row=1, col=2)

    fig.update_layout(
        title=f"Average Valuations: Red, Blue = {calc_average_valuations(df.iloc[0])}",
        showlegend=False,
        hovermode='closest',
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None, {"frame": {"duration": 500, "redraw": True},
                                       "fromcurrent": True}]),
                     dict(label="Pause",
                          method="animate",
                          args=[[None], {"frame": {"duration": 0, "redraw": True},
                                         "mode": "immediate",
                                         "transition": {"duration": 0}}])]
        )],
        sliders=[dict(
            steps=[
                dict(method='animate',
                     args=[[f't{k}'],
                           dict(mode='immediate',
                                frame=dict(duration=300, redraw=True),
                                transition=dict(duration=0))
                           ],
                     label=f'{k}'
                     ) for k in range(len(frames))
            ],
            transition=dict(duration=0),
            x=0,
            y=0,
            currentvalue=dict(font=dict(size=12), prefix="Time step: ", visible=True, xanchor="right"),
            len=1.0
        )]
    )
    
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)

    fig.frames = frames

    return fig

def create_simplified_table_data(G):
    """ Create simplified table data for the current timestep """
    nodes = sorted(G.nodes())
    wallet_ratios = [np.mean(G.nodes[node]['wallet']) if G.nodes[node]['wallet'].any() else 0 for node in nodes]
    sorted_indices = np.argsort(wallet_ratios)[::-1]  # Descending order
    sorted_nodes = [nodes[i] for i in sorted_indices]
    sorted_wallet_ratios = [wallet_ratios[i] for i in sorted_indices]
    return [sorted_nodes, sorted_wallet_ratios]

def calc_average_valuations(row):
    """ Calculate the average valuations of the grid at a given timestep """
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
    
    num_timesteps = len(df)
    print(f"Number of time steps in the simulation: {num_timesteps}")
    
    print("Displaying figure")
    fig.show()
