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
    ## May also want to change in general to something else
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
    
    fig = make_subplots(rows=2, cols=2, 
                        column_widths=[0.7, 0.3],
                        row_heights=[0.6, 0.4],
                        specs=[[{"type": "scatter", "rowspan": 2}, {"type": "scatter"}],
                               [None, {"type": "scatter"}]])
    
    wallet_data = {}

    def safe_log10(x):
        return np.log10(max(abs(x), 1e-10))

    for t, G in enumerate(networks):
        pos = {node: node for node in G.nodes()}
        node_x = [pos[node][1] for node in G.nodes()]
        node_y = [pos[node][0] for node in G.nodes()]
        
        total_wallet_values = [sum(G.nodes[node]['wallet']) for node in G.nodes()]
        max_wallet_value = max(total_wallet_values)
        min_wallet_value = min(total_wallet_values)
        
        ## Scale da node sizes between 5 and 30 based on wallet value
        node_sizes = [5 + 25 * (value - min_wallet_value) / (max_wallet_value - min_wallet_value) 
                      if max_wallet_value != min_wallet_value else 15 for value in total_wallet_values]
        
        node_colors = [G.nodes[node]['color'] for node in G.nodes()]
        
        hover_texts = []
        for node in G.nodes():
            if node not in wallet_data:
                wallet_data[node] = []
            wallet_data[node].append(sum(G.nodes[node]['wallet']))
            
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
        
        scatter_x = [G.nodes[node]['pricing_assessment'][0] for node in G.nodes()]
        scatter_y = [G.nodes[node]['pricing_assessment'][1] for node in G.nodes()]
        
        log_x = [safe_log10(x) for x in scatter_x]
        log_y = [safe_log10(y) for y in scatter_y]
        
        log_x_min, log_x_max = min(log_x), max(log_x)
        log_y_min, log_y_max = min(log_y), max(log_y)
        
        x_padding = (log_x_max - log_x_min) * 0.1
        y_padding = (log_y_max - log_y_min) * 0.1
        
        log_x_range = [log_x_min - x_padding, log_x_max + x_padding]
        log_y_range = [log_y_min - y_padding, log_y_max + y_padding]
        
        scatter_trace = go.Scatter(
            x=scatter_x,
            y=scatter_y,
            mode='markers',
            marker=dict(
                size=8,
                color=node_colors,
                colorscale='Viridis',
                showscale=False
            ),
            text=[f"Node: {node}<br>Assessment: {G.nodes[node]['pricing_assessment']}" for node in G.nodes()],
            hoverinfo='text',
            name='Pricing Assessments'
        )
        
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
            data=[edge_trace, node_trace, scatter_trace] + wallet_traces,
            name=f't{t}',
            layout=go.Layout(
                title=f"Average Valuations: Red, Blue = {avg_asmt}",
                xaxis2=dict(range=log_x_range, type="log"),
                yaxis2=dict(range=log_y_range, type="log")
            ),
            traces=[0, 1, 2] + list(range(3, 3 + len(wallet_traces)))
        ))

    # Set initial ranges using the first frame
    initial_log_x_range = frames[0].layout.xaxis2.range
    initial_log_y_range = frames[0].layout.yaxis2.range

    fig.add_trace(frames[0].data[0], row=1, col=1)
    fig.add_trace(frames[0].data[1], row=1, col=1)
    fig.add_trace(frames[0].data[2], row=1, col=2)
    
    for wallet_trace in frames[0].data[3:]:
        fig.add_trace(wallet_trace, row=2, col=2)

    fig.update_layout(
        title=f"Average Valuations: Red, Blue = {calc_average_valuations(df.iloc[0])}",
        showlegend=False,
        hovermode='closest',
        height=1000,
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None, {"frame": {"duration": 500, "redraw": True},
                                       "fromcurrent": True,
                                       "transition": {"duration": 300, "easing": "quadratic-in-out"}}]),
                     dict(label="Pause",
                          method="animate",
                          args=[[None], {"frame": {"duration": 0, "redraw": False},
                                         "mode": "immediate",
                                         "transition": {"duration": 0}}])]
        )],
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
                     showexponent="all",
                     range=initial_log_x_range)
    fig.update_yaxes(title_text="Blue Currency Assessment", row=1, col=2, 
                     type="log", 
                     exponentformat="power",
                     showexponent="all",
                     range=initial_log_y_range)
    
    fig.update_xaxes(title_text="Time Step", row=2, col=2, 
                     range=[0, len(networks)-1],
                     dtick=10)  # Set tick interval to 10
    
    all_wallet_values = [value for values in wallet_data.values() for value in values]
    min_wallet, max_wallet = min(all_wallet_values), max(all_wallet_values)
    fig.update_yaxes(title_text="Total Wallet Value", row=2, col=2, 
                     type="log",
                     exponentformat="power",
                     showexponent="all",
                     range=[safe_log10(min_wallet*0.9), safe_log10(max_wallet*1.1)])

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
    
    num_timesteps = len(df) - 1
    print(f"Number of time steps in the simulation: {num_timesteps}")
    
    print("Displaying figure")
    fig.show()