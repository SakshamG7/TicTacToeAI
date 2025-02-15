#!/usr/bin/env python3
"""
Visualization for Saksham's Self Learning Neural Network

This script loads a JSON file representing the network and creates a visual graph.
Neurons are represented as nodes:
    - Green: Input neurons
    - Red: Output neurons
    - Blue: Hidden neurons
Connections are drawn as directed edges with weight labels.

Usage:
    python visualize_network.py ssnn.json
"""

import json
import sys
import networkx as nx
import matplotlib.pyplot as plt

def visualize_network(filename: str):
    # Load network data from JSON file
    with open(filename, 'r') as file:
        data = json.load(file)

    neurons = data.get('neurons', {})
    connections = data.get('connections', {})
    input_ids = set(data.get('input_ids', []))
    output_ids = set(data.get('output_ids', []))
    
    G = nx.DiGraph()

    # Add neurons as nodes with appropriate color and label.
    for neuron_id_str, neuron_data in neurons.items():
        # Note: JSON keys are strings. Convert to int for proper comparison.
        neuron_id = int(neuron_id_str)
        bias, usage, value = neuron_data
        if neuron_id in input_ids:
            node_color = 'green'
        elif neuron_id in output_ids:
            node_color = 'red'
        else:
            node_color = 'blue'
        label = f"ID: {neuron_id}\nBias: {bias:.2f}"
        # Use the string version of the neuron id as the node key.
        G.add_node(str(neuron_id), label=label, color=node_color)

    # Add connections as edges with weight labels.
    for connection_id_str, connection_data in connections.items():
        source, target, weight, usage = connection_data
        # Convert source and target to string to match node keys.
        G.add_edge(str(source), str(target), label=f"{weight:.2f}")

    # Use a spring layout for positioning nodes.
    pos = nx.spring_layout(G)

    # Prepare node colors and labels.
    node_colors = [G.nodes[node].get('color', 'blue') for node in G.nodes()]
    node_labels = {node: G.nodes[node].get('label', '') for node in G.nodes()}
    edge_labels = nx.get_edge_attributes(G, 'label')

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, labels=node_labels,
            node_color=node_colors, node_size=1500, font_size=10, arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='gray')
    plt.title("Visualization of Self Learning Neural Network")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python visualize_network.py <network_file.json>")
        sys.exit(1)
    visualize_network(sys.argv[1])
