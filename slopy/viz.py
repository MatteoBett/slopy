import os
from typing import Dict, List
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as bpdf
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import networkx as nx
import matplotlib.colors as mcolors

import slopy.loader as loader

sns.set_theme()

def normalize_node(name: str) -> str:
    return name.strip()

def show_graph(batches: Dict[int, List]):
    G = nx.DiGraph()
    pos = {}
    node_to_clusters = {}
    cluster_colors = {}

    x_spacing = 10
    y_spacing = 1.5

    # Step 1: Build cluster-to-nodes map
    cluster_id = 0
    all_clusters = []
    size_cluster_map = defaultdict(list)  # map size -> list of (cluster_id, nodes)

    for size, listseq in batches.items():
        seen_clusters = set()
        for i, seq in enumerate(listseq):
            if seq.cluster_seq:
                cluster = tuple(sorted(set(seq.cluster_seq + [i])))
                if cluster not in seen_clusters:
                    cluster_nodes = [f"{size}_{idx}" for idx in cluster]
                    all_clusters.append((cluster_id, cluster_nodes))
                    size_cluster_map[size].append((cluster_id, cluster_nodes))
                    seen_clusters.add(cluster)
                    cluster_id += 1

    # Step 2: Assign colors to clusters
    available_colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    np.random.seed(42)
    np.random.shuffle(available_colors)
    for i, (cid, _) in enumerate(all_clusters):
        cluster_colors[cid] = available_colors[i % len(available_colors)]

    # Step 3: Build node-to-clusters map
    for cid, nodes in all_clusters:
        for n in nodes:
            node_to_clusters.setdefault(n, set()).add(cid)

    # Step 4: Build nodes, edges, positions
    node_cluster_sets = {}
    for col_index, (size, listseq) in enumerate(sorted(batches.items())):
        x = col_index * -x_spacing
        placed = set()

        # Compute vertical center for this size group
        num_rows = 0
        for cid, nodes in size_cluster_map[size]:
            num_rows += len(nodes) + 1  # +1 for spacing between clusters
        num_rows += sum(1 for i in range(len(listseq)) if f"{size}_{i}" not in node_to_clusters)
        total_height = num_rows * y_spacing
        y_cursor = total_height / 2

        # First place clustered sequences grouped by cluster
        for cid, nodes in size_cluster_map[size]:
            nodes_sorted = sorted(nodes, key=lambda n: int(n.split('_')[1]))
            for n in nodes_sorted:
                G.add_node(n)
                pos[n] = (x, y_cursor - y_spacing)
                y_cursor -= y_spacing
                node_cluster_sets[n] = node_to_clusters.get(n, set())
                placed.add(n)
            y_cursor -= y_spacing  # extra space between clusters

        # Then place unclustered sequences
        for i, seq in enumerate(listseq):
            n = f"{size}_{i}"
            if n in placed:
                continue
            G.add_node(n)
            pos[n] = (x, y_cursor - y_spacing)
            y_cursor -= y_spacing
            node_cluster_sets[n] = node_to_clusters.get(n, set())

        # Add edges
        for row_index, seq in enumerate(listseq):
            source = normalize_node(f"{size}_{row_index}")
            for key, ilist in seq.targets.items():
                for elt in ilist:
                    target = normalize_node(f"{key}_{elt}")
                    G.add_node(target)
                    if target not in pos:
                        tx = x + x_spacing * 2
                        ty = -elt * y_spacing
                        pos[target] = (tx, ty)
                    G.add_edge(source, target)

    # Step 5: Draw edges
    plt.figure(figsize=(14, 10))
    nx.draw_networkx_edges(G, pos=pos, edge_color='gray', arrows=True)

    # Step 6: Draw nodes
    ax = plt.gca()
    radius = 0.4
    for node, (x, y) in pos.items():
        clusters = node_cluster_sets.get(node, set())
        if not clusters:
            circle = plt.Circle((x, y), radius, color="#CCCCCC", zorder=2)
            ax.add_patch(circle)
        elif len(clusters) == 1:
            color = cluster_colors[list(clusters)[0]]
            circle = plt.Circle((x, y), radius, color=color, zorder=2)
            ax.add_patch(circle)
        else:
            clusters_list = list(clusters)
            num = len(clusters_list)
            angle_per_cluster = 360 / num
            for i, cid in enumerate(clusters_list):
                theta1 = i * angle_per_cluster
                theta2 = theta1 + angle_per_cluster
                wedge = mpatches.Wedge(
                    (x, y), radius, theta1, theta2, facecolor=cluster_colors[cid], zorder=2, edgecolor='black'
                )
                ax.add_patch(wedge)

    # Draw labels
    for node, (x, y) in pos.items():
        ax.text(x, y, node, ha='center', va='center', fontsize=8, zorder=3)

    plt.axis('equal')
    plt.axis('off')
    plt.title("Graph with Clustered and Size-Aligned Nodes (Vertically Centered)")
    plt.tight_layout()
    plt.show()
    
def get_length_distribution(batches : Dict[int, List[loader.SeqSlope]], pdf : bpdf.PdfPages):
    fig, ax = plt.subplots(1, 1, figsize = (14,8))

    sizes = list(batches.keys())
    counts = [len(batch) for batch in batches.values()]

    ax.bar(sizes, counts)
    ax.set_xlabel("Sequence size (nuc)")
    ax.set_ylabel("Occurence count")
    ax.set_title(f"Sequence's size distribution")

    fig.savefig(pdf, format='pdf')
    plt.close(fig)

def make_heatmap(pdf : bpdf.PdfPages, distmat : np.matrix, size : int, k:int):
    fig, ax = plt.subplots(1, 1, figsize = (10,8))
    sns.heatmap(data=distmat, cmap="magma", cbar=True, ax=ax)
    
    ax.set_title(f"Jaccard distance matrix between sequences of size {size} and k = {k}")
    fig.savefig(pdf, format='pdf')
    plt.close(fig)


def show_inter_dist(batches : Dict[int, List[loader.SeqSlope]], pdf : bpdf.PdfPages, k : int):
    mat = np.zeros((len(batches.keys()), len(batches.keys())))

    for i, (_, batch) in enumerate(batches.items()):
        tmp = np.array([np.mean(list(seq.targets.values())) for seq in batch]).mean(axis=0)
        mat[i, i+1:] = tmp

    fig, ax = plt.subplots(1, 1, figsize = (14,8))
    sns.heatmap(data=mat, cmap="magma", cbar=True, ax=ax)
    
    ax.set_title(f"Jaccard distance matrix between sequences with k = {k}")
    fig.savefig(pdf, format='pdf')
    plt.close(fig)


def main_display(path_report : str, batches : Dict[int, loader.SeqSlope]) -> bpdf.PdfPages:
    pdf = bpdf.PdfPages(path_report)

    get_length_distribution(batches=batches, pdf=pdf)

    return pdf

