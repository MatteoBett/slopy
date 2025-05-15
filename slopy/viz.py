import os
from typing import Dict, List
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as bpdf
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import networkx as nx
import matplotlib.colors as mcolors
import pandas as pd

import slopy.loader as loader

sns.set_theme()

def normalize_node(name: str) -> str:
    return name.strip()


def show_graph(batches: Dict[int, List[loader.SeqSlope]]):
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

    for size, seqs in batches.items():
        seen_clusters = set()
        for i, seq in enumerate(seqs):
            cluster = tuple(sorted(set([seq.node] + seq.cluster_seq)))
            if (cluster not in seen_clusters):
                all_clusters.append((cluster_id, list(cluster)))
                size_cluster_map[size].append((cluster_id, list(cluster)))
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
        num_rows += sum(1 for seq in listseq if normalize_node(seq.node) not in node_to_clusters)
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
        for seq in listseq:
            n = normalize_node(seq.node)
            if n in placed:
                continue
            G.add_node(n)
            pos[n] = (x, y_cursor - y_spacing)
            y_cursor -= y_spacing
            node_cluster_sets[n] = node_to_clusters.get(n, set())

        # Add edges
        for seq in listseq:
            source = normalize_node(seq.node)
            for key, ilist in seq.targets.items():
                for ti, elt in enumerate(ilist):
                    target = normalize_node(f"{elt}")
                    G.add_node(target)
                    if target not in pos:
                        tx = x + x_spacing * 2
                        ty = -ti * y_spacing
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

def make_heatmap(pdf : bpdf.PdfPages, mat : np.matrix, size : int, k:int):
    fig, ax = plt.subplots(1, 1, figsize = (10,8))
    sns.heatmap(data=mat, cmap="magma", cbar=True, ax=ax)
    
    ax.set_title(f"Adjacency and degree matrix between sequences of size {size} and k = {k}")
    fig.savefig(pdf, format='pdf')
    plt.close(fig)


"""
    sns.barplot(data=df, x="len_size", y="tot_degrees", ax=axes[0], legend=True, color='red')
    #sns.histplot(data=df, x="len_size",  kde=True, bins=50, ax=axes[0], element="step", stat="density", legend=True, color="blue")
    sns.barplot(data=df, x="len_size", y="out_degrees", ax=axes[1], legend=True, color='red')
    #sns.histplot(data=df, x="len_size",  kde=True, bins=50, ax=axes[1], element="step", stat="density", legend=True, color="blue")    
    sns.barplot(data=df, x="len_size", y="in_degrees", ax=axes[2], legend=True, color='red')
    #sns.histplot(data=df, x="len_size",  kde=True, bins=50, ax=axes[2], element="step", stat="density", legend=True, color="blue")   
"""

def degree_v_size(batches: Dict[int, List[loader.SeqSlope]], size_fam: int, pdf: bpdf.PdfPages):
    in_deg = np.zeros(shape=(size_fam,))
    out_deg = np.zeros(shape=(size_fam,))
    tot_deg = np.zeros(shape=(size_fam,))
    size = np.zeros(shape=(size_fam,))

    for i, (length, batch) in enumerate(batches.items()):
        for seq in batch:
            deg = 0
            o_deg = 0
            i_deg = 0
            for targ in seq.targets.values():
                deg += len(targ)
                o_deg += len(targ)
            for ancs in seq.ancestor.values():
                deg += len(ancs)
                i_deg += len(ancs)

            in_deg[seq._id] = i_deg
            out_deg[seq._id] = o_deg
            tot_deg[seq._id] = deg
            size[seq._id] = length

    df = pd.DataFrame({
        "len_size": size,
        "in_degrees": in_deg,
        "out_degrees": out_deg,
        "tot_degrees": tot_deg,
    })

    df["in_degrees"] /= df["in_degrees"].sum()
    df["out_degrees"] /= df["out_degrees"].sum()
    df["tot_degrees"] /= df["tot_degrees"].sum()

    grouped = df.groupby("len_size").mean(numeric_only=True).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharex=True, sharey=True)

    sns.lineplot(data=grouped, x="len_size", y="tot_degrees", ax=axes[0], legend=True, color='red')
    sns.histplot(data=df, x="len_size", kde=True, bins=50, ax=axes[0], element="step", stat="density", legend=True, color="blue")

    sns.lineplot(data=grouped, x="len_size", y="out_degrees", ax=axes[1], legend=True, color='red')
    sns.histplot(data=df, x="len_size", kde=True, bins=50, ax=axes[1], element="step", stat="density", legend=True, color="blue")

    sns.lineplot(data=grouped, x="len_size", y="in_degrees", ax=axes[2], legend=True, color='red')
    sns.histplot(data=df, x="len_size", kde=True, bins=50, ax=axes[2], element="step", stat="density", legend=True, color="blue")

    axes[0].set_title("Sequences' all nodes degree depending on their size")
    axes[1].set_title("Sequences' target nodes degree depending on their size")
    axes[2].set_title("Sequences' ancestor nodes degree depending on their size")

    legend_handles = [
        mpatches.Patch(color=sns.color_palette("tab10")[0], label="Seq Size distribution"),
        mpatches.Patch(color='red', label="Avg normalized degree")
    ]
    fig.legend(handles=legend_handles, loc=[0.35, 0.95], ncol=3, fontsize='large')

    axes[0].set_xlabel("Sequence length (nuc)")
    axes[0].set_ylabel("Density")
    axes[0].set_xticks(np.arange(start=size.min(), stop=size.max(), step=10))
    fig.savefig(pdf, format='pdf')
    plt.close(fig)

def plot_size_v_lenpath(batches : Dict[str, loader.SeqSlope], pdf : bpdf.PdfPages):
    lenpaths = [seq.lenpath for _, seq in batches.items()]
    sizes = [len(seq.seq) for _, seq in batches.items()]
    count_sizes = Counter(sizes)
    
    tmp = {size : [] for size in set(sizes)}
    for node, seq in batches.items():
        tmp[len(seq.seq)].append(seq.lenpath)

    avg_len = {k:np.mean(v) for k, v in tmp.items()}

    fig, ax = plt.subplots(1, 1, figsize=(18,5))

    ax2 = ax.twinx()
    ax2.bar(list(count_sizes.keys()), list(count_sizes.values()), zorder=1)
    ax2.set_ylabel("Sequences count")
    ax.scatter(sizes, lenpaths, c='r', zorder=2)
    ax.plot(list(avg_len.keys()), list(avg_len.values()), c='black', marker='.', zorder=2)

    ax.set_xlabel("Sequence size (bp)")
    ax.set_ylabel("Path lengths (per seq)")
    ax.set_title("Path length depending on size")
    
    fig.savefig(pdf, format='pdf')
    plt.close(fig)   


def slope_length(batches : Dict[str, loader.SeqSlope], pdf : bpdf.PdfPages):
    diffsize, lengths, std = [], [], []

    for _, seq in batches.items():
        if len(seq.path) > 0:
            diffsize.append((int(seq.path[0].split("_")[0])-int(seq.path[-1].split("_")[0]))/int(seq.path[0].split("_")[0]))
            lengths.append(len(seq.seq))
            std.append(np.std([int(i.split("_")[0]) for i in seq.path]))
    
    sizes = pd.DataFrame({"sizes":diffsize, "lengths":lengths, "std":std})
        
    fig, axes = plt.subplots(1, 3, figsize=(18,5))

    sns.histplot(data=sizes, x="sizes", ax=axes[0])
    axes[0].set_title(f"Density of size reduction through the slope")
    axes[0].set_xlabel("% of sequence size reduction")

    sns.lineplot(data=sizes, x="lengths", y="sizes", ax=axes[1])
    axes[1].set_title(f"% reduction of sequences depending on starting sequence's size")
    axes[1].set_xlabel("Starting sequence's size")
    axes[1].set_ylabel("% of sequence size reduction")

    sns.lineplot(data=sizes, x="lengths", y="std", ax=axes[2])
    axes[2].set_title(f"standard stepwise sequence's reduction depending on starting sequence's size")
    axes[2].set_xlabel("Starting sequence's size")
    axes[2].set_ylabel("standard sequence size reduction")

    fig.savefig(pdf, format='pdf')
    plt.close(fig)



def show_inter_dist(batches : Dict[int, List[loader.SeqSlope]], pdf : bpdf.PdfPages, k : int):
    mat = np.zeros((len(batches.keys()), len(batches.keys())))

    for i, (_, batch) in enumerate(batches.items()):
        tmp = np.array([np.mean(list(seq.targets_sim.values())) for seq in batch]).mean(axis=0)
        mat[i, i+1:] = tmp

    fig, ax = plt.subplots(1, 1, figsize = (14,8))
    sns.heatmap(data=mat, cmap="magma", cbar=True, ax=ax)
    
    ax.set_title(f"Jaccard distance matrix between sequences with k = {k}")
    fig.savefig(pdf, format='pdf')
    plt.close(fig)


def spectral_clustering(Laplacian : np.matrix, batches:Dict[int, List[loader.SeqSlope]], pdf : bpdf.PdfPages):
    sizes = np.zeros((len(Laplacian,)))
    degrees = np.zeros((len(Laplacian,)))
    _, eigenvectors = np.linalg.eigh(Laplacian)

    for size, batch in batches.items():
        for elt in batch:
            sizes[elt._id] = size
            degrees[elt._id] = len(elt.ancestor) + len(elt.targets)

    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(projection='3d')
    sc = ax.scatter(eigenvectors[:, 1], eigenvectors[:, 2], eigenvectors[:,3], c=sizes, cmap='magma')  # Optional: set colormap
    plt.colorbar(sc, ax=ax)
    pdf.savefig(fig)
    plt.close(fig)
    
    fig1 = plt.figure(figsize=(12,10))
    ax1 = fig1.add_subplot(projection='3d')
    sc1 = ax1.scatter(eigenvectors[:, 1], eigenvectors[:, 2], eigenvectors[:,3], c=degrees, cmap='magma')  # Optional: set colormap
    plt.colorbar(sc1, ax=ax1)
    
    pdf.savefig(fig1)
    plt.close(fig1)

def main_display(path_report : str, batches : Dict[int, loader.SeqSlope]) -> bpdf.PdfPages:
    pdf = bpdf.PdfPages(path_report)

    get_length_distribution(batches=batches, pdf=pdf)

    return pdf

