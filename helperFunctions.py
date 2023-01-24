import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# create network graph
def netWorkGraph(matrix, model, printLabel=True):
    size = model.N
    # Create a graph object with 5 nodes
    G = nx.DiGraph(seed=1)
    G.add_nodes_from(list(range(size)))
    # Create a list of edge weights
    weightedEdges = []
    for i in range(size):
        for j in range(size):
            if matrix[i][j] > 0.1:
                # direction of the edge is the direction of the money flow
                weightedEdges.append((j, i, matrix[i][j]))
    G.add_weighted_edges_from(weightedEdges)
    nodeSize = matrix.sum(axis=0) * 100 
    bigLabelIndex = np.where(nodeSize >= np.percentile(nodeSize, 96))[0]
    bigLabel = [model.banks[i] if i in bigLabelIndex else "" for i in range(size)]
    # Set the labels for the nodes using a list of variables
    label_dict = {node: label for node, label in zip(G.nodes, bigLabel)}
    edges = G.edges()
    edgesWidth = [G[u][v]['weight'] * 2 for u,v in edges]
    # change the color of the center nodes
    node_colors = ['red' if node in bigLabelIndex else 'lightblue' for node in G.nodes()]
    pos = nx.fruchterman_reingold_layout(G, scale=10)
    nx.draw_networkx_nodes(G, pos, node_size=nodeSize,node_color=node_colors, alpha=0.9)
    if printLabel:
        label_options = {"ec": "k", "fc": "white", "alpha": 0.7}
        nx.draw_networkx_labels(G, pos,labels = label_dict,
                                font_size=10, bbox=label_options)
    nx.draw_networkx_edges(G, pos, width=edgesWidth, alpha=0.5, connectionstyle="arc3,rad=0.05")
    plt.axis('off')
    plt.show()

    
def simulationMonitor(agent_data, model_data, simulationSteps):
    numberOfDefault = [agent_data.xs(i, level="Step")["Default"].sum() for i in range(simulationSteps)]
    averageLeverage = [agent_data.xs(i, level="Step")["Leverage"].sum() / (100 - agent_data.xs(i, level="Step")["Default"].sum()) for i in range(simulationSteps)]
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1)
    fig.set_size_inches(40, 40)
    ax1.set_title("Single simulation average leverage")
    ax1.plot(range(simulationSteps), averageLeverage)
    ax2.plot(range(simulationSteps), [agent_data.xs(i, level="Step")["PortfolioValue"].sum() for i in range(simulationSteps)])
    ax2.set_title("Single simulation Aggregated Asset Values")
    ax3.bar(range(1, simulationSteps), np.diff(numberOfDefault))
    ax3.set_title("Single simulation Number of default banks")
    ax4.plot(np.array([[model_data["Liability Matrix"][i].sum() for i in range(simulationSteps)] for j in range(10)]).mean(axis=0))
    ax4.set_title("Size of borrowing")
    for i in [0,10,20,40,80,499]:
        ax5.plot(range(100),model_data["Trust Matrix"][i].sum(axis = 0), label = "step " + str(i))
    ax5.set_title("Accumulated belief of approving loan requests")
    plt.show()