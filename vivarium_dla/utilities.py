import numpy as np, pandas as pd, matplotlib.pyplot as plt, networkx as nx, sklearn.neighbors

def make_graph(df: pd.DataFrame) -> nx.Graph:
    """
    Generate a NetworkX graph consisting of edges between
    df.index[i] and df.frozen[i].

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the nodes.

    Returns:
        networkx.Graph: A graph object with edges between the nodes.
        
    Note:
        this function requires the indices of df to be unique
    """
    assert df.index.nunique() == len(df), 'index values of df must be unique'
    graph = nx.DiGraph()
    for v1 in df.index:
        v2 = df.frozen[v1]
        graph.add_edge(v1, v2)
    
    return graph

# Test 1: Empty DataFrame
df_empty = pd.DataFrame(columns=['frozen', 'x', 'y', 'z'])
graph_empty = make_graph(df_empty)
assert len(graph_empty.nodes()) == 0
assert len(graph_empty.edges()) == 0

# Test 2: Single-row DataFrame
df_single_row = pd.DataFrame({'frozen': ['A']}, index=['1'],)
graph_single_row = make_graph(df_single_row)
assert len(graph_single_row.nodes()) == 2
assert len(graph_single_row.edges()) == 1
assert ('1', 'A') in graph_single_row.edges()

# # Test 3: DataFrame with duplicate index values
# # don't allow this case (added an assert )
# df_duplicate_index = pd.DataFrame({'frozen': ['A', 'B', 'C']}, index=['1', '1', '1'])
# graph_duplicate_index = make_graph(df_duplicate_index)
# assert len(graph_duplicate_index.nodes()) == 4
# assert len(graph_duplicate_index.edges()) == 3
# assert ('1', 'A') in graph_duplicate_index.edges()
# assert ('1', 'B') in graph_duplicate_index.edges()
# assert ('1', 'C') in graph_duplicate_index.edges()

# Test 4: DataFrame with complex index values
df_complex_index = pd.DataFrame({'frozen': ['A', 'B', 'C']}, index=[('1', 'a'), ('2', 'b'), ('3', 'c')])
graph_complex_index = make_graph(df_complex_index)
assert len(graph_complex_index.nodes()) == 6
assert len(graph_complex_index.edges()) == 3
assert (('1', 'a'), 'A') in graph_complex_index.edges()
assert (('2', 'b'), 'B') in graph_complex_index.edges()
assert (('3', 'c'), 'C') in graph_complex_index.edges()


def find_leaf_nodes(graph: nx.Graph) -> list:
    """
    Find the leaf nodes (nodes with degree 1) in the given graph.

    Parameters:
        graph (nx.Graph): The input graph.

    Returns:
        list: A list of leaf nodes in the graph.
        
    """
    if len(graph.nodes()) == 1:  # special case
        return graph
    
    leaf_nodes = [node for node in graph.nodes() if graph.degree[node] == 1]
    return leaf_nodes

# Test 1: Empty graph
graph_empty = nx.Graph()
leaf_nodes_empty = find_leaf_nodes(graph_empty)
assert len(leaf_nodes_empty) == 0

# Test 2: Single leaf node  # caught an edge case!
graph_single_leaf = nx.Graph()
graph_single_leaf.add_node('A')
leaf_nodes_single = find_leaf_nodes(graph_single_leaf)
assert len(leaf_nodes_single) == 1
assert 'A' in leaf_nodes_single

# Test 3: Graph with multiple leaf nodes
graph_multiple_leaves = nx.Graph()
graph_multiple_leaves.add_edge('A', 'B')
graph_multiple_leaves.add_edge('B', 'C')
graph_multiple_leaves.add_edge('D', 'E')
graph_multiple_leaves.add_edge('E', 'F')
graph_multiple_leaves.add_edge('F', 'G')
leaf_nodes_multiple = find_leaf_nodes(graph_multiple_leaves)
assert len(leaf_nodes_multiple) == 4
assert 'A' in leaf_nodes_multiple
assert 'C' in leaf_nodes_multiple
assert 'D' in leaf_nodes_multiple
assert 'G' in leaf_nodes_multiple


def plot_vessels(pop, color):
    for i in pop[~pop.frozen.isnull()].index:
        j = pop.loc[i, 'frozen']
        xx = [pop.x[i], pop.x[j]]
        yy = [pop.y[i], pop.y[j]]
        plt.plot(xx, yy, '-', alpha=.5, linewidth=1, color=color)


    bnds = plt.axis()
    max_bnd = np.max(bnds)
    plt.axis(xmin=-max_bnd, xmax=max_bnd, ymin=-max_bnd, ymax=max_bnd)


def match_A_leaves(dfa: pd.DataFrame, dfb: pd.DataFrame) -> nx.Graph:
    """
    Take DataFrames dfa and dfb, convert them to graphs using make_graph,
    and return a digraph that is a has edges between
    leaves of dfa nearest point in dfb.

    Parameters:
        dfa (pandas.DataFrame): The first input DataFrame.
        dfb (pandas.DataFrame): The second input DataFrame.

    Returns:
        networkx.DiGraph: A digraph object representing the matching
        between the leaves of dfa and dfb.
    """
    # Convert DataFrames to graphs
    graph_a = make_graph(dfa)

    # Get the leaf nodes of each graph
    leaves_a = find_leaf_nodes(graph_a)
    nodes_b = np.array(dfb.index, dtype='int')

    # Extract the coordinates (x, y, z) of the leaf nodes in each graph
    coordinates_a = [dfa.loc[node, ['x', 'y', 'z']].tolist() for node in leaves_a]
    coordinates_b = dfb.loc[:, ['x', 'y', 'z']].values

    # Use KDTree to find the closest points between the leaf nodes of A and B
    kdtree = sklearn.neighbors.KDTree(coordinates_b)
    distances, indices = kdtree.query(coordinates_a, k=1)

    # Create a new graph for the matching
    connecting_graph = nx.DiGraph()

    # Add edges between nearby leaves based on the calculated distances
    for i, j in enumerate(indices):
        node_a = int(leaves_a[i])
        node_b = int(nodes_b[j])
        if distances[i] < 5:        # TODO: only include edges within a certain critical radius
            connecting_graph.add_edge(('a',node_a), ('b',node_b))

    return connecting_graph



def pruned_network(dfa, dfb, M):
    """add direected edges from dfa and dfb to M and then restrict to vertices reachable from dfa
    """
    G = nx.DiGraph()
    
    for u,v in make_graph(dfa).edges(): 
        G.add_edge(('a', int(v)), ('a', int(u)))

    for u,v in make_graph(dfb).edges():
        G.add_edge(('b', int(u)), ('b', int(v)))
        
    for u, v in M.edges():
        G.add_edge(u,v)

    # now restrict G to only nodes
    # reachable by a directed path from node ('a', 0)
    
    reachable_nodes = nx.descendants(G, ('a', 0))
    # todo: reverse edges and get ancestors of ('b', 0)
    G_sub1 = G.subgraph(reachable_nodes)
    if ('b', 0) not in G_sub1.nodes():
        return G_sub1
    else:
        reversed_G_sub1 = G_sub1.reverse()
        reachable_nodes = nx.descendants(reversed_G_sub1, ('b', 0))
        G_sub2 = G_sub1.subgraph(reachable_nodes)
        
        return G_sub2




def plot_vascular_network(G, dfa, dfb):
    color_a = 'C3'
    color_b = 'C0'
    color_ab = 'C4'
    
    for (p1,i),(p2,j) in G.edges():
        if p1 == p2 == 'a':
            xx = [dfa.x[i], dfa.x[j]]
            yy = [dfa.y[i], dfa.y[j]]
            color = color_a
            linewidth=1
        elif p1 == p2 == 'b':
            xx = [dfb.x[i], dfb.x[j]]
            yy = [dfb.y[i], dfb.y[j]]
            color = color_b
            linewidth=1
        else:
            xx = [dfa.x[i], dfb.x[j]]
            yy = [dfa.y[i], dfb.y[j]]
            color = color_ab
            linewidth=1
        plt.plot(xx, yy, '-', alpha=.25, linewidth=linewidth, color=color)



def interpolate_edges(G, dfa, dfb):
    """interpolate edges of G using dfa and dfb
    
    Parameters:
        G (nx.Graph): a graph with edges between nodes of dfa and dfb
        dfa (pd.DataFrame): a DataFrame with columns x, y, z
        dfb (pd.DataFrame): a DataFrame with columns x, y, z

    Returns:
        np.array: an array of interpolated points 
    """
    dots_per_unit = 500
    interpolated_points = []
    
    for (p1,i),(p2,j) in G.edges():
        if p1 == p2 == 'a':
            xx = [dfa.x[i], dfa.x[j]]
            yy = [dfa.y[i], dfa.y[j]]
            zz = [dfa.z[i], dfa.z[j]]
            type = 'a'
        elif p1 == p2 == 'b':
            xx = [dfb.x[i], dfb.x[j]]
            yy = [dfb.y[i], dfb.y[j]]
            zz = [dfb.z[i], dfb.z[j]]
            type='b'
        else:
            xx = [dfa.x[i], dfb.x[j]]
            yy = [dfa.y[i], dfb.y[j]]
            zz = [dfa.z[i], dfb.z[j]]
            type='c'

        x0,x1 = xx
        y0,y1 = yy
        z0,z1 = zz
        
        edge_length = np.sqrt((x0-x1)**2 + (y0-y1)**2 + (z0-z1)**2)
        n_points = int(np.floor(edge_length * dots_per_unit) + 2)
        if type == 'b':
            n_points *= 4
        alpha = np.linspace(0, 1, n_points)
        
        points_to_add = ([x0,y0,z0] + np.outer(alpha, [x1-x0, y1-y0, z1-z0]))

        if type == 'b':
            points_to_add += np.random.normal(0, (1/dots_per_unit) * 200, points_to_add.shape)
        interpolated_points += points_to_add.tolist()
        
    interpolated_points = np.array(interpolated_points)
    
    # add noise to interpolated points
    interpolated_points += np.random.normal(0, (1/dots_per_unit) * 100, interpolated_points.shape)

    return interpolated_points


def to_voxels(G, dfa, dfb):
    """interpolate edges of G using dfa and dfb
    
    Parameters:
        G (nx.Graph): a graph with edges between nodes of dfa and dfb
        dfa (pd.DataFrame): a DataFrame with columns x, y, z
        dfb (pd.DataFrame): a DataFrame with columns x, y, z

    Returns:
        np.array: a 3d array of point counts
    """

    points = interpolate_edges(G, dfa, dfb)
    hist, (xedges, yedges, zedges) = np.histogramdd(points, bins=(500, 500, 50))

    return hist

def build_network(fname1, fname2, scale_factor):
    dfa = pd.read_csv(fname1, index_col=0)
    dfa = dfa[np.isfinite(dfa.frozen)].copy()  # just want frozen nodes

    dfb = pd.read_csv(fname2, index_col=0)
    dfb = dfb[np.isfinite(dfb.frozen)].copy()  # just want frozen nodes

    dfa['x'] = dfa['x']/scale_factor
    dfa['y'] = dfa['y']/scale_factor

    M = match_A_leaves(dfa, dfb)
    N = pruned_network(dfa, dfb, M)

    return N, dfa, dfb
