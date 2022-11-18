"""
This was taken from: https://github.com/deepmind/jraph
"""

import jraph
import jax.numpy as jnp

# Define a three node graph, each node has an integer as its feature.
node_features = jnp.array([[0.0], [1.0], [2.0]])

# We will construct a graph for which there is a directed edge between each node
# and its successor. We define this with `senders` (source nodes) and `receivers`
# (destination nodes).
senders = jnp.array([0, 1, 2])
receivers = jnp.array([1, 2, 0])

# You can optionally add edge attributes.
edges = jnp.array([[5.0], [6.0], [7.0]])

# We then save the number of nodes and the number of edges.
# This information is used to make running GNNs over multiple graphs
# in a GraphsTuple possible.
n_node = jnp.array([3])
n_edge = jnp.array([3])

# Optionally you can add `global` information, such as a graph label.

global_context = jnp.array([[1]])
graph = jraph.GraphsTuple(
    nodes=node_features,
    senders=senders,
    receivers=receivers,
    edges=edges,
    n_node=n_node,
    n_edge=n_edge,
    globals=global_context,
)
