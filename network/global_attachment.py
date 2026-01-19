#!/usr/bin/env python
# coding: utf-8
import dgl
from dgl import AddEdge, AddReverse

def global_attachment(agent_graph, device, ratio: float):
    '''
        global_attachment - randomly connects different agents globally based on a ratio

        Args: 
            agent_graph: DGLGraph with agent nodes and edges connecting agents
            ratio: ratio of number of new edges to add to total existing edges in graph

        Output:
            Modified agent_graph with new edges introduced 
    '''
    
    # Add edges based on ratio
    agent_graph = AddEdge(ratio=ratio)(agent_graph)

    # Add reverse edges
    agent_graph = AddReverse()(agent_graph)

    # Remove duplicate edges
    # dgl.to_simple works only on device=cpu hence we move the graph to cpu:
    agent_graph = dgl.to_simple(agent_graph.to('cpu'), return_counts='cnt')
    # move the graph back to user choice of device.
    # This is necessary for running on cuda or other hardware.
    agent_graph = agent_graph.to(device)
