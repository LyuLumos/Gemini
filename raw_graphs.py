

import networkx as nx

class raw_graph:
	def __init__(self, funcname, g, func_f):
		self.funcname = funcname
		self.old_g = g[0]
		self.g = nx.DiGraph()
		self.entry = g[1]
		self.fun_features = func_f
		self.attributing()


class raw_graphs:
	def __init__(self, binary_name):
		self.binary_name = binary_name
		self.raw_graph_list = []

	def append(self, raw_g):
		self.raw_graph_list.append(raw_g)

	def __len__(self):
		return len(self.raw_graph_list)
