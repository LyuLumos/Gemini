import pickle
import numpy as np

filepath = "/workspaces/Gemini/openssl-101f_x86_gcc_O3_openssl.cfg"

class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj

    def read(self, size):
        return self.fileobj.read(size).encode()

    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()


with open(filepath, 'r') as f:
    p = pickle.load(StrToBytes(f))
    print(vars(p.raw_graph_list[0]))
    print(p.raw_graph_list[0].g)
    print(p.raw_graph_list[0].g.nodes)
    print(p.raw_graph_list[0].g.edges)
    print(p.raw_graph_list[0].g.adj)