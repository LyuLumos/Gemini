# Gemini

## Data Format

```bash
├─── binary_name
└─── raw_graph_list         # list
    ├─── g                  # <DiGraph>
    │   ├─── nodes
    │   ├─── edges
    │   └─── adj 
    ├─── discovre_features
    └─── funcname           
```

e.g.:

```bash
# a raw_graph
{'old_g': <networkx.classes.digraph.DiGraph object at 0x7ff18b657040>, 'discovre_features': [2, 3, 3, 0, 3, 3, 4, 36, 0.0, ['h', 'h'], [4294967295, 4294967295, 3066169, 1, 16]], 'g': <networkx.classes.digraph.DiGraph object at 0x7ff18af3a0e0>, 'funcname': 'check_end'}
# a DiGraph object
DiGraph with 3 nodes and 3 edges
[0, 1, 2]
[(0, 2), (1, 0), (1, 2)]
{0: {2: {}}, 1: {0: {}, 2: {}}, 2: {}}
```