import angr
import os
import pickle
import graph as gh
import re


def get_disasm(file_path):
    proj = angr.Project(file_path, auto_load_libs=False)
    cfg = proj.analyses.CFGFast(normalize=True, start_at_entry=False)
    cfg_asm = {}
    for func in cfg.functions.values():
        # BUG: func.size is unreachable
        asm_list = get_str_list(proj, func)
        cfg_asm[func.name] = asm_list
    return cfg_asm


def get_corpus(folder_path):
    corpus = {}
    for file in os.listdir(folder_path):
        if file.endswith('.dll'):
            continue
        abs_file_path = os.path.join(folder_path, file)
        print(f'Processing {abs_file_path}...')
        corpus.update(get_disasm(abs_file_path))

    if not os.path.exists('corpus.pkl'):
        with open('corpus.pkl', 'wb') as f:
            pickle.dump(corpus, f)
    return corpus



def get_str_list(proj, func):
    str_list = []
    sg = gh.to_supergraph(func.transition_graph)
    sg_nodes = sorted(sg.nodes(), key=lambda node: node.addr)
    for node in sg_nodes:
        for block in node.cfg_nodes:
            block = proj.factory.block(block.addr)
            for insn in block.capstone.insns:
                str_list.append(insn.mnemonic + " " + insn.op_str)
    return str_list


def read_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        corpus = pickle.load(f)
    print(f'Loaded {len(corpus)} functions.')

    # show some functions
    # for i, (k, v) in enumerate(corpus.items()):
    #     print(k, v)
    #     if i == 3:
    #         break
 
    small_corpus = {}
    for i, (k, v) in enumerate(corpus.items()):
        small_corpus[k] = v
        if i == 3:
            break
    return corpus, small_corpus


def asmlist2tokens(asmlist) -> list:
    tokens = []
    for asm in asmlist:
        asm = asm.replace('[', '[ ').replace(']', ' ]')
        asm = asm.split(',')

        processed_asm = []
        for i, a in enumerate(asm):
            if '[' in a:
                processed_asm.append(a[:a.index('[')].strip())
                processed_asm.append(a[a.index('['):].strip())
            else:
                processed_asm.append(a.strip())
        
        processed_asm = [re.sub(r'0x[0-9a-fA-F]+', 'mem', a) if '[' in a else re.sub(r'0x[0-9a-fA-F]+', 'imm', a) for a in processed_asm]
        processed_asm = ' '.join(processed_asm).split()
        # processed_asm 仍然保留句信息。

        for a in processed_asm:
            a = a.replace('(', ' ( ').replace(')', ' ) ').replace('+', ' + ').replace('-', ' - ').replace('*', ' * ').replace(':', ' : ')
            tokens.extend(a.split())           
        # tokens.extend(processed_asm)
    return tokens


def data_preprocess(corpus):
    tokens_list = []
    for asm_list in corpus.values():
        tokens_list.extend(asmlist2tokens(asm_list))

    tokens_list = list(set(tokens_list))
    print(f'tokens list: {tokens_list}')
    if not os.path.exists('tokens_list.pkl'):
        with open('tokens_list.pkl', 'wb') as f:
            pickle.dump(tokens_list, f)
    return tokens_list


if __name__ == '__main__':
    # get_corpus('./data/x86')
    corpus, small_corpus = read_pkl('corpus.pkl')
    data_preprocess(corpus)

    
