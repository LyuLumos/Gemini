from train import Gemini
import config
import torch
import numpy as np
from data_processing import StrToBytes, zero_padded_adjmat, feature_vector
import pickle


def load_model():
    model = Gemini()
    model.load_state_dict(torch.load(config.Gemini_model_save_path))
    return model


def read_cfg(cfg_path):
    func_dict = {}
    with open(cfg_path, "r") as f:
        picklefile = pickle.load(StrToBytes(f))
    print(f'function num: {len(picklefile.raw_graph_list)}')
    for func in picklefile.raw_graph_list:
        if len(func.g) < config.min_nodes_threshold:
            continue
        if func_dict.get(func.funcname) == None:
            func_dict[func.funcname] = []
        func_dict[func.funcname].append(func.g)
    return func_dict


def emebdding_generate(model, func_dict):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    new_embedding_dict = {}
    for funcname, graphs in func_dict.items():
        new_embedding_dict[funcname] = []
        for graph in graphs:
            adjmat = zero_padded_adjmat(graph, config.max_nodes)
            featmat = feature_vector(graph, config.max_nodes)
            adjmat = torch.Tensor(adjmat).unsqueeze(0)
            featmat = torch.Tensor(featmat).unsqueeze(0)
            adjmat = adjmat.to(device)
            featmat = featmat.to(device)
            with torch.no_grad():
                _, embedding, _ = model(adjmat, featmat, adjmat, featmat)
            new_embedding_dict[funcname].append(embedding.cpu().numpy())
    return new_embedding_dict


def save_embedding(embedding_dict):
    with open(config.Gemini_embedding_save_path, "wb") as f:
        pickle.dump(embedding_dict, f)
    print("Embedding saved.")


def find_similar(embedding_dict, funcname_embedding):
    sim_scores = []
    for funcname, embeddings in embedding_dict.items():
        for embedding in embeddings:
            sim_scores.append((funcname, cosine_similarity(embedding, funcname_embedding)))
    sim_scores.sort(key=lambda x: x[1], reverse=True)
    return sim_scores



def cosine_similarity(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


if __name__ == "__main__":
    model = load_model()
    func_dict = read_cfg('data/extracted-acfg/openssl-101a_arm_gcc_O3_openssl.cfg')
    embedding_dict = emebdding_generate(model, func_dict)
    save_embedding(embedding_dict)
    sim_score = find_similar(embedding_dict, embedding_dict['EVP_DigestInit_ex'][0])
    print(sim_score[:5])