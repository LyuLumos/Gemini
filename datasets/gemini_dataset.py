from torch.utils.data import Dataset, DataLoader
import pickle
from .gemini_processing import zero_padded_adjmat, feature_vector, read_cfg, dataset_split
import numpy as np

import sys
sys.path.append("../")
from configs import gemini_config as config


class GeminiDataset(Dataset):
    def __init__(self, type, label):
        assert type in ["train", "test", "valid"], "dataset type error!"
        filepath = config.dataset_dir + type
        self.func_dict = pickle.load(open(filepath, "rb"))
        self.funcname_list = list(self.func_dict.keys())
        self.label = label

    def __len__(self):
        return len(self.funcname_list)

    def __getitem__(self, idx):
        funcname = self.funcname_list[idx]
        func_list = self.func_dict[funcname]
        for i, g in enumerate(func_list):
            g_adjmat = zero_padded_adjmat(g, config.max_nodes)
            g_featmat = feature_vector(g, config.max_nodes)
            if self.label == 'positive':
                g1_index = np.random.randint(0, len(func_list))
                while g1_index == i:
                    g1_index = np.random.randint(0, len(func_list))
                g1 = func_list[g1_index]
                g1_adjmat = zero_padded_adjmat(g1, config.max_nodes)
                g1_featmat = feature_vector(g1, config.max_nodes)
                return g_adjmat, g_featmat, g1_adjmat, g1_featmat, 1
            else:
                index = np.random.randint(0, len(self.funcname_list))
                while self.funcname_list[index] == funcname:
                    index = np.random.randint(0, len(self.funcname_list))
                g2_index = np.random.randint(
                    0, len(self.func_dict[self.funcname_list[index]]))
                g2 = self.func_dict[self.funcname_list[index]][g2_index]
                g2_adjmat = zero_padded_adjmat(g2, config.max_nodes)
                g2_featmat = feature_vector(g2, config.max_nodes)
                return g_adjmat, g_featmat, g2_adjmat, g2_featmat, 0


def dataloader_generate():
    train_dataset = GeminiDataset("train", "positive") + GeminiDataset("train", "negative")
    test_dataset = GeminiDataset("test", "positive") + GeminiDataset("test", "negative")
    valid_dataset = GeminiDataset("valid", "positive") + GeminiDataset("valid", "negative")
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.mini_batch, shuffle=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.mini_batch, shuffle=True)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=config.mini_batch, shuffle=True)
    print(f'train dataset size: {len(train_dataset)}, test dataset size: {len(test_dataset)}, valid dataset size: {len(valid_dataset)}')
    print(f'train dataloader size: {len(train_dataloader)}, test dataloader size: {len(test_dataloader)}, valid dataloader size: {len(valid_dataloader)}')
    return train_dataloader, test_dataloader, valid_dataloader


if __name__ == '__main__':
    all_func_dict = read_cfg()
    dataset_split(all_func_dict)