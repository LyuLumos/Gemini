version = ["openssl-101f"]
arch = ["x86"]
compiler = ["gcc", "clang"]
optimizer = ["O0", "O1", "O2", "O3"]
dir_name = "data/extracted-acfg/"


Gemini_rawdata_dir = "data/extracted-acfg"
Gemini_dataset_dir = "data/Gemini/"
Gemini_feature_size = 9  # （max_constant_1,max_constant_2,num of strings,....）
Gemini_model_save_path = "output/Gemini/model_weight"
Gemini_figure_save_path = "output/Gemini/"
Gemini_embedding_save_path = "output/Gemini/embeddings.pkl"



max_nodes = 500
min_nodes_threshold = 0
Buffer_Size = 1000
mini_batch = 8



learning_rate = 0.0001
epochs = 20
step_per_epoch = 5000
valid_step_pre_epoch = 3000
test_step_pre_epoch = 3000
T = 5
embedding_size = 64
embedding_depth = 2
