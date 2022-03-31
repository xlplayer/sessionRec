dataset = 'Tmall'
num_node = 43098
dim =256
epoch = 10
activate = 'relu'
batch_size = 512
lr = 0.001
lr_dc = 0.1
lr_dc_step = 5
l2 = 1e-5
hop = 2
dropout_local = 0
dropout_attn = 0.5
feat_drop = 0.15
alpha = 0.2
weight_decay = 0
order = 1
lb_smooth = 0.4

l = 0.85
window_size = 2

# graph_path = "./data/"+dataset+"_edges.csv"

if dataset == "diginetica":
    num_node = 43098
    dropout_local=0.0

elif dataset == "gowalla":
    feat_drop = 0.5

elif dataset == "lastfm":
    lr_dc_step = 3

elif dataset == "Nowplaying":
    num_node = 60417
    dropout_local = 0.0

elif dataset == "Tmall":
    num_node = 40728
    lb_smooth = 0
    feat_drop = 0.15
    window_size = 4
    # weight_decay = 5e-5
    # dropout_local = 0.5





