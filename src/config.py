dataset = 'Tmall'
num_node = 43098
dim =256
epoch = 10
activate = 'relu'
batch_size = 512
lr = 0.001
lr_dc = 0.1
lr_dc_step = 3
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
unique = True
add_self_loop = False
mixup = False

# graph_path = "./data/"+dataset+"_edges.csv"

if dataset == "diginetica":
    num_node = 43098
    lb_smooth = 0.4

elif dataset == "gowalla":
    window_size = 4
    lb_smooth = 0.8
    
elif dataset == "lastfm":
    lb_smooth = 0.8

elif dataset == "Nowplaying":
    num_node = 60417
    lb_smooth = 0.4

elif dataset == "Tmall":
    num_node = 40728
    lb_smooth = 0.6

elif dataset == "yoochoose1_4":
    num_node = 37484

elif dataset == "yoochoose1_64":
    num_node = 37484
    lb_smooth = 0.6
