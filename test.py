# # -*- coding: utf-8 -*-
# # @Author  : Yue Liu
# # @Email   : yueliu19990731@163.com
# # @Time    : 2021/11/25 11:11
from dgc.rand import setup_seed
from dgc.utils import normalize_adj
from dgc.utils import load_graph_data
from dgc.augmentation import diffusion_adj, add_edge, drop_edge, mask_feat
from dgc.visualization import similarity_plot, t_sne
from dgc.clustering import k_means

# rand
setup_seed(0)

# data loading
X, y, A = load_graph_data(root_path='./', dataset_name="dblp", show_details=True)

# data processing
norm_A = normalize_adj(A, self_loop=True, symmetry=True)

# augmentations
diff_A = diffusion_adj(A)
drop_A = drop_edge(A, 0.2)
add_A = add_edge(A, 0.2)
masked_X = mask_feat(X, 0.2)

# t_sne
similarity_plot(embedding=X, label=y, sample_num=1000, show_fig=True)

# similarity plot
similarity_plot(embedding=X, label=y, sample_num=1000, show_fig=True)

# clustering
acc, nmi, ari, f1, center = k_means(embedding=X, k=max(y), y_true=y, device="cpu")
