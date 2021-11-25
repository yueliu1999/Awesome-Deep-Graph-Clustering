[python-img]: https://img.shields.io/github/languages/top/yueliu1999/Awesome-Deep-Graph-Clustering?color=lightgrey
[stars-img]: https://img.shields.io/github/stars/yueliu1999/Awesome-Deep-Graph-Clustering?color=yellow
[stars-url]: https://github.com/yueliu1999/Awesome-Deep-Graph-Clustering/stargazers
[fork-img]: https://img.shields.io/github/forks/yueliu1999/Awesome-Deep-Graph-Clustering?color=lightblue&label=fork
[fork-url]: https://github.com/yueliu1999/Awesome-Deep-Graph-Clustering/network/members
[visitors-img]: https://visitor-badge.glitch.me/badge?page_id=yueliu1999.Awesome-Deep-Graph-Clustering
[adgc-url]: https://github.com/yueliu1999/Awesome-Deep-Graph-Clustering

# ADGC: Awesome Deep Graph Clustering

ADGC is a collection of state-of-the-art (SOTA), novel deep graph clustering methods (papers, codes and datasets). Any other interesting papers and codes are welcome. Any problems, please contact yueliu19990731@163.com.

[![Made with Python][python-img]][adgc-url]
[![GitHub stars][stars-img]][stars-url]
[![GitHub forks][fork-img]][fork-url]
[![visitors][visitors-img]][adgc-url]

--------------

## What's Deep Graph Clustering?

Deep graph clustering, which aims to reveal the underlying graph structure and divide the nodes into different groups, has attracted intensive attention in recent years.

## Important Survey Papers



## Papers

1. K-Means: "Algorithm AS 136: A k-means clustering algorithm" \[[pdf](http://danida.vnu.edu.vn/cpis/files/Refs/LAD/Algorithm%20AS%20136-%20A%20K-Means%20Clustering%20Algorithm.pdf)|[code]()]
2. DCN (ICML17): "Towards k-means-friendly spaces: Simultaneous deep learning and clustering"  \[[pdf](http://proceedings.mlr.press/v70/yang17b/yang17b.pdf)|[code](https://github.com/boyangumn/DCN)]
3. DEC (ICML16): "Unsupervised Deep Embedding for Clustering Analysis" \[[pdf](http://proceedings.mlr.press/v48/xieb16.pdf)|[code](https://github.com/piiswrong/dec)]
4. IDEC (IJCAI17): "Improved Deep Embedded Clustering with Local Structure Preservation" \[[pdf](https://www.researchgate.net/profile/Xifeng-Guo/publication/317095655_Improved_Deep_Embedded_Clustering_with_Local_Structure_Preservation/links/59263224458515e3d4537edc/Improved-Deep-Embedded-Clustering-with-Local-Structure-Preservation.pdf)|[code](https://github.com/XifengGuo/IDEC)]
5. GAE/VGAE : "Variational Graph Auto-Encoders" \[[pdf](https://arxiv.org/pdf/1611.07308.pdf%5D)|[code](https://github.com/DaehanKim/vgae_pytorch)]
6. DAEGC (IJCAI19): "Attributed Graph Clustering: A Deep Attentional Embedding Approach" \[[pdf](https://www.ijcai.org/proceedings/2019/0509.pdf)|[code](https://github.com/Tiger101010/DAEGC)]
7. ARGA/ARVGA (TCYB19): "Learning Graph Embedding with Adversarial Training Methods" \[[pdf](https://arxiv.org/pdf/1901.01250)|[code](https://github.com/GRAND-Lab/ARGA)]
8. SDCN/SDCN_Q (WWW20): "Structural Deep Clustering Network" \[[pdf](https://arxiv.org/pdf/2002.01633)|[code](https://github.com/bdy9527/SDCN)]
9. DFCN (AAAI21): "Deep Fusion Clustering Network" \[[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/17198/17005)|[code](https://github.com/WxTu/DFCN)]
10. MVGRL (ICML20): "Contrastive Multi-View Representation Learning on Graphs" \[[pdf](http://proceedings.mlr.press/v119/hassani20a/hassani20a.pdf)|[code](https://github.com/kavehhassani/mvgrl)]



## Benchmark Datasets

We divide the datasets into two categories, i.e. graph datasets and non-graph datasets. Graph datasets are some graphs in real-world, such as citation networks, social networks and so on. Non-graph datasets are NOT graph type. However, if necessary, we could construct "adjacency matrices"  by K-Nearest Neighbors (KNN) algorithm.

#### Quick Start

- Step1: Download all datasets from \[[Google Drive](https://drive.google.com/drive/folders/1thSxtAexbvOyjx-bJre8D4OyFKsBe1bK?usp=sharing)|[Baidu Netdisk]()]. Optionally, download some of them from URLs in the tables (Google Drive)

- Step2: Unzip them to **./dataset/**

- Step3: Run the **./dataset/utils.py**

  Two functions **load_graph_data** and **load_data** are provided in **./dataset/utils.py** to load graph datasets and non-graph datasets, respectively.

#### Datasets Details

1. Graph Datasets

   | Dataset  | Samples | Dimension | Edges  | Classes |                             URL                              |
   | :------: | :-----: | :-------: | :----: | :-----: | :----------------------------------------------------------: |
   |   DBLP   |  4057   |    334    |  3528  |    4    | [dblp.zip](https://drive.google.com/file/d/1XWWMIDyvCQ4VJFnAmXS848ksN9MFm5ys/view?usp=sharing) |
   |   CITE   |  3327   |   3703    |  4552  |    6    | [cite.zip](https://drive.google.com/file/d/1U4q84d_n57BquHhUvpLtDzGQ1wzPGF71/view?usp=sharing) |
   |   ACM    |  3025   |   1870    | 13128  |    3    | [acm.zip](https://drive.google.com/file/d/19j7zmQ-AMgzTX7yZoKzUK5wVxQwO5alx/view?usp=sharing) |
   |   AMAP   |  7650   |    745    | 119081 |    8    | [amap.zip](https://drive.google.com/file/d/1qqLWPnBOPkFktHfGMrY9nu8hioyVZV31/view?usp=sharing) |
   |   AMAC   |  13752  |    767    | 245861 |   10    | [amac.zip](https://drive.google.com/file/d/1DJhSOYWXzlRDSTvaC27bSmacTbGq6Ink/view?usp=sharing) |
   |  PUBMED  |  19717  |    500    | 44325  |    3    | [pubmed.zip](https://drive.google.com/file/d/1tdr20dvvjZ9tBHXj8xl6wjO9mQzD0rzA/view?usp=sharing) |
   | CORAFULL |  19793  |   8710    | 63421  |   70    | [corafull.zip](https://drive.google.com/file/d/1XLqs084J3xgWW9jtbBXJOmmY84goT1CE/view?usp=sharing) |
   |   CORA   |  2708   |   1433    |  6632  |    7    | [cora.zip](https://drive.google.com/file/d/1_LesghFTQ02vKOBUfDP8fmDF1JP3MPrJ/view?usp=sharing) |
   | CITESEER |  3327   |   3703    |  6215  |    6    | [citeseer.zip](https://drive.google.com/file/d/1dEsxq5z5dc35tS3E46pg6pc2LUMlF6jF/view?usp=sharing) |

2. Non-graph Datasets

   | Dataset | Samples | Dimension |  Type  | Classes |                             URL                              |
   | :-----: | :-----: | :-------: | :----: | :-----: | :----------------------------------------------------------: |
   |  USPS   |  9298   |    256    | Image  |   10    | [usps.zip](https://drive.google.com/file/d/19oBkSeIluW3A5kcV7W0UM1Bt6V9Q62e-/view?usp=sharing) |
   |  HHAR   |  10299  |    561    | Record |    6    | [hhar.zip](https://drive.google.com/file/d/126OFuNhf2u-g9Tr0wukk0T8uM1cuPzy2/view?usp=sharing) |
   |  REUT   |  10000  |   2000    |  Text  |    4    | [reut.zip](https://drive.google.com/file/d/12MpPWyN87bu-AQYTyjdEcofy1mgjgzi9/view?usp=sharing) |





If you find this repository useful to your research or work, it is really appreciate to star this repository.​ :heart: