import torch
import networkx as nx
from torch_geometric.data import Data
import json
import os
import numpy as np
from torch_geometric.utils import to_networkx

# 生成各个数据集中图节点的位置特征
def generate_position_feature(edge_index_file):
  fileList = os.listdir(edge_index_file)
  fileList = [file for file in fileList if '.pt' in file ]
  fileList = ['Kinship-edge_index.pt','UMLS-edge_index.pt','FB15k-237-edge_index.pt', 'WN18RR-edge_index.pt']
  import time

  for file in fileList:
    start_time = time.time()
    
    dataName = file.split('-')[0]
    savePath = edge_index_file + "/" + dataName + "/"
    if not os.path.exists(savePath):
      os.mkdir(savePath)
    edge_index = torch.load(edge_index_file + "/" + file)
    if dataName == 'FB15k':
      x = torch.arange(0, 14541).unsqueeze(1)
    if dataName == 'WN18RR':
      x = torch.arange(0, 40943).unsqueeze(1)
    if dataName == 'Kinship':
      x = torch.arange(0, 104).unsqueeze(1)
    if dataName == 'UMLS':
      x = torch.arange(0, 135).unsqueeze(1)
    data = Data(x, edge_index=edge_index)
    print(data.num_nodes)
    G = to_networkx(data)
    MG_simple = nx.Graph(G)
    
    # 计算图的直径
    try:
        # 获取最大连通子图
        largest_cc = max(nx.connected_components(MG_simple), key=len)
        largest_subgraph = MG_simple.subgraph(largest_cc)
        
        # 计算最大连通子图的直径
        diameter = nx.diameter(largest_subgraph)
        print(f"{dataName} 最大连通子图的直径: {diameter}")
        
        # 输出连通分量的信息
        n_components = nx.number_connected_components(MG_simple)
        print(f"{dataName} 图的连通分量数量: {n_components}")
    except nx.NetworkXError as e:
        print(f"{dataName} 图处理时出错: {str(e)}")
        diameter = -1  # 或者其他适当的默认值
    
    # 度整数torch保存为pt文件
    # int_dg = nx.degree(MG_simple)
    # int_dg = torch.LongTensor([d[1] for d in int_dg])
    # torch.save(int_dg, savePath + dataName + "-int_dg.pt")

    # # 度中心性值
    # dg_json = nx.degree_centrality(MG_simple)
    # dg_json_str = json.dumps(dg_json)
    # with open(savePath + 'degree.json', 'w') as f:
    #     f.write(dg_json_str)
    # print(savePath + "degree.json存储完成")

    
  

    # # 特征向量中心性
    # cen_json = nx.eigenvector_centrality(G)  # gives dictionary of velaue per node
    # cen_json_str = json.dumps(cen_json)
    # with open(savePath + "eigenvector_centrality.json", 'w') as f:
    #   f.write(cen_json_str)
    # print(savePath + "eigenvector_centrality.json存储完成")

    # # 节点的pagerank值
    # pgrank_json = nx.pagerank(MG_simple, alpha=0.85)  # not implemented for muktigraphs
    # pgrank_json_str = json.dumps(pgrank_json)
    # with open(savePath + "pagerank.json", 'w') as f:
    #   f.write(pgrank_json_str)
    # print(savePath + "pagerank.json存储完成")
    
    # # 介数中心性
    # netw_json = nx.betweenness_centrality(MG_simple)
    # netw_json_str = json.dumps(netw_json)
    # with open(savePath + 'betweenness_centrality.json', 'w') as f:
    #     f.write(netw_json_str)
    # print(savePath + "betweenness_centrality.json存储完成")

    # # katz
    # alpha = float(max(nx.adjacency_spectrum(MG_simple)))
    # katz_centrality_json = nx.katz_centrality(MG_simple,1 / alpha - 0.01)
    # katz_centrality_json_str = json.dumps(katz_centrality_json)
    # with open(savePath + 'katz_centrality.json', 'w') as f:
    #     f.write(katz_centrality_json_str)
    # print(savePath + "katz_centrality.json存储完成")

    # 最短路径
    # shortest_paths_json = dict(nx.all_pairs_dijkstra_path_length(MG_simple))
    # shortest_paths_json_str = json.dumps(shortest_paths_json)
    # with open(savePath + "shortest_path.json", 'w') as f:
    #   f.write(shortest_paths_json_str)
    # print(savePath + "shortest_path.json存储完成")

    # 与随机点的最短距离
    # nodes = list(G.nodes)
    # # nodes = [node for node in nodes]
    # number_of_random_nodes_paths = 30   #该值与节点总数有关
    # random_nodes = np.random.choice(nodes, number_of_random_nodes_paths)
    # random_paths_all = []
    # for node1 in nodes:
    #     random_paths = []
    #     for node2 in random_nodes:
    #         path_ = shortest_paths_json.get(node1, -1).get(node2, -1)
    #         random_paths.append(path_)
    #     random_paths_all.append(random_paths)
    # random_paths_all = torch.tensor(random_paths_all)

    # torch.save(random_paths_all, savePath + dataName + "-random_paths.pt")
    # print(savePath + "random_paths.pt存储完成")

    ##################################################################
    # with open(savePath + "degree.json") as f:
    #   dg_json = json.load(f)
    # with open(savePath + "eigenvector_centrality.json") as f:
    #   cen_json = json.load(f)
    # with open(savePath + "pagerank.json") as f:
    #   pgrank_json = json.load(f)
    # with open(savePath + "betweenness_centrality.json") as f:
    #   netw_json = json.load(f)
    # with open(savePath + "katz_centrality.json") as f:
    #   katz_centrality_json = json.load(f)
    # with open(savePath + "shortest_path.json") as f:
    #   shortest_paths_json = json.load(f)
    # print(savePath + "加载完成")
    ##################################################################

    # position_feature = torch.empty(0,5)
    # for node_key in dg_json:
    #   node_feature = torch.tensor([dg_json[node_key],cen_json[node_key],pgrank_json[node_key],netw_json[node_key],katz_centrality_json[node_key]]).view(1,5)
    #   position_feature = torch.cat((position_feature, node_feature), dim=0)
    # torch.save(position_feature, savePath + dataName + "-position_feature.pt")
    # print(savePath + "position_feature.pt存储完成")
    
    # end_time = time.time()
    # processing_time = end_time - start_time
    # print(f"{savePath}处理完毕，耗时: {processing_time:.2f} 秒")

generate_position_feature('./ptFile')