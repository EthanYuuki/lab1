import re
from collections import defaultdict
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
import random

# 读取文件,转换为单词序列
def process_text_file(file_path):
    # 读取文本文件
    with open(file_path, 'r') as file:
        text = file.read()

    # 将文本转换为单词序列
    words_list = re.findall(r'\b[A-Za-z]+\b', text.lower())
    file.close()

    return words_list

# 构建有向图
def generate_directed_graph(words_list):
    graph = defaultdict(dict)
    for i in range(len(words_list) - 1):
        current_word = words_list[i]
        next_word = words_list[i + 1]
        if next_word not in graph[current_word]:
            graph[current_word][next_word] = 1
        else:
            graph[current_word][next_word] += 1

    return graph

# 绘制有向图
def show_directed_graph(graph):
    G = nx.DiGraph()
    
    # 添加节点
    for node in graph:
        G.add_node(node)
    
    # 添加边
    for node, edges in graph.items():
        for neighbor, weight in edges.items():
            G.add_edge(node, neighbor, weight=weight)
    
    # 选择布局
    pos = nx.spring_layout(G)
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue')
    
    # 绘制有向边
    nx.draw_networkx_edges(G, pos, width=3.0, alpha=0.5, edge_color='black', arrows=True)
    
    # 添加节点标签
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
    
    # 添加边权重标签
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    # 显示图形
    plt.axis('off')
    plt.show()

# 查找桥接词
def query_bridge_words(graph, word1, word2):
    word1 = word1.lower()
    word2 = word2.lower()

    if word1 not in graph or word2 not in graph:
        return "No '{}' or '{}' in the graph!\n".format(word1, word2)

    bridge_words = []
    for bridge_word in graph[word1]:
        if word2 in graph[bridge_word]:
            bridge_words.append(bridge_word)

    if not bridge_words:
        return "No bridge words from '{}' to '{}'!\n".format(word1, word2)
    else:
        return "The bridge words from '{}' to '{}' are: '{}'\n".format(word1, word2, ", ".join(bridge_words))

# 根据桥接词生成新文本
def generate_new_text(graph, input_text):
    words = re.findall(r'\b[A-Za-z]+\b', input_text.lower())
    new_text = []

    for i in range(len(words) - 1):
        current_word = words[i]
        next_word = words[i + 1]
        new_text.append(current_word)

        if current_word in graph and next_word in graph:
            bridge_words = []
            for bridge_word in graph[current_word]:
                if next_word in graph[bridge_word]:
                    bridge_words.append(bridge_word)
            if bridge_words:
                selected_bridge_word = random.choice(bridge_words)
                new_text.append(selected_bridge_word)

    new_text.append(words[-1])
    return ' '.join(new_text)

# 计算两个单词之间的最短路径
def calc_shortest_path(graph, start, end):
    start = start.lower()
    end = end.lower()

    if start not in graph or end not in graph:
        return "No path between {} and {}!\n".format(start, end)

    # 使用BFS搜索最短路径
    queue = deque([(start, [start])])
    visited = set()

    while queue:
        current_word, path = queue.popleft()
        visited.add(current_word)

        if current_word == end:
            return "The shortest path from '{}' to '{}' is: {}\n".format(start, end, path)

        for neighbor in graph[current_word]:
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor]))

# 随机游走
def random_walk(graph):
    # 随机选择起始节点
    current_node = random.choice(words_list)
    nodes_visited = [current_node]
    edges_visited = []

    while True:
        # 如果当前节点没有出边
        if current_node not in graph:
            break

        # 获取当前节点的出边对应的结点
        out_edges = graph[current_node]

        if not out_edges:
            break

        # 随机选择下一个节点
        next_node = random.choice(list(out_edges))

        # 将访问过的节点加入已访问节点列表
        nodes_visited.append(next_node)

        # 如果边已经访问过，则结束遍历
        if (current_node, next_node) in edges_visited:
            edges_visited.append((current_node, next_node))
            break

        # 将访问过的边加入已访问节点列表
        edges_visited.append((current_node, next_node))

        # 更新当前节点为下一个节点
        current_node = next_node

    return nodes_visited, edges_visited


# 测试 1
file_path = 'sample.txt'
words_list = process_text_file(file_path)
graph = generate_directed_graph(words_list)
print("\nWords list: {}\n".format(words_list))
print("Graph: {}\n".format(graph))

# 测试 2
show_directed_graph(graph)

# 测试 3
word1 = 'new'
word2 = 'and'
print(query_bridge_words(graph, word1, word2))

# 测试 4
input_text = 'seek to explore new and exciting synergies'
print("Input text: {}".format(input_text))
print("New text: {}\n".format(generate_new_text(graph, input_text)))

# 测试 5
start_word = 'to'
end_word = 'and'
# 输入了两个词
if start_word and end_word: 
    print(calc_shortest_path(graph, start_word, end_word))
# 输入了一个词
elif start_word and not end_word: 
    if start_word not in graph:
        print("No {} in the graph!\n".format(start_word))
    else:
        print("Here are the shortest paths from the word {} to any other word in the graph:\n".format(start_word))
        for target_word in words_list:
            if target_word != start_word:
                print(calc_shortest_path(graph, start_word, target_word))

# 测试 6
# 执行随机游走
visited_nodes, visited_edges = random_walk(graph)
# 输出遍历结果
print("Visited nodes:", visited_nodes)
print("Visited edges:", visited_edges)