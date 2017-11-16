"""
QUESTION#3
Given an undirected graph G, find the minimum spanning tree within G. A minimum
spanning tree connects all vertices in a graph with the smallest possible total
weight of edges. Your function should take in and return an adjacency list
structured like this:
{'A': [('B', 2)],
 'B': [('A', 2), ('C', 5)],
 'C': [('B', 5)]}
Vertices are represented as unique strings. The function definition should be question3(G)
"""

def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, parent[i])

def union(parent, rank, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)

    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    else:
        parent[yroot] = xroot
        rank[xroot] += 1

def KruskalMST(graph, V, inv_dict):
    result = []
    i = 0
    e = 0

    #Step1: Sort all edges in non-decreasing order of their weight
    graph = sorted(graph, key = lambda i: i[2])

    parent = [node for node in range(V)]
    rank = [0] * V

    while e < V-1:
        u,v,w = graph[i]
        i += 1
        x = find(parent, u)
        y = find(parent, v)

        if x != y:
            e += 1
            result.append([u,v,w])
            union(parent, rank, x, y)

    final_result = {}

    for u, v, weight in result:
        p1 = [(inv_dict[v], weight)]
        if inv_dict[u] not in final_result:
            final_result[inv_dict[u]] = p1
        else:
            final_result[inv_dict[u]] = final_result[inv_dict[u]].append(p1)

    return final_result

def question3(s1):
    n = len(s1)
    temp_dict = {}
    inv_dict = {}
    count = 0
    #u,v,w = None, None, None
    graph = []

    for i in s1:
        temp_dict[i] = count
        inv_dict[count] = i
        count += 1

    for i in s1:
        for j in s1[i]:
            u,v,w = temp_dict[i], temp_dict[j[0]], j[1]
            graph.append([u,v,w])
    return KruskalMST(graph, count, inv_dict)


def main():
    # Testcase1
    s1 = {'A': [('B', 2)],
          'B': [('A', 4), ('C', 2)],
          'C': [('A', 2), ('B', 5)]}
    print(question3(s1))

    # Testcase2
    s1 = {'A': [('B', 2), ('B', 4)],
          'B': [('A', 4), ('C', 2), ('C', 5)],
          'C': [('A', 2), ('B', 5)]}
    print(question3(s1))

    # Testcase3
    s1 = {'C': [('A', 7)],
          'A': [('C', 4)]}
    print(question3(s1))

    # Testcase4
    s1 = {}
    print(question3(s1))
if __name__ == '__main__':
    main()