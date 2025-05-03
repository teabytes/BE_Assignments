#include <omp.h>
#include <bits/stdc++.h>
using namespace std;


class Graph {
    int nodes;  // no. of nodes or vertices
    vector<vector<int>> adj;  // adjacency matrix
    vector<bool> visited;  // tracks visited nodes

    public:
    Graph(int n) {
        nodes = n;
        adj.resize(n);
        visited.resize(n, false);
    }

    // adds v to u's list and u to v's
    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);  // undirected graph
    }

    // reset visited[] for new traversal
    void resetVisited() {
        fill(visited.begin(), visited.end(), false);
    }


    void sequentialBFS(int start) {     
        queue<int> q;
        visited[start] = true;
        q.push(start);

        while (!q.empty()) {
            int node = q.front();
            q.pop();
            cout<<node<<" ";

            // traverse all neighbors of the node
            for (int neighbour : adj[node]) {
                if (!visited[neighbour]) {  // if the neighbor hasn't been visited, mark and push
                    visited[neighbour] = true;
                    q.push(neighbour);
                }
            }
        }
    }


    void sequentialDFS(int start) {
        stack<int> s;
        visited[start] = true;
        s.push(start);

        while (!s.empty()) {
            int node = s.top();
            s.pop();
            cout<<node<<" ";

            for (auto it = adj[node].rbegin(); it!=adj[node].rend(); it++) {
                int neighbour = *it;
                if (!visited[neighbour]) {
                    visited[neighbour] = true;
                    s.push(neighbour);  
                }
            }
        }
    }


    void parallelBFS(int start) {
        queue<int> q;
        visited[start] = true;
        q.push(start);

        while (!q.empty()) {
            int levelSize = q.size();
            vector<int> currentLevel;

            for (int i=0; i<levelSize; i++) {
                int node = q.front();
                q.pop();
                cout<<node<<" ";
                currentLevel.push_back(node);
            }

            # pragma omp parallel for  // parallelize the loop for traversing neighbors
            for (int i=0; i<currentLevel.size(); i++) {
                int node = currentLevel[i];
                for (int neighbour : adj[node]) {
                    if (!visited[neighbour]) {
                        #pragma omp critical  // ensure thread-safe access to visited[]
                        {
                            if (!visited[neighbour]) {  // double-check inside critical section
                                visited[neighbour] = true;
                                q.push(neighbour);
                            }
                        }
                    }
                }
            }
        }
    }


    void parallelDFSUtil(int node) {
        if (visited[node]) {
            return;  // skip if already visited
        }

        visited[node] = true;
        cout<<node<<" ";

        #pragma omp parallel for  // parallelize the loop for traversing neighbors
        for (int i=0; i<adj[node].size(); i++) {
            int u = adj[node][i];

            #pragma omp critical  // ensure thread-safe access to visited[]
            {
                if (!visited[u]) {
                    // spawn a new parallel task for the neighbor
                    #pragma omp task
                    parallelDFSUtil(u);
                }
            }
        }
    }


    void parallelDFS(int start) {
        #pragma omp parallel  // parallel section to process each neighbor of the start node
        {
            #pragma omp single  // ensures only one thread starts the root DFS
            {
                parallelDFSUtil(start);
            }
        }
    }
};


int main() {
    int nodes, edges;
    cout<<"\nEnter no. of nodes & edges (N E): ";
    cin>>nodes>>edges;

    Graph g(nodes);
    cout<<"Enter edges (u v): "<<endl;
    for (int i=0; i<edges; i++) {
        int u, v;
        cout << "Edge " << i + 1 << ": ";
        cin >> u >> v;
        g.addEdge(u, v);
    }

    int src;
    cout<<"\nEnter the starting node for BFS and DFS: ";
    cin>>src;

    cout << "\nSequential BFS: ";
    double start = omp_get_wtime();
    g.resetVisited();
    g.sequentialBFS(src);
    double end = omp_get_wtime();
    cout << "\nTime: " << (end - start) * 1000 << " ms\n";

    cout << "\nSequential DFS: ";
    start = omp_get_wtime();
    g.resetVisited();
    g.sequentialDFS(src);
    end = omp_get_wtime();
    cout << "\nTime: " << (end - start) * 1000 << " ms\n";

    cout << "\nParallel BFS: ";
    start = omp_get_wtime();
    g.resetVisited();
    g.parallelBFS(src);
    end = omp_get_wtime();
    cout << "\nTime: " << (end - start) * 1000 << " ms\n";

    cout << "\nParallel DFS: ";
    start = omp_get_wtime();
    g.resetVisited();
    g.parallelDFS(src);
    end = omp_get_wtime();
    cout << "\nTime: " << (end - start) * 1000 << " ms\n";

    return 0;
}