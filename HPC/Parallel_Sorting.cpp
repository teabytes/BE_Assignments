/* Write a program to implement Parallel Bubble Sort and Merge sort using OpenMP.
Use existing algorithms and measure the performance of sequential and parallel algorithms. */

#include <omp.h>
#include <bits/stdc++.h>

using namespace std;

class Sorter {
    vector<int> arr;

    public:
    Sorter(vector<int> input) {
        arr = input;
    }

    void reset(vector<int> input) {
        arr = input;
    }

    void printArray() {
        for (int val : arr) {
            cout<<val<<" ";
        }
        cout<<endl;
    }

    void sequentialBubbleSort() {
        int n = arr.size();
        for (int i=0; i<n-1; i++) {
            for (int j=0; j<n-i-1; j++) {
                if (arr[j] > arr[j+1])
                swap(arr[i], arr[j+1]);
            }
        }
    }

    void parallelBubbleSort() {
        int n = arr.size();
        bool sorted = false;

        // odd-even transposition sort pattern
        while (!sorted) {
            bool localSorted = true;

            // even-indexed passes (0,2,4,...)
            #pragma omp parallel for reduction(&&:localSorted)
            for (int i=0; i<n-1; i+=2) {
                if (arr[i] > arr[i+1]) {
                    swap(arr[i], arr[i+1]);
                    localSorted = false;
                }
            }

            // odd-indexed passes (1,3,5,...)
            #pragma omp parallel for reduction(&&:localSorted)
            for (int i=1; i<n-1; i+=2) {
                if (arr[i] > arr[i+1]) {
                    swap(arr[i], arr[i+1]);
                    localSorted = false;
                }
            }

            // stop when no swaps were made in both passes
            sorted = localSorted;
        }
    }

    void sequentialMergeSort(int left, int right) {
        if (left < right) {
            int mid = left + (right - left) / 2;

            sequentialMergeSort(left, mid);
            sequentialMergeSort(mid+1, right);
            
            merge(left, mid, right);
        }        
    }

    void parallelMergeSort(int left, int right, int depth=0) {
        if (left < right) {
            int mid = left + (right - left) / 2;

            // parallelize only until a certain recursion depth to avoid too many threads
            if (depth <= 6) {
                #pragma omp parallel sections
                {
                    #pragma omp section
                    parallelMergeSort(left, mid, depth+1);

                    #pragma omp section
                    parallelMergeSort(mid+1, right, depth+1);
                }
            }
            else {  // beyond depth 4, use sequential recursion
                parallelMergeSort(left, mid, depth+1);
                parallelMergeSort(mid+1, right, depth+1);
            }
        }
    }

    void merge(int left, int mid, int right) {
        vector<int> temp;
        int i = left;
        int j = mid + 1;

        // merge two sorted sub-arrays into one
        while (i<=mid && j<=right) {
            if (arr[i] <= arr[j]) {
                temp.push_back(arr[i++]);
            }
            else {
                temp.push_back(arr[j++]);
            }
        }

        while(i<=mid)
            temp.push_back(arr[i++]);
        while(j<=right) 
            temp.push_back(arr[j++]);

        // copy the sorted result back into original array
        for (int k=0; k<temp.size(); k++) {
            arr[left+k] = temp[k];
        }
    }
};


int main() {
    int n = 100000;  // large array size
    vector<int> input(n);

    // generate a random array
    srand(time(0));  // seeds the generator with current time
    for (int i=0; i<n; i++) {
        input[i] = rand() % 100 + 1;
    }

    Sorter s(input);

    double start = omp_get_wtime();
    s.sequentialBubbleSort();
    double end = omp_get_wtime();
    cout << "\nSequential Bubble Sort Time: " << (end - start) << " secs";

    s.reset(input);
    start = omp_get_wtime();
    s.parallelBubbleSort();
    end = omp_get_wtime();
    cout << "\nParallel Bubble Sort Time: " << (end - start) << " secs";

    s.reset(input);
    start = omp_get_wtime();
    s.sequentialMergeSort(0, n - 1);
    end = omp_get_wtime();
    cout << fixed << setprecision(3);
    cout << "\nSequential Merge Sort Time: " << (end - start) * 1000 << " ms";

    s.reset(input);
    start = omp_get_wtime();
    s.parallelMergeSort(0, n - 1);
    end = omp_get_wtime();
    cout << fixed << setprecision(3);
    cout << "\nParallel Merge Sort Time: " << (end - start) * 1000 << " ms\n";

    return 0;
}