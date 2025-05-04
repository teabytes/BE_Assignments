/* Implement Min, Max, Sum and Average operations using Parallel Reduction.
Measure the performance of sequential and parallel algorithms. */

#include <omp.h>
#include <bits/stdc++.h>

using namespace std;

class Reduction {
    vector<int> arr;
    int size;

    public:
    Reduction(int n) {
        size = n;
        arr.resize(n);

        srand(time(nullptr));  // seed random number generator with current time
        for (int i=0; i<n; i++) {
            arr[i] = rand() % 100 + 1;
        }
    }

    // sequential min, max, sum & average
    void sequentialOps() {
        int minVal = INT_MAX;
        int maxVal = INT_MIN;
        long long sum = 0;

        double start = omp_get_wtime();

        for (int i=0; i<size; i++) {
            if (arr[i] < minVal)
                minVal = arr[i];
            if (arr[i] > maxVal)
                maxVal = arr[i];
            sum += arr[i];
        }

        double average = sum / size;

        double end = omp_get_wtime();

        cout<<"\nSequential Operations:"<<endl;
        cout<<"Min = "<<minVal<<endl;
        cout<<"Max = "<<maxVal<<endl;
        cout<<"Sum = "<<sum<<endl;
        cout<<"Average = "<<average<<endl;
        cout<<"Time taken: "<<(end-start)*1000<<" milliseconds"<<endl;
    }


    void parallelOps() {
        int minVal = INT_MAX;
        int maxVal = INT_MIN;
        long long sum = 0;

        double start = omp_get_wtime();

        // each thread gets a private copy of the variable used in the reduction
        // and each works on a chunk of the loop iteration space
        #pragma omp parallel for reduction(min:minVal)  // each thread gets a private minVal initialized to INT_MAX.
        for (int i=0; i<size; i++) {
            if (arr[i] < minVal) 
                minVal = arr[i];
        }

        #pragma omp parallel for reduction(max:maxVal)  // each thread gets a private maxVal initialized to INT_MIN.
        for (int i=0; i<size; i++) {
            if (arr[i] > maxVal)
                maxVal = arr[i];
        }

        #pragma omp parallel for reduction(+:sum)  // each thread gets a private sum initialized to 0
        for (int i=0; i<size; i++) {
            sum += arr[i];
        }

        // once threads complete their execution, 
        // OpenMP merges the private copies into a single final result using:
        // minVal = min(thread1_val, thread2_val, ...)
        // maxVal = max(thread1_val, thread2_val, ...)
        // sum = thread1_sum + thread2_sum + ...

        double average = sum / size;

        double end = omp_get_wtime();

        cout<<"\nParallel Operations:"<<endl;
        cout<<"Min = "<<minVal<<endl;
        cout<<"Max = "<<maxVal<<endl;
        cout<<"Sum = "<<sum<<endl;
        cout<<"Average = "<<average<<endl;
        cout<<"Time taken: "<<(end-start)*1000<<" milliseconds"<<endl;
    }
};


int main() {
    int n = 1000000;

    Reduction red(n);

    red.sequentialOps();
    red.parallelOps();
}