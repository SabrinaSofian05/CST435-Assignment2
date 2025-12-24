#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>  
#include <fstream>  
#include <iomanip>  

using namespace std;

// Structure to hold data from a single run
struct RunStats {
    string time = "N/A";
    string count = "0";
};

// Structure for the final summary table
struct Summary {
    int threads;
    string ompTime;
    string thrTime;
};

RunStats runAndGetStats(string cmd) {
    string fullCmd = cmd + " > temp_output.txt"; 
    int result = system(fullCmd.c_str());
    
    RunStats stats;
    if (result != 0) return stats;

    ifstream file("temp_output.txt");
    string line;
    while (getline(file, line)) {
        // Look for Time
        if (line.find("TOTAL TIME:") != string::npos) {
            stats.time = line.substr(line.find(":") + 1);
            stats.time.erase(0, stats.time.find_first_not_of(" ")); // Trim
        }
        // Look for Image Count
        if (line.find("Images Processed:") != string::npos) {
            stats.count = line.substr(line.find(":") + 1);
            stats.count.erase(0, stats.count.find_first_not_of(" ")); // Trim
        }
    }
    file.close();
    return stats;
}

int main() {
    vector<int> threadCounts = {1, 2, 4, 8};
    vector<Summary> summaryList;

    cout << "===========================================" << endl;
    cout << "   PARALLEL EXECUTION BENCHMARK" << endl;
    cout << "===========================================" << endl;

    // Compile
    system("g++ ../src_openmp/main.cpp -o ../src_openmp/main.exe -fopenmp -std=c++17 -I../include");
    system("g++ ../src_threads/main.cpp -o ../src_threads/main.exe -pthread -std=c++17 -I../include");

    for (int t : threadCounts) {
        cout << "\n>>>> RUNNING WITH " << t << " THREAD(S) <<<<" << endl;
        
        // --- OpenMP Run ---
        cout << "OpenMP Implementation:" << endl;
        RunStats omp = runAndGetStats("..\\src_openmp\\main.exe " + to_string(t));
        cout << "  - Images Processed: " << omp.count << endl;
        cout << "  - Total Time      : " << omp.time << endl;

        // --- Threads Run ---
        cout << "\nC++ Threads Implementation:" << endl;
        RunStats thr = runAndGetStats("..\\src_threads\\main.exe " + to_string(t));
        cout << "  - Images Processed: " << thr.count << endl;
        cout << "  - Total Time      : " << thr.time << endl;
        cout << "-------------------------------------------" << endl;

        summaryList.push_back({t, omp.time, thr.time});
    }

    // --- FINAL SUMMARY TABLE ---
    cout << "\n\n===========================================" << endl;
    cout << "          FINAL PERFORMANCE SUMMARY" << endl;
    cout << "===========================================" << endl;
    cout << "+----------+-----------------+-----------------+" << endl;
    cout << "| Threads  | OpenMP Time     | Threads Time    |" << endl;
    cout << "+----------+-----------------+-----------------+" << endl;

    for (const auto& s : summaryList) {
        cout << "| " << left << setw(8) << s.threads 
             << " | " << setw(15) << s.ompTime 
             << " | " << setw(15) << s.thrTime << " |" << endl;
    }
    cout << "+----------+-----------------+-----------------+" << endl;
    
    system("del temp_output.txt");
    return 0;
}