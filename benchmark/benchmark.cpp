#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>  
#include <fstream>  
#include <iomanip>  
#include <algorithm> // Added for trimming

using namespace std;

struct RunStats {
    string time = "N/A";
    string count = "0";
};

struct Summary {
    int threads;
    string ompTime;
    string thrTime;
};

// Improved helper to trim whitespace and carriage returns (important for Linux)
string cleanString(string s) {
    s.erase(0, s.find_first_not_of(" \t\r\n"));
    s.erase(s.find_last_not_of(" \t\r\n") + 1);
    return s;
}

RunStats runAndGetStats(string cmd) {
    // 1. Redirect both standard output and errors (2>&1)
    string fullCmd = cmd + " > temp_output.txt 2>&1"; 
    system(fullCmd.c_str());
    
    RunStats stats;
    ifstream file("temp_output.txt");
    string line;
    
    while (getline(file, line)) {
        // Use find for case-insensitive-like matching or exact match
        if (line.find("TOTAL TIME:") != string::npos) {
            stats.time = cleanString(line.substr(line.find(":") + 1));
        }
        if (line.find("Images Processed:") != string::npos) {
            stats.count = cleanString(line.substr(line.find(":") + 1));
        }
    }
    file.close();
    return stats;
}

int main() {
    vector<int> threadCounts = {1, 2, 4, 8};
    vector<Summary> summaryList;

    cout << "===========================================" << endl;
    cout << "   PARALLEL EXECUTION BENCHMARK (LINUX)" << endl;
    cout << "===========================================" << endl;

    // --- CHANGE 1: Compile without .exe extension for Linux ---
    cout << "Compiling implementations..." << endl;
    system("g++ ../src_openmp/main.cpp -o ../src_openmp/main_omp -fopenmp -std=c++17 -I../include");
    system("g++ ../src_threads/main.cpp -o ../src_threads/main_thr -pthread -std=c++17 -I../include");

    for (int t : threadCounts) {
        cout << "\n>>>> RUNNING WITH " << t << " THREAD(S) <<<<" << endl;
        
        // --- CHANGE 2: Use Forward Slashes (/) and remove .exe ---
        // Also added "./" to ensure Linux looks in the correct relative path
        cout << "OpenMP Implementation:" << endl;
        RunStats omp = runAndGetStats("../src_openmp/main_omp " + to_string(t));
        cout << "  - Images Processed: " << omp.count << endl;
        cout << "  - Total Time      : " << omp.time << endl;

        cout << "\nC++ Threads Implementation:" << endl;
        RunStats thr = runAndGetStats("../src_threads/main_thr " + to_string(t));
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
    
    // --- CHANGE 3: Use 'rm' instead of 'del' for Linux ---
    system("rm temp_output.txt");
    return 0;
}