/**
 * @file benchmark.cpp
 * @brief Automated Performance Analysis for OpenMP vs. C++ Std::Threads
 * @course CST435: Parallel Computing
 * * This script automates the execution of both parallel implementations 
 * across varying thread counts (1, 2, 4, 8) and aggregates the results 
 * into a formatted summary table for performance analysis.
 */

#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>  
#include <fstream>  
#include <iomanip>  
#include <algorithm> // Added for trimming

using namespace std;

// Stores the extracted performance metrics from individual program runs
struct RunStats {
    string time = "N/A";
    string count = "0";
};

// Stores the final comparison data for the summary table
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
        // Look for the total time reported by the high_resolution_clock in the main code
        if (line.find("TOTAL TIME:") != string::npos) {
            stats.time = cleanString(line.substr(line.find(":") + 1));
        }
        // Verify the workload consistency by checking processed image count
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
    cout << "   PARALLEL EXECUTION BENCHMARK" << endl;
    cout << "===========================================" << endl;

    /**
     * Compilation Phase
     * -fopenmp: Required for OpenMP directives
     * -phthread: Required for C++ std::thread library
     * -I../include: Links the STB image headers
     */
    cout << "Compiling implementations..." << endl;
    system("g++ ../src_openmp/main.cpp -o ../src_openmp/main_omp -fopenmp -std=c++17 -I../include");
    system("g++ ../src_threads/main.cpp -o ../src_threads/main_thr -pthread -std=c++17 -I../include");

    // Benchmarking loop
    for (int t : threadCounts) {
        cout << "\n>>>> TESTING SCALE: " << t << " THREAD(S) <<<<" << endl;
        
        // Test 1: C++ Std::Threads ---
        cout << "[Test 1] C++ Threads Running...:" << endl;
        RunStats thr = runAndGetStats("../src_threads/main_thr " + to_string(t));
        cout << "Done! (Time: " << thr.time << ", Images: " << thr.count << ")" << endl;

        cout << "\nC++ Threads Implementation:" << endl;
        RunStats omp = runAndGetStats("../src_omp/main_omp " + to_string(t));
        cout << "Done! (Time: " << omp.time << ", Images: " << omp.count << ")" << endl;
        cout << "-------------------------------------------" << endl;

        summaryList.push_back({t, thr.time, omp.time});
    }

    // --- FINAL SUMMARY TABLE ---
    cout << "\n\n===========================================" << endl;
    cout << "          FINAL PERFORMANCE SUMMARY" << endl;
    cout << "===========================================" << endl;
    cout << "+----------+-----------------+-----------------+" << endl;
    cout << "| Threads  | Std::Threads (s)|  OpenMP (S)     |" << endl;
    cout << "+----------+-----------------+-----------------+" << endl;

    for (const auto& s : summaryList) {
        cout << "| " << left << setw(8) << s.threads 
             << " | " << setw(15) << s.thrTime 
             << " | " << setw(15) << s.ompTime << " |" << endl;
    }
    cout << "+----------+-----------------+-----------------+" << endl;
    
    // Cleanup temporary files to leave the environment tidy
    system("rm temp_output.txt");
    return 0;
}