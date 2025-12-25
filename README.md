# CST435 Assignment 2: Parallel Image Processing on Google Cloud Platform (GCP)

## 👥 Team Information
* **Course:** CST435-Parallel and Cloud Computing
* **Group:** Group 46
* **Members:**
  1. Ain Nabihah Binti Mahamad Chah Pari (162321)
  2. Jasmine Binti Mohd Shaiful Adli Chung (164191)
  3. Wan Nurmaisarah Binti Wan Muslim (164323)
  4. Sabrina Binti Sofian (164740)

---

## 📌 Project Overview
This system implements a high-performance image processing pipeline that applies five filters (Grayscale, Blur, Edge Detection, Sharpen, and Brightness) to the Food-101 dataset. The project demonstrates parallel computing scalability by comparing:
* **OpenMP** (Implicit threading using compiler directives)
* **C++ Standard Threads** (Explicit threading using `std::thread`)

---

## 🛠 Installation & Environmental Setup

### Cloud Environment (Google Cloud Platform)

This project was developed and tested on a **Google Cloud Platform (GCP) Compute Engine** instance with the following configuration:

- **VM Instance Name:** `cst435-parallel-vm-assignment2`  
- **Machine Type:** e2-standard-4 (4 vCPUs, 16 GB memory)  
- **Operating System:** Ubuntu 22.04 LTS  
- **Region:** us-central1-c 

> **Note:** All required system dependencies have been pre-installed on the shared GCP VM  
> (`cst435-parallel-vm-assignment2`). 
Group members only need to SSH into the VM to run the code.

#### Navigate to project
 ```bash
 cd /home/shared/CST435-Assignment2
  ```

If you get "Permission denied" error, please check the folder access with:
 ```bash
ls -ld /home/shared/CST435-Assignment2
  ```
  > The output should show at least r-x (read & execute) for others.
  > If not, please contact the repo owner to fix permission. 


#### Pull the latest update from GitHub repository
 ```bash
git pull origin main
  ```

---

## Execution Guide
### Option 1: Benchmark File
The Benchmark Manager automatically compiles both implementations and runs them across 1, 2, 4, and 8 threads to generate a live comparison table.

```bash
cd /home/shared/CST435-Assignment2/benchmark
g++ benchmark.cpp -o manager
./manager 
```

### Option 2: Manual Execution (in folder openmp / threads)
#### OpenMP Implementation: 
```bash
cd /home/shared/CST435-Assignment2/src_openmp
g++ main.cpp -o main -fopenmp -std=c++17 -I../include
./main 4 #(4 threads)
```
#### C++ Threads Implementation:
```bash
cd /home/shared/CST435-Assignment2/src_threads
g++ main.cpp -o main -pthread -std=c++17 -I../include
./main 4 #(4 threads)
```

#### All commands assume the repo is in /home/shared/CST435-Assignment2

---

## 📂 Project Structure
```text
CST435_Assignment2/
├── benchmark/           # Performance Analysis Tools
│   └── benchmark.cpp    # Automation script for multi-thread testing
├── data/                # Input Dataset
│   └── images/          # Food-101 image subsets [cite: 20, 21]
├── include/             # Third-party Libraries
│   ├── stb_image.h      # Image loading library
│   └── stb_image_write.h# Image saving library
├── output/              # Processed Results
│   ├── sample-images/   # Validated samples (IDs: 38795, 63651, 64846)
├── src_openmp/          # OpenMP Implementation
│   └── main.cpp         # Parallel filters using compiler directives
├── src_threads/         # C++ std::thread Implementation
│   └── main.cpp         # Parallel filters using manual thread management
├── .gitignore           # Excludes large binaries and temp files 
└── README.md            # Comprehensive project documentation
