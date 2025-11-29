# ADSampling Code Analysis

This document provides a comprehensive analysis of the ADSampling codebase, which implements the algorithms described in the SIGMOD 2023 paper: **"High-Dimensional Approximate Nearest Neighbor Search: with Reliable and Efficient Distance Comparison Operations"**.

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Core Algorithm: ADSampling](#core-algorithm-adsampling)
4. [HNSW Implementation](#hnsw-implementation)
5. [IVF Implementation](#ivf-implementation)
6. [Data Preprocessing](#data-preprocessing)
7. [Build and Usage](#build-and-usage)
8. [Algorithm Variants](#algorithm-variants)

---

## Overview

The ADSampling algorithm addresses a fundamental challenge in high-dimensional approximate nearest neighbor (ANN) search: **Distance Comparison Operations (DCOs)**. Traditional ANN algorithms require comparing distances between vectors, which is expensive in high dimensions. ADSampling provides a more efficient approach by using **adaptive dimensional sampling** combined with **statistical hypothesis testing**.

### Key Insight

Instead of computing the full distance between two high-dimensional vectors, ADSampling:
1. Samples a subset of dimensions
2. Extrapolates the partial distance to estimate the full distance
3. Uses hypothesis testing to determine if a candidate is likely a true nearest neighbor
4. Only computes the full distance when necessary

---

## Repository Structure

```
ADSampling/
├── src/
│   ├── adsampling.h          # Core ADSampling algorithm
│   ├── hnswlib/              # HNSW library with ADSampling integration
│   │   ├── hnswalg.h         # Main HNSW algorithm implementation
│   │   ├── hnswlib.h         # HNSW library interfaces
│   │   ├── space_l2.h        # L2 distance space implementation
│   │   ├── space_ip.h        # Inner product space implementation
│   │   ├── bruteforce.h      # Brute force baseline
│   │   └── visited_list_pool.h # Memory pool for visited lists
│   ├── ivf/
│   │   └── ivf.h             # IVF implementation with ADSampling
│   ├── matrix.h              # Matrix operations and I/O
│   ├── utils.h               # Utility functions (timing, memory)
│   ├── index_hnsw.cpp        # HNSW index building
│   ├── index_ivf.cpp         # IVF index building
│   ├── search_hnsw.cpp       # HNSW search with ADSampling
│   └── search_ivf.cpp        # IVF search with ADSampling
├── data/
│   ├── randomized.py         # Random orthogonal transformation
│   └── ivf.py                # K-means clustering for IVF
├── script/
│   ├── index_hnsw.sh         # HNSW indexing script
│   ├── index_ivf.sh          # IVF indexing script
│   ├── search_hnsw.sh        # HNSW search script
│   └── search_ivf.sh         # IVF search script
└── results/                  # Output directory for search results
```

---

## Core Algorithm: ADSampling

**Location:** `src/adsampling.h`

### Key Parameters

```cpp
namespace adsampling {
    unsigned int D = 960;        // Dataset dimensionality
    float epsilon0 = 2.1;        // Hypothesis testing threshold (recommended: 1.0-4.0)
    unsigned int delta_d = 32;   // Sampling interval (sample every delta_d dimensions)
}
```

### The `ratio()` Function

```cpp
inline float ratio(const int &D, const int &i){
    if(i == D) return 1.0;
    return 1.0 * i / D * (1.0 + epsilon0 / std::sqrt(i)) * (1.0 + epsilon0 / std::sqrt(i));
}
```

This function computes the threshold ratio for hypothesis testing. The key formula is:

**Hypothesis Test:** Check if √(D/d) × dis' > (1 + ε₀/√d) × r

Where:
- `D` = total dimensions
- `d` = sampled dimensions (variable `i` in code)
- `dis'` = partial squared distance computed so far
- `r` = distance threshold (kth nearest neighbor distance)
- `ε₀` = epsilon0 parameter

The function returns: `(d/D) × (1 + ε₀/√d)²`

This allows the equivalent check: `dis' > ratio(D, d) × r`

### The `dist_comp()` Function

```cpp
float dist_comp(const float& dis, const void *data, const void *query, 
                float res = 0, int i = 0)
```

**Parameters:**
- `dis`: Distance threshold (current kth nearest neighbor distance)
- `data`: Pointer to database vector
- `query`: Pointer to query vector
- `res`: Accumulated partial distance (default 0)
- `i`: Starting dimension index (default 0)

**Returns:**
- **Positive value**: Exact distance (candidate is a "positive" object, likely a true NN)
- **Negative value**: Approximate distance (candidate is a "negative" object, pruned early)

**Algorithm Flow:**

```
1. If starting from non-zero dimension (IVF++ case):
   - Immediately test hypothesis with existing partial distance
   - If rejected: return negative approximate distance

2. Main loop (while i < D):
   a. Sample next delta_d dimensions
   b. Compute partial squared distance for these dimensions
   c. Add to accumulated distance
   d. Perform hypothesis test: res >= dis * ratio(D, i)?
      - If YES (reject null hypothesis): 
        Return -res * D / i (negative approximate distance)
      - If NO: continue sampling

3. After sampling all D dimensions:
   Return res (exact distance, positive)
```

**Example:**
```cpp
// For a 960-dimensional dataset with epsilon0=2.1 and delta_d=32:
// - First check at dimension 32
// - Then at 64, 96, 128, ... up to 960
// - Early termination if partial distance extrapolates to > threshold
```

---

## HNSW Implementation

**Location:** `src/hnswlib/hnswalg.h`

HNSW (Hierarchical Navigable Small World) is a graph-based ANN index. This implementation provides three search variants:

### 1. `searchBaseLayerST` - Standard HNSW

```cpp
template <bool has_deletions, bool collect_metrics=false>
std::priority_queue<...> searchBaseLayerST(tableint ep_id, const void *data_point, size_t ef)
```

- Uses **Full Distance Scanning (FDScanning)**
- Computes exact distances for all candidates
- Traditional HNSW approach

### 2. `searchBaseLayerAD` - HNSW+

```cpp
template <bool has_deletions, bool collect_metrics=false>
std::priority_queue<...> searchBaseLayerAD(tableint ep_id, const void *data_point, size_t ef)
```

- Uses **ADSampling** for DCOs
- Distance threshold = N_ef th nearest neighbor distance
- Approximate distances for early pruning

**Key Logic:**
```cpp
// If result set not full: compute exact distance
if (top_candidates.size() < ef) {
    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
    // Add to candidates...
}
// Otherwise: use ADSampling
else {
    dist_t dist = adsampling::dist_comp(lowerBound, getDataByInternalId(candidate_id), data_point, 0, 0);
    if(dist >= 0) {  // Positive object
        // Add to candidates with exact distance
    }
    // Negative objects are pruned
}
```

### 3. `searchBaseLayerADstar` - HNSW++

```cpp
template <bool has_deletions, bool collect_metrics=false>
std::priority_queue<...> searchBaseLayerADstar(tableint ep_id, const void *data_point, size_t ef, size_t k)
```

- Uses **ADSampling** for DCOs
- Distance threshold = **K th nearest neighbor distance** (stricter than HNSW+)
- Maintains separate result sets:
  - `answers`: True KNN set (R1)
  - `top_candidates`: Search candidates (R2)
- **Key innovation**: Uses approximate distances as routing keys for negative objects

**Key Logic:**
```cpp
// Maintain two sets: answers (size k) and top_candidates (size ef)
if (answers.size() < k) {
    // Compute exact distance and add to both sets
}
else {
    dist_t dist = adsampling::dist_comp(lowerBound, currObj1, data_point, 0, 0);
    if(dist >= 0) {  // Positive object
        // Add to both answers and top_candidates
    }
    else {  // Negative object
        // Use APPROXIMATE distance (-dist) for routing in top_candidates
        top_candidates.emplace(-dist, candidate_id);
        candidate_set.emplace(dist, candidate_id);  // Note: dist is negative
    }
}
```

### Search Entry Point: `searchKnn()`

```cpp
std::priority_queue<std::pair<dist_t, labeltype>>
searchKnn(void *query_data, size_t k, int adaptive=0)
```

**Parameters:**
- `query_data`: Query vector
- `k`: Number of nearest neighbors
- `adaptive`: Algorithm selection
  - `0`: Standard HNSW (FDScanning)
  - `1`: HNSW++ (ADSampling with K-based threshold)
  - `2`: HNSW+ (ADSampling with ef-based threshold)

---

## IVF Implementation

**Location:** `src/ivf/ivf.h`

IVF (Inverted File Index) clusters data vectors and searches only relevant clusters.

### Data Layout

```cpp
class IVF {
    size_t N;           // Number of vectors
    size_t D;           // Dimensionality
    size_t C;           // Number of clusters
    size_t d;           // First d dimensions stored separately

    float* L1_data;     // First d dimensions (N × d)
    float* res_data;    // Remaining D-d dimensions (N × (D-d))
    float* centroids;   // Cluster centroids (C × D)

    size_t* start;      // Start index for each cluster
    size_t* len;        // Length of each cluster
    size_t* id;         // Original vector IDs
};
```

### Three IVF Variants

The `adaptive` parameter controls data layout:

| Variant | `adaptive` | `d` value | Description |
|---------|------------|-----------|-------------|
| IVF     | 0          | D         | Full Distance Scanning |
| IVF+    | 2          | 0         | Pure ADSampling |
| IVF++   | 1          | 32        | ADSampling with cache optimization |

### Constructor Logic

```cpp
IVF::IVF(const Matrix<float> &X, const Matrix<float> &_centroids, int adaptive) {
    // ... clustering logic assigns vectors to clusters ...
    // id[i] = original index of i-th vector in clustered order
    
    if(adaptive == 1) d = 32;        // IVF++: optimize cache
    else if(adaptive == 0) d = D;    // IVF: plain scan
    else d = 0;                      // IVF+: plain ADSampling
    
    // Store first d dimensions in L1_data, remaining in res_data
    // Note: vectors are stored in clustered order for better locality
    for(int i=0; i<N; i++) {
        int x = id[i];  // x = original index in X
        for(int j=0; j<D; j++) {
            if(j < d) L1_data[i*d + j] = X.data[x*D + j];
            else res_data[i*(D-d) + j-d] = X.data[x*D + j];
        }
    }
}
```

### Search Algorithm

```cpp
ResultHeap IVF::search(float* query, size_t k, size_t nprobe, float distK)
```

**Algorithm:**

1. **Find closest clusters:** Compute distance to all C centroids, select top nprobe
2. **Scan L1_data:** Compute partial distances for first d dimensions
3. **Complete distance computation:**
   - **IVF (d=D):** Already have full distances, sort and return top-k
   - **IVF+/IVF++ (d<D):** Use ADSampling to complete distance computation

```cpp
// For IVF+/IVF++
if(d < D) {
    for each candidate in selected clusters:
        // dist = partial distance from L1_data
        // res_data starts from dimension d
        float tmp_dist = adsampling::dist_comp(
            distK,                          // Current kth distance threshold
            res_data + can * (D-d),         // Remaining dimensions
            query + d,                      // Query starting from dimension d
            dist[cur],                      // Partial distance
            d                               // Starting dimension
        );
        
        if(tmp_dist > 0) {  // Positive object
            KNNs.emplace(tmp_dist, id[can]);
            // Update distK if necessary
        }
}
```

---

## Data Preprocessing

### Random Orthogonal Transformation

**Location:** `data/randomized.py`

**Purpose:** Apply random orthogonal transformation to spread information across all dimensions, making ADSampling more effective.

```python
def Orthogonal(D):
    G = np.random.randn(D, D).astype('float32')
    Q, _ = np.linalg.qr(G)  # QR decomposition gives orthogonal matrix
    return Q

# Apply transformation: X' = X × P
XP = np.dot(X, P)
```

**Why this helps:**
- Original data may have uneven information distribution across dimensions
- Important features might cluster in certain dimensions
- Orthogonal transformation distributes information uniformly
- Makes early dimension sampling more representative of full distance

### K-means Clustering for IVF

**Location:** `data/ivf.py`

```python
# Using FAISS for efficient clustering
index = faiss.index_factory(D, f"IVF{K},Flat")
index.train(X)
centroids = index.quantizer.reconstruct_n(0, index.nlist)
```

---

## Build and Usage

### Prerequisites

1. **Eigen 3.4.0:** Download from https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
2. Place `Eigen` folder in `./src/`

### Building

The code uses standard C++ compilation. Example:

```bash
# HNSW indexing
g++ -O3 -I./src -o index_hnsw src/index_hnsw.cpp

# HNSW search
g++ -O3 -I./src -DCOUNT_DIMENSION -o search_hnsw src/search_hnsw.cpp
```

### Usage

**Index Building:**
```bash
# HNSW
./index_hnsw -d data_path -i index_path -e ef_construction -m M

# IVF
./index_ivf -d data_path -c centroid_path -i index_path -a adaptive_mode
```

**Search:**
```bash
# HNSW
./search_hnsw -d randomized_mode -i index_path -q query_path -g groundtruth_path \
              -r result_path -t transformation_path -k k -e epsilon0 -p delta_d

# IVF
./search_ivf -d randomized_mode -i index_path -q query_path -g groundtruth_path \
             -r result_path -t transformation_path -k k -e epsilon0 -p delta_d
```

---

## Algorithm Variants

### Summary Table

| Algorithm | Index | DCO Method | Threshold | Cache Optimization |
|-----------|-------|------------|-----------|-------------------|
| HNSW      | HNSW  | FDScanning | N/A       | No |
| HNSW+     | HNSW  | ADSampling | ef-th NN  | No |
| HNSW++    | HNSW  | ADSampling | k-th NN   | Approximate routing |
| IVF       | IVF   | FDScanning | N/A       | No |
| IVF+      | IVF   | ADSampling | k-th NN   | No |
| IVF++     | IVF   | ADSampling | k-th NN   | First 32 dims cached |

### Key Insights

1. **HNSW++ vs HNSW+:** HNSW++ uses a tighter threshold (k-th vs ef-th), enabling more aggressive pruning. It also innovatively uses approximate distances for graph routing.

2. **IVF++ cache optimization:** By storing the first 32 dimensions separately, IVF++ achieves better cache locality and can perform initial partial distance computation efficiently.

3. **Hypothesis testing trade-off:** Smaller ε₀ = more aggressive pruning but higher risk of missing true neighbors. Larger ε₀ = safer but less speedup.

4. **Dimension sampling interval (δd):** Smaller values = finer-grained decisions but more overhead. Larger values = coarser decisions but less overhead.

---

## Performance Metrics

The code tracks several metrics (when `COUNT_DIMENSION` is defined):

```cpp
namespace adsampling {
    long double distance_time = 0;              // Total time in distance computation
    unsigned long long tot_dimension = 0;       // Total dimensions accessed
    unsigned long long tot_dist_calculation = 0; // Total distance calculations
    unsigned long long tot_full_dist = 0;       // Full distance calculations
}
```

Output format:
```
[parameter] [recall%] [time_us_per_query] [total_dimensions_accessed]
```

---

## References

- Paper: "High-Dimensional Approximate Nearest Neighbor Search: with Reliable and Efficient Distance Comparison Operations" (SIGMOD 2023)
- HNSW: Based on https://github.com/nmslib/hnswlib
- Technical report: `technical_report.pdf`
