# Algorithm Design project - Recursive multi-state network reliability

For more details, check the student paper on the following [link] (https://github.com/teodora338/recursive-multi-state-network-reliability/blob/master/Heuristic_Strategies_for_Accelerating_Recursive_Reliability_Evaluation_of_Multistate_Networks_Using_Minimal_Path_Vectors.pdf).

## Representation of a Weighted Graph through a Vector
Let a weighted graph be given as shown in the figure, representing a flow network.

![Flow netowrk (graph)](https://github.com/teodora338/recursive-multi-state-network-reliability/blob/master/readme_pictures/Picture1.png)

The graph can be represented by a vector whose components are the weights of the edges.
We assign each edge to the corresponding component in the vector.

![Assinging edges to vector component](https://github.com/teodora338/recursive-multi-state-network-reliability/blob/master/readme_pictures/Picture2.png)

Accordingly, the graph can be represented by the vector **(2, 2, 2, 2, 2, 2, 1, 1)**.

---

## Minimal Paths
A minimal path is a vector whose components guarantee connectivity (flow) between the source and the sink. However, if we remove any component, the connectivity is lost.

For example, a minimal path of level 2 (since a flow of 2 units is guaranteed) for the given graph would be the vector **(2, 2, 2, 0, 0, 0, 0, 0)**, while a minimal path of level 1 would be the vector **(2, 1, 2, 0, 0, 0, 0, 0)**.

---

## System Reliability
Let the network/graph represent a system. We say that the system functions if there is flow through the network (from the source to the sink).

Let the edges of the graph have different weights, with each weight occurring with a certain probability. Then, we are dealing with a **multi-state network**.

For example, let the edges of the graph from the figure take weights 0, 1, 2, and 3. Then, we define a probability matrix **P**, where **P[i][j]** represents the probability that the *i-th component/edge* will be in state *j* (i.e., have weight/capacity *j*).

![Component probabilities matrix](https://github.com/teodora338/recursive-multi-state-network-reliability/blob/master/readme_pictures/Picture3.png)

To calculate the probability that the system works (its reliability), we can use the minimal paths of the graph. Specifically, the system works if its vector is greater than or equal to at least one of the minimal paths—that is, the union of the minimal paths.

Let **x** be the vector of the graph and **u1, u2, u3** be the minimal paths.

The probability that the system works – **p** – can be computed as:

```
p = P(x ≥ u1 ∨ x ≥ u2 ∨ x ≥ u3)
```

---

## Iterative Approach
In the iterative approach, the probability is computed using the **principle of inclusion and exclusion**.

Let **x** be the vector of the graph and **u1, u2, u3** be the minimal paths.

According to the principle of inclusion and exclusion:

```
p = P(x ≥ u1) + P(x ≥ u2) + P(x ≥ u3) – P(x ≥ u1 ∧ x ≥ u2) – P(x ≥ u2 ∧ x ≥ u3) – P(x ≥ u1 ∧ x ≥ u3) + P(x ≥ u1 ∧ x ≥ u2 ∧ x ≥ u3)
```

It is important to note that **P(x ≥ u1 ∧ x ≥ u2)** can be represented as follows:

Let **u1 = (1, 3, 5, 7)** and **u2 = (8, 6, 4, 2)**. Then:

```
P(x ≥ u1 ∧ x ≥ u2) = P(x ≥ (8, 6, 5, 7))
```

i.e., the intersection of multiple vectors can be represented by the vector containing the **maximum** component values.

---

## Recursive Approach
This approach uses the **divide-and-conquer technique**.

For example, let **u1, u2, u3, u4** be minimal vectors. Then:

```
p = P(x ≥ u1 ∨ x ≥ u2 ∨ x ≥ u3 ∨ x ≥ u4)
  = P(x ≥ u1 ∨ x ≥ u2) + P(x ≥ u3 ∨ x ≥ u4) – P((x ≥ u1 ∨ x ≥ u2) ∧ (x ≥ u3 ∨ x ≥ u4))
```

That is, the initial set of all minimal vectors **M = {u1, u2, u3, u4}** is split into two subsets:
- **A = {u1, u2}**
- **B = {u3, u4}**

In this way, one probability is computed by calculating three smaller probabilities.

In the code, the splitting of sets is done in four main ways, with additional parameterization:

1. Comparing vectors with representatives of the subsets (the two vectors with maximum distance) and assigning them to the subset of the closer representative.
2. Comparing vectors with every vector in the subsets and assigning them to the subset containing the least distant vector.
3. Clustering using the **K-means algorithm**.
4. Clustering using **K-means** but with centroids chosen manually (the two vectors with maximum distance).

The recursion continues until subsets with 4 or fewer elements remain, at which point the iterative method is applied.

---

## Verification of Computed Probability
To verify the accuracy of the computed probability, examples from published papers are used, where the exact probability has already been calculated (see: `recursive_multi-state_network_reliability_documented_code.py` and `other_code_versions/testing_accuracy_many_test_cases.py`).

---

## Performance Evaluation of the Recursive Function
To compare the speed of the recursive method with other methods, and to compare different splitting strategies, specific codes (`other_code_versions/measuring_codes`) were created to measure:

- Execution time
- Maximum recursion depth reached
- Number of recursive calls
- Average percentage of discarded vectors

---


## Measuring Execution Time and Analyzing Results

For the measuring, the vectors given in the testing code were used.  
It is important to emphasize that these are **minimal paths** (no vector is greater than another vector), but they were obtained by generating random vectors which were then pruned using the function `prune_to_minimal_set()`. In other words, these are not minimal paths from a specific graph.

### Results

The figure bellow shows calculating reliability execution times for different sets and different number of vector components, using different splitting methods (see table in `measuring_execution_time_results.xlsx`).  
The order of the set-splitting methods corresponds to the order in the performance measurement code.

![Execution time results](https://github.com/teodora338/recursive-multi-state-network-reliability/blob/master/readme_pictures/Picture4_results.PNG)

---

### Analysis - Comparison of Set-splitting methods

#### Only Representatives vs. Each Vector in Subsets vs. K-means vs. K-means with Custom Centroids

- **Methods 1 and 2** (comparison with only representatives and comparison with each vector in the subsets)  
  - Produced significantly better results than **Methods 3 and 4** (K-means clustering).  
  - **Method 1 (representatives)**:  
    - Best results in most cases.  
    - Dominated for vectors with fewer components (**6 and 7**) and smaller sets (**< 50 elements**). 
  - **Method 2 (all vectors)**:  
    - In almost same number of cases as Method 1 gave best results.
    - Had similar results to Method 1. 
    - Better for larger vectors (**8, 10, 15 components**) and larger sets (**50–200 elements**).  

- **Method 4 (K-means with custom centroids)**  
  - Centroids chosen as two vectors at maximum distance.  
  - Stronger performance for larger sets (**50–200 elements**) and higher-dimensional vectors (**8, 10, 15 components**).  

- **Method 3 (Standard K-means with automatic centroids)**  
  - Worst results overall.  
  - Possible reason: Repeated clustering (**10 iterations** in this particular code) to achieve better splits and then final clustering selection is based on **minimum SSE** (sum of squared Euclidean distances). Thus, extra computations and comparsions increase execution time.  
  - Performed better on larger both sets and higher-dimensional vectors than on smaller ones.  


#### Comparison of Distance Metrics in Method 1 and 2

- **Euclidean distance** - Best overall (Mehtod 1 and 2).  
- **Manhattan distance** - Solid performance, close to Euclidean.  
- **Chebyshev distance** - Worst results:  
  - Gave similar results to the other distances for smaller sets and fewer components.  
  - Performed significantly worse on larger sets and higher-dimensional vectors.  


#### Comparison of Centroid Initialization Algorithms

- **Random initialization** - Faster and better in most cases.  
- **k-means++ initialization** - Slower due to additional probability calculations for centroid selection.  


#### Comparison of Clustering Algorithms

- **Lloyd’s algorithm ("full")** - Outperformed Elkan’s in most cases.  
- **Elkan’s algorithm** - Expected to be faster (it does fewer distance calculations) but underperformed.  
  - Likely due to dataset-specific characteristics.  

---

## Comparison to other Algorithms

#### RSDP Algorithm

It is worth mentioning that execution time of running the RSDP algorithm on the vectors used in the measuring code was worse than the best time the Recursive algorithm provided in some cases.

More info about the RSDP algorithm on the following [link](https://www.researchgate.net/publication/228975349_An_efficient_method_for_reliability_evaluation_of_multistate_networks_given_all_minimal_path_vectors).
