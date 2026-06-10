# -*- coding: utf-8 -*-
import itertools
import time
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


# ---- Interface functions ----

def set_splitting_parameters():
    """User interaction during execution

    Description: The function offers the user selection of: set splitting method, metric (when calling choose_metric()), centroids initialization method (when calling choose_initial_centroids()), clustering algorithm (when calling choose_kmeans_algorithm()).
    ---
    Returns: Selected values in order to split the sets according to requirements (int, float, int, int values, respectively).
    ---
    Called from: main function.
    ---
    Note: In case of invalid input, the function immediately asks for re-entering value (only for the exact same parameter)."""

    # printing possible set splitting methods and instructions for selection
    print("Please select the method you want to use for splitting the sets:\n"
          "Compare only with vector representatives -> insert 1\n"
          "Compare with every vector in subset -> insert 2\n"
          "K-Means clustering -> insert 3\n"
          "K-Means clustering with custom centroids (most distant vectors) -> insert 4")

    method_num = int(input())  # entering set splitting method
    metric_p = 2  # setting predefined value for metric
    centroid = 1  # setting predefined value for centroids initialization
    alg = 1  # setting predefined value for clustering algorithm

    if method_num not in (1, 2, 3, 4):
        print("Wrong input, try again!")
        method_num, metric_p, centroid, alg = set_splitting_parameters()  # calling the function again in case of invalid input
    else:
        if method_num != 3:
            # if the chosen method is NOT "K-Means clustering", then the function for choosing metric is called
            metric_p = choose_metric()
        else:
            # if the chosen method is "K-Means clustering", then the function for selecting centroids initialization method is called
            centroid = choose_initial_centroids()

        if method_num in (3, 4):
            # if the chosen method is "K-Means clustering" or "K-Means clustering with custom centroids", then the function for selecting clustering algorithm is called
            alg = choose_kmeans_algorithm()

    return method_num, metric_p, centroid, alg


def choose_metric():
    """Interaction during execution - selecting a vector distance metric

    Returns: entered metric (float value).
    ---
    Called from: set_splitting_parameters().
    ---
    Note: In case of invalid input, the function immediately asks for re-entering value (only for the exact same parameter)."""

    # printing possible metric values and instructions for selecting
    print("Please select the metric you want to use for comparing vectors:\n"
          "L1 Minkowski / Manhattan / Cityblock norm -> insert 1\n"
          "L2 Minkowski / Euclidean norm -> insert 2\n"
          "Lp Minkowski norm, p > 1 -> insert p\n"
          "L∞ Minkowski / Lmax Minkowski / Chebyshev / Supremum norm -> insert inf\n"
          "Warning: If you chose K-means with custom centroids method (4), this metric will only be used for finding most distant vectors which will be used as centroids. Kmeans() from scikit-learn only uses Euclidean distance for its calculations.")

    p = -1
    inp = str(input())
    if inp == "inf":
        p = np.inf  # setting the metric to Chebyshev distance
    elif inp[0].isalpha():
        print("Invalid value, try again!")
        p = choose_metric()  # calling the function again in case of invalid input
    else:
        p = float(inp)  # setting other distances or invalid input
        if p < 1:
            print("Invalid value, try again!")
            p = choose_metric()  # calling the function again in case of invalid input

    return p


def choose_initial_centroids():
    """Interaction during execution - selection of centroids initialization method

    Returns: value that represents the chosen centroid initialization method (int value)
    ---
    Called from: set_splitting_parameters().
    ---
    Note: In case of invalid input, the function immediately asks for re-entering value (only for the exact same parameter)."""

    # printing possible centroids initialization methods and instructions for selection
    print("Please select method for choosing initial centroids:\n"
          "k-means++ -> insert 1\n"
          "random -> insert 2")

    c = int(input())
    if c not in (1, 2):
        print("Wrong input, try again!")
        c = choose_initial_centroids()  # calling the function again in case of invalid input

    return c


def choose_kmeans_algorithm():
    """Interaction during execution - selection of clustering algorithm

    Returns: value that represents the chosen clustering algorithm (int value).
    ---
    Called from: set_splitting_parameters().
    ---
    Note: In case of invalid input, the function immediately asks for re-entering value (only for the exact same parameter)."""

    # printing possible clustering algorithms and instructions for selection
    print("Please select the K-means clustering algorithm you want to use:\n"
          "Elkan’s algorithm -> insert 1\n"
          "Lloyd’s -> insert 2")

    a = int(input())
    if a not in (1, 2):
        print("Wrong input, try again!")
        a = choose_kmeans_algorithm()  # calling the function again in case of invalid input

    return a


# ---- Utility functions for splitting the sets ----

def find_most_distant_vectors(distances_matrix):
    """Finding a pair of vectors with maximum distance

    Parameters: distances between every pair of vectors (matrix).
    ---
    Returns: indices (from the one-dimensional list that contains all vectors) of the two vectors with maximum distance (int values).
    ---
    Called from: split_into_2_subsets()."""

    # np.argmax transforms a 2D structure into 1D structure and returns index of maximum value
    # np.unravel_index() transforms 1D index into 2D index according to the shape of the matrix
    i, j = np.unravel_index(np.argmax(distances_matrix), distances_matrix.shape)

    return i, j


def compare_distances_only_with_subset_representatives(vectors, distances, index_a, index_b):
    """Method 1 for set splitting - comparing distances only with subset representatives (vectors with maximum distance)

    Parameters: set (list containing vectors), distances between every pair of vectors in the set (matrix) and indices of vectors with maximum distances (int values).
    ---
    Description: The two vectors with maximum distance are called subset representatives. Each of them is added into two different subsets they are representatives of. For each vector that is not representative we compute distances between the vector and the subset representatives. Consequently, the distances to the representatives are compared and the vector is added to the subset whose subset representative is less distant. If a vector is equally distant to both of the representatives, it is added to the subset with fewer elements.
    ---
    Returns: 2 subsets (lists containing vectors).
    ---
    Called from: split_into_2_subsets()."""

    subset_a = [vectors[index_a]]
    subset_b = [vectors[index_b]]

    for i in range(len(vectors)):
        if i != index_a and i != index_b:
            # distances between vector and subset representatives
            distance_a = distances[i][index_a]
            distance_b = distances[i][index_b]

            # comparing distances and adding the vector
            if distance_a == distance_b:
                # if a vector is equally distant to both of the representatives, it is added to the subset with fewer elements
                if len(subset_a) <= len(subset_b):
                    subset_a.append(vectors[i])
                else:
                    subset_b.append(vectors[i])
            elif distance_a < distance_b:
                subset_a.append(vectors[i])
            else:
                subset_b.append(vectors[i])

    return subset_a, subset_b


def compare_with_every_vector_in_subset(vectors, index_a, index_b, metric_parameter_value):
    """Method 2 for set splitting - comparison with each vector from subsets

    Parameters: set (list containing vectors), indices of 2 vectors with maximum distance (int values) and distance metric (float value).
    ---
    Description: The two vectors with maximum distance are called subset representatives. Each of them is added into two different subsets they are representatives of. Then we iterate through vectors that are not vector representatives. For each of them we find minimum distance to vector that is already added to the first subset and minimum distance to vector already added to the second subset. We compare the minimum distances and add the vector to the subset where the less distant vector is placed. If both minimum distances are equal, it is added to the subset with fewer elements.
    ---
    Returns: 2 subsets (lists containing vectors).
    ---
    Called from: split_into_2_subsets()."""

    subset_a = [vectors[index_a]]
    subset_b = [vectors[index_b]]

    for i in range(len(vectors)):
        if i != index_a and i != index_b:
            min_dist_a = min_distance_subset(vectors[i], subset_a, metric_parameter_value)
            min_dist_b = min_distance_subset(vectors[i], subset_b, metric_parameter_value)

            # comparing distances and adding to subset
            if min_dist_a == min_dist_b:
                # if both minimum distances are equal, it is added to the subset with fewer elements
                if len(subset_a) <= len(subset_b):
                    subset_a.append(vectors[i])
                else:
                    subset_b.append(vectors[i])
            elif min_dist_a <= min_dist_b:
                subset_a.append(vectors[i])
            else:
                subset_b.append(vectors[i])

    return subset_a, subset_b


def min_distance_subset(v, subset, metric_param):
    """Finding minimum distance between given vector and a vector in the subset

    Parameters: vector (tuple), subset (list containing vectors) and distance metric (float value).
    ---
    Returns: minimum distance between given vector and a vector in the subset (float value).
    ---
    Called from: compare_with_every_vector_in_subset()"""

    # cdist() calculates Minkowski distances between given vector and all the vectors in the subset
    # np.min() returns the minimum distance
    return np.min(cdist([v], subset, metric='minkowski', p=metric_param))


def kmeans(vectors, centroid_init, kmeans_algorithm):
    """Method 3 for set splitting - K-Means clustering algorithm

    Parameters: set (list containing vectors), centroids initialization method (int value), clustering algorithm (int value).
    ---
    Description: The function uses the received arguments to prepare arguments that will be given to the KMeans() function (that returns object which represents clustering model), in order to cluster the vectors as the user chose with their input. After that kmeans_clusters() returns the clusters.
    ---
    Returns: 2 clusters (lists containing vectors).
    ---
    Called from: split_into_2_subsets().
    ---
    Note: KMeans() from scikit-learn uses only Euclidean distance when measuring distance between vectors

    -----

    General info about K-means algorithm

    * The number of needed clusters MUST be known before using K-means algorithm.
    * Better clustering is clustering that has smaller SSE (Sum of Squared Errors - sum of squared Euclidean distances between each point and the centroid of its assigned cluster).

    -----

    KMeans() function - description

    * It represents a constructor, meaning it returns an object that represents clustering model.
    * It is important to mention that KMeans() from scikit-learn uses only Euclidean distance when measuring distance between vectors.


    Description of parameters:
        a) n_clusters - number of cluster

        b) init - centroids initialization method:
            - 'k-means++':
                1. One of the vectors is randomly chosen as centroid (uniform distribution);
                2. We calculate Euclidean distance between the centroid and the other vectors (let them be denoted by Di and let S = sum((Di)^2) for each i);
                3. For each vector that is not centroid the probability Pi = ((Di)^2 / S ) is calculated;
                4. The next centroid is chosen randomly (the probability for a vector to be chosen is Pi);
                5. We calculate Euclidean distance between centroids and vectors that are not centroids;
                6. We repeat step 3, but now Di is the minimum Euclidean distance to some of the centroids;
                7. We repeat steps 4, 5 and 6 until k centroids are chosen.
            - 'random' - randomly choosing k centroids from the set.
            - custom_centroids - we send centroids chosen by us as argument (Important! - When we send custom centroids we must set n_init=1 (more details in the part where n_init is described)).

        c) algorithm - algorithm that will be used:
            - "full" / Lloyd's algorithm:
                1. Initial centroids are chosen;
                2. Each vector is assigned to the cluster whose centroid is closest (minimum Euclidean distance);
                3. Centroids are updated as the arithmetic mean of all vectors currently assigned to their cluster;
                4. We repeat steps 2 and 3 until:
                    -- Case 1: Centroids remain unchanged in the next iteration;
                    -- Case 2: Difference between centroids from this and the previous iteration is less than or equal to some specified threshold;
                    -- Case 3: Maximum number of iterations is reached (specified before the clustering);
                    -- Case 4: Percentage of vectors that changed their cluster is less than or equal to some specified threshold.
            - "elkan":
                1. Initial centroids are chosen;
                2. Each vector is assigned to the cluster whose centroid is closest (minimum Euclidean distance);
                3. For each vector in the set we calculate upper bound (Ui) (maximum distance at which the vector can remain in the same cluster);
                4. For each vector in the set we calculated lower bound (Lij) (minimum distance at which the vector can NOT be assigned to other cluster);
                5. For each centroid we calculate distance to other centroids, the minimum is chosen, and it is divided by 2 (let this be denoted by Sj);
                6. Using triangle inequality we avoid calculating distances between each vector and each centroid, and we compare bounds to determine if a vector might need reassignment;
                7. Bounds are updated;
                8. Centroids are updated as the arithmetic mean of all vectors currently assigned to their cluster;
                9. Repeat steps 3–8 until one of the stopping conditions from Lloyd’s algorithm is met (no centroid change, small change, max iterations, or few vectors changed clusters).

        d) n_init - number of times K-means runs with different centroid seeds. The best clustering (lowest SSE) is kept. If you provide custom centroids, set n_init=1 so it doesn’t try other initializations."""

    # setting arguments for centroid initialization
    cm = 'k-means++'  # predefined value
    if centroid_init == 2:
        cm = 'random'

    # setting arguments for centroid initialization
    kmeans_algo = 'elkan'  # predefined value
    if kmeans_algorithm == 2:
        kmeans_algo = "full"

    # instancing KMeans object that represents clustering model
    k = KMeans(n_clusters=2, init=cm, algorithm=kmeans_algo, random_state=0)

    set_a, set_b = kmeans_clusters(vectors, k)

    return set_a, set_b


def kmeans_custom_centroids(vectors, index_a, index_b, kmeans_algorithm):
    """Method 4 for set splitting - K-Means clustering algorithm with custom centroids (two vectors with maximum distance)

    Parameters: set (list containing vectors), indices of vectors with maximum distance (int values), clustering algorithm (int value).
    ---
    Description: The function uses the received arguments to prepare arguments that will be given to the KMeans() function (that returns object which represents clustering model), in order to cluster the vectors as the user chose with their input. After that kmeans_clusters() returns the clusters.
    ---
    Returns: 2 clusters (lists containing vectors).
    ---
    Called from: split_into_2_subsets().
    ---
    Note: KMeans() from scikit-learn uses only Euclidean distance when measuring distance between vectors

    -----

    General info about K-means algorithm
    - check documentation for kmeans() function

    -----

    KMeans() function - description
    - check documentation for kmeans() function"""

    # setting arguments for clustering algorithm
    kmeans_algo = 'elkan'  # predefined value
    if kmeans_algorithm == 2:
        kmeans_algo = "full"

    # converting the centroids into np.array (needed format for Kmeans function)
    custom_centroids = np.array([list(vectors[index_a]), list(vectors[index_b])])

    # instancing Kmeans object that represents clustering model
    k = KMeans(n_clusters=2, init=custom_centroids, n_init=1, algorithm=kmeans_algo, random_state=0)

    set_a, set_b = kmeans_clusters(vectors, k)

    return set_a, set_b


def kmeans_clusters(vectors, k):
    """Clustering

    Parameters: set (list containing vectors) and clustering model (object from the class KMeans).
    ---
    Description: Clustering (when calling the function fit(), where as argument we send the set of vectors).
    ---
    Returns: clusters (lists containing vectors).
    ---
    Called from: kmeans() and kmeans_custom_centroids()."""

    # clustering
    k.fit(vectors)

    # labels where the i-th element represents the cluster which is assigned to the i-th vector in the set (the labels are numbers from the set N_0)
    labels = k.labels_

    # adding the vectors to their subsets, according to the clustering
    clusters = [[] for i in range(2)]
    for point, label in zip(vectors, labels):
        clusters[label].append(point)

    return clusters[0], clusters[1]


# ---- Set splitting function ----

def split_into_2_subsets(split_method, vectors, metric_norm, initializing_centroids, kmeans_algorithm):
    """Calling the set splitting functions according to specified criteria

    Parameters: set splitting method (int value), distance metric (float value), centroid initialization method (int value), clustering algorithm (int value).
    ---
    Description: According to the chosen set splitting method we send the specified arguments to the function which splits the set. As a result, we get the subsets.
    ---
    Returns: subsets (lists containing vectors).
    ---
    Called from: recursive_reliability()."""

    if split_method == 1:
        # cdist() calculates distances between every pair of vectors in the sets and returns them as a matrix
        distances = cdist(vectors, vectors, metric="minkowski", p=metric_norm)
        # finding indices of most distant vectors
        index_a, index_b = find_most_distant_vectors(distances)

        # calling set splitting function - comparison only with subset representatives
        a_subset, b_subset = compare_distances_only_with_subset_representatives(vectors, distances, index_a, index_b)
    elif split_method == 2:
        distances = cdist(vectors, vectors, metric="minkowski", p=metric_norm)
        index_a, index_b = find_most_distant_vectors(distances)

        # calling set splitting function - comparison only with every subset vector
        a_subset, b_subset = compare_with_every_vector_in_subset(vectors, index_a, index_b, metric_norm)
    elif split_method == 3:
        # calling set splitting function - K-means clustering algorithm
        a_subset, b_subset = kmeans(vectors, initializing_centroids, kmeans_algorithm)
    else:
        # Note: Two most distant vectors can be found using different metrics, even though KMeans from scikit-learn uses only Euclidean distance in its calculations
        distances = cdist(vectors, vectors, metric="minkowski", p=metric_norm)
        index_a, index_b = find_most_distant_vectors(distances)

        # calling set splitting function - K-means clustering algorithm with custom centroids (two vectors with maximum distance)
        a_subset, b_subset = kmeans_custom_centroids(vectors, index_a, index_b, kmeans_algorithm)

    return a_subset, b_subset


# ---- Utility functions for calculating probability ----

def is_strictly_dominated(v1, v2):
    """Checking whether v2 is strictly greater than v1

    Parameters: first vector (tuple), second vectors (tuple).
    ---
    Description: v2 > v1 if:
        - all components of v2 ≥ corresponding components of v1, and
        - any component of v2 > its corresponding component in v1
    ---
    Returns: Bool value.
    ---
    Called from: prune_to_minimal_set()."""

    return all(a <= b for a, b in zip(v1, v2)) and any(a < b for a, b in zip(v1, v2))


def vector_max(vectors):
    """Finding maximum vector (intersection)

    Parameters: set (list containing vectors).
    ---
    Description: Finds vector with max value in each component
    ---
    Returns: maximum vector (tuple).
    ---
    Called from: recursive_reliability(), iterative_reliability()."""

    if not vectors:
        # if the set is empty it returns an empty set
        return ()

    # zip(*vectors) groups the first components together, the second together, etc.
    # max(values) takes the maximum in each group
    # returns the maximum vector
    return tuple(max(values) for values in zip(*vectors))


def prob_max_vector(vec, component_probabilities):
    """
    Calculating the probability that the graph vector will be greater than or equal to the maximum vector (the event that the graph will be in state S >= max_vector)

    Parameters: maximum vector (tuple), probability matrix.
    ---
    Description: For each component, the probabilities of it being in a state greater than or equal to the maximum are summed. The sum is then multiplied by the product of the previous probabilities (independent events).
    ---
    Returns: probability (float value).
    ---
    Called from: iterative_reliability()"""

    # setting initial probability as 1 (neutral element for calculating product of probabilities)
    prob = 1.0

    # enumerate returns tuples in format (index, value)
    # iterating through components
    for i, state in enumerate(vec):
        # probabilities of the component being in a state greater than or equal to the maximum are summed
        prob_i = sum(component_probabilities[i][state:])
        # multiplying this probability by the product of previous components (independent events)
        prob *= prob_i
    return prob  # returning the product of probabilites


def prune_to_minimal_set(vectors):
    """Finding minimum vectors from set

    Parameters: set (list containing vectors).
    ---
    Description: A vector v is minimal if there is no vector u in the set such that u < v (i.e. u is strictly greater than v). The vectors are sorted (heuristic). Then we iterate through each of the sorted vectors (candidates) and check whether the candidate is greater than any of the current minimal paths. If it is not, we then check whether any of the current minimal paths are greater than the candidate; if so, that path is removed from the set of minimal vectors. The heuristic helps us find the minimal paths faster and compare vectors only with them, instead of with the entire set.
    ---
    Returns: set containing minimum vectors (list containing vectors).
    ---
    Called from: recursive_reliability()."""

    # if the set is empty, empty set is returned
    if not vectors:
        return []

    # Sorting the vectors. This heuristic places at the beginning of the list those vectors that are more likely to be minimal. This improves efficiency.
    # set removes duplicates
    candidates = sorted(list(set(vectors)))

    minimal_set = []  # list that stores the minimum vectors

    # iterating through sorted candidates
    for cand in candidates:
        # candidate cand is dominated if there exists minimum vector such that m < cand
        # is_strictly_dominated(m, cand) returns True if cand > m.
        is_dominated = any(is_strictly_dominated(m, cand) for m in minimal_set)

        if not is_dominated:
            # if the candidate is not dominated (there isn't minimum vector less than the candidate), we remove every minimum vector that is greater than this vector
            minimal_set = [m for m in minimal_set if not is_strictly_dominated(cand, m)]
            # adding the candidate
            minimal_set.append(cand)

    return minimal_set


# ---- Function for iterative calculating reliability ----

def iterative_reliability(mpvs, component_probabilities):
    """Iterative method for calculating probability using the inclusion-exclusion principle

    Parameters: list of vectors (list of tuples), probability matrix (list of lists)
    ---
    Description: Add the probabilities that the vector of the graph is greater than each of the vectors, then subtract the probabilities that the vector of the graph is greater than the intersection of every two vectors, then add the probabilities that it is greater than the intersection of every three vectors, and so on.
    ---
    Returns: the probability that the components of the graph will be in states such that the vector is greater than or equal to the union of the minimal paths of the graph (float value).
    ---
    Called from: recursive_reliability() and the main function."""

    total = 0.0  # initial probability
    n = len(mpvs)  # number of minimal paths (vectors)

    # for each subset size k (from 1 to n), we consider all combinations of minimal paths
    for k in range(1, n + 1):
        # generating all k-element subsets of the minimal paths and iterating through them
        for subset in itertools.combinations(mpvs, k):
            # finding the component-wise maximum vector of this subset
            max_vec = vector_max(list(subset))
            # computing the probability that the graph's vector is greater than or equal to max_vec
            prob = prob_max_vector(max_vec, component_probabilities)
            # add this probability if k is odd, subtract if k is even, this corrects for over-counting intersections of multiple subsets
            sign = (-1) ** (k - 1)
            total += sign * prob

    return total


# ---- Function for recursive calculating reliability ----

# memoization dictionary to store results for specific sets of vectors
reliability_cache = {}

# specifying maximum recursion depth to prevent stack overflow
MAX_RECURSION_DEPTH = 1000

# global list to store recursion depths from every call
LIST = list()


def recursive_reliability(splitting_method, metric_parameter, choosing_centroid, kmeans_algorithm, mpvs,
                          component_probabilities, depth=0):
    """Recursive calculating probability using divide-and-conquer

    Parameters: set splitting method (int value), distance metric (float value), centroids initialization method (int value), clustering algorithm (int value), set (list containing vectors), probability matrix (list of lists), current recursion depth (int value).
    ---
    Decription:
    The function first checks base cases:
      - If maximum recursion depth is exceeded, raises an exception.
      - If the input set is empty, returns probability 0.
      - If the probability for the subset is already computed, returns it from memoization.
      - If the subset has <= 4 elements, computes probability iteratively using inclusion-exclusion.

    If no base case applies, the recursive step follows divide-and-conquer:
      - Divide: split the set into two subsets using the specified method (if K-Means produces an empty cluster, fallback to custom centroids K-means algorithm).
      - Recurse: compute probabilities for each subset and their intersection (component-wise max).
      - Conquer: apply the inclusion-exclusion principle to compute the total probability.
    ---
    Returns: probability that the graph vector is greater than or equal to the union of the minimal paths (float value).
    ---
    Called from: main function."""

    # storing depth in global list
    LIST.append(depth)

    # base cases

    # if the maximum recursion depth is exceeded, raise an exception
    if depth > MAX_RECURSION_DEPTH:
        raise RecursionError("Exceeded maximum allowed recursion depth")

    # if an empty set is passed, return probability 0 (no minimal paths means no flow)
    if not mpvs:
        return 0.0

    # checking whether the probability for the current subset has already been computed (stored in the dictionary)
    # sorting the vectors because dictionary keys are stored in sorted order and converting to tuple because tuples are immutable and hashable
    mpvs_key = tuple(sorted(mpvs))
    if mpvs_key in reliability_cache:
        # if probability has already been computed, return it immediately
        return reliability_cache[mpvs_key]

    # if the subset has <= 4 elements, compute probability using standard inclusion-exclusion (iterative method)
    if len(mpvs) <= 4:
       total = iterative_reliability(mpvs, component_probabilities)
        # storing the computed probability in the dictionary for memoization
       reliability_cache[mpvs_key] = total
       return total

    # recusrive step

    # divide
    # splitting the set into two subsets according to specified parameter values
    A, B = split_into_2_subsets(splitting_method, mpvs, metric_parameter, choosing_centroid, kmeans_algorithm)

    # if the set cannot be split into two non-empty subsets (rare case, can happen with method 3 - K-Means clustering), perform clustering again using method 4 (custom centroids) to ensure both subsets are non-empty
    if not A or not B:
        print("Warning: K-Means produced an empty cluster, falling back to custom centroids.")
        A, B = split_into_2_subsets(4, mpvs, metric_parameter, choosing_centroid, kmeans_algorithm)

    # computing probability for each subset
    pA = recursive_reliability(splitting_method, metric_parameter, choosing_centroid, kmeans_algorithm, A,
                               component_probabilities, depth + 1)
    pB = recursive_reliability(splitting_method, metric_parameter, choosing_centroid, kmeans_algorithm, B,
                               component_probabilities, depth + 1)

    # finding the intersection (component-wise max) of the two subsets
    AB_union_max = [vector_max([a, b]) for a in A for b in B]

    # removing non-minimal vectors from the intersection (if the graph vector is greater than some not minimal vector, then it also greater than the minimal vector, so the event "graph vector > not minimal" is a subset of the event "vector > minimal path")
    AB_pruned = prune_to_minimal_set(AB_union_max)

    # computing probability for the intersection
    pAB = recursive_reliability(splitting_method, metric_parameter, choosing_centroid, kmeans_algorithm, AB_pruned,
                                component_probabilities, depth + 1)

    # conquer
    # computing the probability for the set according to the inclusion-exclusion principle
    result = pA + pB - pAB
    # storing the computed probability in the dictionary
    reliability_cache[mpvs_key] = result
    return result


# ---- Main function for finding maximum recursion depth ----

if __name__ == "__main__":
    # arguments
    values = [
       [1, 1, 1, 1],   # 1 - 1
       [1, 2, 1, 1],   # 1 - 2
       [1, np.inf, 1, 1],   # 1 - 3
        [2, 1, 1, 1],   # 2 - 1
        [2, 2, 1, 1],   # 2 - 2
        [2, np.inf, 1, 1],  # 2 - 3
        [3, 2, 1, 2],   # 3 - а
        [3, 2, 1, 1],   # 3 - б
        [3, 2, 2, 2],   # 3 - в
        [3, 2, 2, 1],   # 3 - г
        [4, 2, 1, 2],   # 4 - а
        [4, 2, 1, 1]    # 4 - б
    ]

    # if we want to read the arguments from input
  #  method, metric, centroid_method, algorithm = set_splitting_parameters()
  #  values = [[method, metric, centroid_method, algorithm]]

    # lists containing minimal vectors

    # vectors with 6 components
    mpv6 = [(3, 0, 1, 1, 1, 3), (1, 0, 2, 2, 2, 2), (1, 0, 2, 3, 3, 0), (2, 3, 1, 1, 3, 0),
            (1, 0, 3, 2, 0, 3), (2, 1, 2, 1, 1, 1), (2, 0, 3, 2, 1, 2), (0, 3, 0, 0, 3, 3),
            (0, 0, 3, 2, 3, 2), (0, 3, 0, 3, 3, 0), (2, 0, 2, 0, 0, 3), (3, 0, 2, 3, 0, 2),
            (0, 2, 2, 1, 2, 0), (3, 2, 0, 2, 1, 0), (2, 0, 3, 3, 0, 1), (1, 0, 0, 3, 3, 3),
            (2, 1, 0, 3, 2, 0), (3, 0, 0, 0, 3, 0), (0, 2, 1, 2, 0, 0), (1, 2, 0, 3, 3, 0),
            (0, 1, 1, 0, 0, 2), (0, 1, 0, 2, 0, 1), (2, 1, 2, 0, 2, 0), (2, 0, 1, 0, 2, 2)]

    # vectors with 7 components
    mpv7 = [(2, 0, 3, 3, 0, 3, 1), (2, 1, 1, 3, 2, 0, 2), (3, 0, 3, 2, 3, 0, 1), (1, 2, 2, 0, 3, 3, 1),
            (0, 1, 1, 0, 0, 2, 2), (0, 3, 0, 3, 3, 0, 3), (3, 3, 0, 3, 1, 2, 2), (0, 2, 1, 2, 0, 0, 1),
            (2, 1, 2, 1, 1, 1, 3), (2, 1, 0, 3, 2, 0, 3), (1, 0, 0, 3, 3, 3, 2), (0, 3, 0, 2, 1, 3, 1),
            (3, 1, 3, 0, 3, 2, 0), (2, 3, 1, 1, 2, 3, 0), (1, 2, 0, 3, 3, 0, 2), (2, 1, 2, 0, 2, 0, 3),
            (0, 2, 2, 1, 2, 0, 0), (3, 3, 0, 3, 2, 0, 0), (3, 3, 3, 0, 0, 3, 1), (0, 2, 3, 0, 3, 3, 1),
            (3, 2, 0, 2, 1, 0, 3), (2, 0, 3, 3, 0, 1, 3), (3, 3, 1, 0, 3, 0, 0), (1, 3, 1, 3, 1, 1, 0),
            (0, 1, 0, 2, 0, 1, 3), (3, 2, 1, 1, 1, 2, 1), (2, 0, 1, 0, 2, 2, 3), (3, 0, 3, 2, 3, 2, 0),
            (2, 2, 2, 0, 1, 2, 1), (2, 1, 3, 2, 1, 1, 2), (3, 2, 0, 1, 3, 3, 2), (1, 0, 3, 2, 0, 3, 3),
            (2, 0, 2, 0, 0, 3, 3), (0, 0, 3, 2, 3, 2, 3), (0, 1, 2, 2, 0, 2, 0), (3, 0, 0, 0, 3, 0, 3),
            (2, 0, 3, 2, 1, 2, 3), (3, 0, 2, 3, 0, 2, 1), (1, 1, 0, 3, 2, 3, 1), (2, 3, 1, 1, 3, 0, 2),
            (0, 3, 0, 0, 3, 3, 3), (2, 1, 2, 2, 2, 0, 2), (1, 0, 2, 3, 3, 0, 2), (1, 0, 2, 2, 2, 2, 2),
            (3, 0, 1, 1, 1, 3, 3)]

    # vectors with 8 components
    mpv8 = [(2, 1, 1, 2, 1, 0, 0, 0), (0, 3, 1, 2, 0, 2, 0, 0), (2, 2, 1, 0, 1, 0, 0, 3), (0, 2, 2, 0, 2, 0, 0, 0),
            (2, 1, 3, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0, 1, 2), (1, 0, 3, 0, 0, 0, 3, 0), (1, 0, 1, 0, 0, 0, 3, 2),
            (0, 0, 1, 0, 0, 0, 2, 3), (2, 1, 2, 2, 0, 0, 0, 0), (0, 2, 0, 3, 1, 0, 0, 0), (0, 3, 0, 1, 0, 3, 0, 2),
            (0, 1, 3, 0, 1, 0, 3, 0), (0, 0, 0, 0, 3, 2, 0, 2), (1, 0, 0, 1, 1, 3, 0, 2), (0, 2, 2, 2, 0, 0, 0, 0),
            (1, 1, 0, 0, 0, 1, 0, 2), (0, 2, 0, 1, 0, 3, 1, 1), (1, 0, 0, 0, 1, 1, 1, 0), (2, 0, 0, 0, 2, 0, 1, 1),
            (0, 1, 3, 0, 0, 2, 2, 1), (0, 0, 1, 0, 2, 0, 2, 0), (0, 0, 0, 1, 0, 0, 3, 1), (1, 3, 0, 3, 0, 1, 0, 0),
            (3, 0, 0, 0, 1, 0, 2, 1), (0, 0, 0, 0, 3, 1, 3, 2), (0, 0, 3, 0, 1, 0, 2, 1), (0, 0, 2, 1, 0, 2, 3, 0),
            (1, 0, 2, 0, 0, 0, 0, 2), (0, 3, 0, 0, 0, 0, 1, 1), (0, 0, 0, 2, 3, 0, 0, 1), (1, 1, 1, 0, 2, 0, 0, 0),
            (0, 2, 1, 3, 0, 0, 0, 0), (0, 0, 0, 1, 1, 0, 1, 0), (0, 0, 0, 0, 0, 3, 3, 1), (0, 3, 1, 1, 0, 1, 3, 0),
            (0, 0, 1, 3, 0, 1, 1, 0), (0, 0, 1, 1, 0, 0, 0, 1), (2, 1, 0, 3, 1, 0, 0, 3), (0, 1, 0, 0, 1, 1, 0, 0),
            (1, 0, 2, 0, 0, 0, 3, 1), (1, 3, 3, 0, 0, 0, 0, 1), (1, 1, 0, 0, 0, 0, 2, 0), (2, 0, 0, 0, 0, 1, 0, 0),
            (0, 0, 1, 0, 1, 1, 0, 0), (0, 3, 2, 0, 1, 0, 0, 2), (0, 1, 0, 3, 0, 1, 0, 1), (0, 3, 1, 0, 2, 0, 1, 0),
            (0, 2, 2, 0, 0, 2, 0, 2), (1, 1, 2, 1, 0, 1, 1, 0), (1, 1, 1, 0, 0, 2, 0, 0), (1, 0, 0, 2, 0, 3, 2, 1),
            (1, 2, 0, 0, 3, 0, 1, 1), (2, 0, 0, 1, 0, 0, 2, 2), (0, 0, 0, 3, 1, 1, 0, 2), (1, 0, 0, 2, 0, 1, 2, 2),
            (1, 0, 1, 2, 0, 3, 3, 0), (2, 0, 1, 0, 1, 0, 1, 0), (0, 0, 0, 1, 2, 1, 0, 1), (1, 2, 0, 2, 0, 0, 0, 2),
            (0, 1, 3, 2, 0, 1, 1, 0), (2, 0, 1, 0, 0, 0, 3, 1), (1, 3, 0, 0, 1, 0, 0, 2)]

    # vectors with 10 components
    mpv10 = [(0, 0, 0, 0, 0, 0, 0, 4, 0, 2), (0, 0, 0, 0, 0, 0, 1, 0, 3, 0), (0, 0, 0, 0, 0, 0, 1, 1, 2, 1),
             (0, 0, 0, 0, 0, 0, 2, 0, 2, 3), (0, 0, 0, 0, 0, 0, 2, 2, 0, 0), (0, 0, 0, 0, 0, 1, 0, 0, 2, 0),
             (0, 0, 0, 0, 0, 2, 1, 3, 0, 1), (0, 0, 0, 0, 0, 2, 3, 0, 1, 4), (0, 0, 0, 0, 0, 3, 2, 0, 0, 2),
             (0, 0, 0, 0, 0, 4, 1, 0, 1, 4), (0, 0, 0, 0, 1, 0, 0, 2, 4, 4), (0, 0, 0, 0, 1, 0, 1, 2, 0, 2),
             (0, 0, 0, 0, 1, 1, 1, 3, 1, 0), (0, 0, 0, 0, 1, 1, 3, 1, 1, 4), (0, 0, 0, 0, 1, 2, 3, 1, 0, 0),
             (0, 0, 0, 0, 1, 3, 0, 0, 0, 4), (0, 0, 0, 0, 1, 3, 0, 3, 0, 0), (0, 0, 0, 0, 1, 4, 1, 0, 1, 0),
             (0, 0, 0, 0, 2, 0, 0, 2, 1, 1), (0, 0, 0, 0, 2, 0, 1, 0, 1, 2), (0, 0, 0, 0, 2, 1, 2, 1, 0, 3),
             (0, 0, 0, 0, 2, 1, 3, 1, 0, 0), (0, 0, 0, 0, 2, 2, 3, 0, 1, 0), (0, 0, 0, 0, 2, 3, 0, 1, 1, 3),
             (0, 0, 0, 0, 2, 3, 1, 0, 0, 2), (0, 0, 0, 0, 2, 4, 0, 1, 0, 1), (0, 0, 0, 0, 3, 0, 2, 0, 0, 4),
             (0, 0, 0, 0, 3, 0, 2, 1, 0, 2), (0, 0, 0, 0, 3, 2, 2, 1, 0, 1), (0, 0, 0, 0, 3, 3, 2, 1, 1, 0),
             (0, 0, 0, 0, 3, 4, 0, 0, 1, 3), (0, 0, 0, 0, 4, 1, 0, 1, 0, 4), (0, 0, 0, 0, 4, 1, 3, 0, 0, 1),
             (0, 0, 0, 0, 4, 2, 2, 0, 0, 3), (0, 0, 0, 0, 4, 3, 0, 2, 0, 1), (0, 0, 0, 0, 4, 4, 0, 2, 0, 0),
             (0, 0, 0, 1, 0, 1, 0, 0, 0, 0), (0, 0, 0, 1, 1, 0, 0, 2, 2, 2), (0, 0, 0, 1, 1, 0, 0, 3, 1, 1),
             (0, 0, 0, 1, 3, 0, 2, 0, 2, 1), (0, 0, 0, 1, 4, 0, 0, 2, 0, 2), (0, 0, 0, 1, 4, 0, 1, 0, 2, 0),
             (0, 0, 0, 2, 0, 0, 0, 3, 2, 2), (0, 0, 0, 2, 1, 0, 3, 0, 1, 4), (0, 0, 0, 2, 3, 0, 3, 0, 0, 0),
             (0, 0, 0, 2, 4, 0, 0, 1, 3, 3), (0, 0, 0, 3, 0, 0, 0, 0, 2, 0), (0, 0, 0, 3, 0, 0, 0, 2, 0, 3),
             (0, 0, 0, 3, 0, 0, 0, 2, 1, 1), (0, 0, 0, 3, 0, 0, 0, 4, 1, 0), (0, 0, 0, 3, 0, 0, 1, 4, 0, 0),
             (0, 0, 0, 3, 0, 0, 3, 0, 1, 3), (0, 0, 0, 4, 0, 0, 3, 1, 1, 2), (0, 0, 0, 4, 1, 0, 1, 0, 1, 4),
             (0, 0, 0, 4, 1, 0, 2, 0, 1, 1), (0, 0, 0, 4, 3, 0, 0, 3, 1, 0), (0, 0, 0, 4, 3, 0, 1, 2, 0, 1),
             (0, 0, 1, 0, 0, 0, 1, 0, 1, 0), (0, 0, 1, 0, 0, 1, 0, 0, 1, 4), (0, 0, 1, 0, 0, 2, 3, 1, 0, 0),
             (0, 0, 1, 0, 1, 1, 1, 3, 0, 1), (0, 0, 1, 0, 2, 2, 0, 1, 1, 2), (0, 0, 1, 0, 2, 3, 0, 0, 1, 0),
             (0, 0, 1, 0, 2, 3, 0, 1, 0, 2), (0, 0, 1, 0, 3, 1, 0, 0, 1, 2), (0, 0, 1, 0, 3, 2, 2, 0, 0, 3),
             (0, 0, 1, 0, 4, 0, 0, 1, 0, 1), (0, 0, 1, 0, 4, 2, 0, 2, 0, 0), (0, 0, 1, 1, 0, 0, 1, 4, 0, 1),
             (0, 0, 1, 1, 0, 0, 3, 0, 0, 1), (0, 0, 1, 1, 3, 0, 1, 1, 0, 1), (0, 0, 1, 2, 3, 0, 0, 1, 1, 2),
             (0, 0, 1, 2, 4, 0, 0, 0, 1, 4), (0, 0, 1, 3, 1, 0, 1, 0, 0, 3), (0, 0, 1, 3, 2, 0, 3, 1, 0, 0),
             (0, 0, 1, 4, 3, 0, 0, 0, 1, 2), (0, 0, 2, 0, 0, 0, 0, 0, 3, 4), (0, 0, 2, 0, 0, 0, 0, 3, 4, 1),
             (0, 0, 2, 0, 0, 0, 3, 1, 0, 3), (0, 0, 2, 0, 0, 1, 0, 1, 0, 3), (0, 0, 2, 0, 0, 2, 0, 0, 0, 3),
             (0, 0, 2, 0, 0, 3, 0, 4, 0, 0), (0, 0, 2, 0, 0, 4, 1, 1, 0, 1), (0, 0, 2, 0, 1, 0, 0, 2, 0, 3),
             (0, 0, 2, 0, 1, 0, 0, 3, 3, 0), (0, 0, 2, 0, 1, 1, 0, 3, 0, 1), (0, 0, 2, 0, 1, 2, 3, 0, 0, 2),
             (0, 0, 2, 0, 2, 0, 1, 0, 0, 0), (0, 0, 2, 0, 3, 2, 0, 1, 0, 1), (0, 0, 2, 1, 0, 0, 0, 0, 0, 2),
             (0, 0, 2, 2, 0, 0, 0, 2, 0, 1), (0, 0, 2, 2, 4, 0, 0, 0, 2, 1), (0, 0, 2, 3, 1, 0, 0, 0, 1, 1),
             (0, 0, 2, 3, 3, 0, 0, 0, 0, 0), (0, 0, 3, 0, 0, 0, 1, 0, 0, 2), (0, 0, 3, 0, 0, 1, 0, 4, 0, 1),
             (0, 0, 3, 0, 0, 3, 0, 1, 1, 1), (0, 0, 3, 0, 0, 4, 0, 3, 0, 2), (0, 0, 3, 0, 1, 0, 0, 0, 0, 1),
             (0, 0, 3, 0, 1, 0, 0, 0, 1, 0), (0, 0, 3, 2, 0, 0, 0, 0, 4, 0), (0, 0, 3, 2, 0, 0, 1, 4, 0, 0),
             (0, 0, 3, 2, 1, 0, 1, 2, 0, 0), (0, 0, 3, 4, 0, 0, 0, 1, 1, 1), (0, 0, 4, 0, 0, 0, 1, 4, 0, 0),
             (0, 0, 4, 0, 0, 4, 2, 1, 0, 0), (0, 0, 4, 0, 1, 2, 1, 1, 0, 0), (0, 0, 4, 0, 4, 1, 0, 1, 0, 0),
             (0, 0, 4, 1, 0, 0, 0, 1, 1, 1), (0, 0, 4, 1, 4, 0, 0, 3, 0, 0), (0, 0, 4, 2, 0, 0, 0, 4, 1, 0),
             (0, 0, 4, 2, 0, 0, 1, 0, 0, 1), (0, 0, 4, 4, 1, 0, 1, 0, 0, 0), (0, 1, 0, 0, 0, 0, 0, 2, 1, 3),
             (0, 1, 0, 0, 0, 0, 0, 2, 4, 2), (0, 1, 0, 0, 0, 0, 2, 0, 0, 0), (0, 1, 0, 0, 1, 0, 0, 1, 3, 4),
             (0, 1, 0, 0, 1, 2, 0, 1, 1, 0), (0, 1, 0, 0, 1, 4, 0, 1, 0, 3), (0, 1, 0, 0, 2, 0, 0, 4, 0, 1),
             (0, 1, 0, 0, 2, 0, 1, 1, 1, 0), (0, 1, 0, 0, 3, 0, 0, 0, 4, 3), (0, 1, 0, 0, 3, 2, 0, 0, 0, 0),
             (0, 1, 0, 1, 0, 0, 0, 0, 4, 0), (0, 1, 0, 1, 0, 0, 0, 2, 1, 0), (0, 1, 0, 1, 1, 0, 1, 0, 2, 0),
             (0, 1, 0, 2, 1, 0, 0, 0, 0, 0), (0, 1, 1, 0, 1, 0, 0, 0, 4, 2), (0, 1, 1, 0, 1, 2, 0, 0, 1, 3),
             (0, 1, 1, 0, 1, 4, 0, 2, 0, 0), (0, 1, 1, 0, 3, 0, 0, 3, 0, 1), (0, 1, 1, 0, 3, 0, 0, 4, 0, 0),
             (0, 1, 1, 0, 4, 0, 0, 0, 4, 1), (0, 1, 1, 1, 1, 0, 0, 1, 1, 2), (0, 1, 2, 0, 0, 2, 1, 0, 0, 0),
             (0, 1, 2, 0, 1, 0, 0, 1, 1, 2), (0, 1, 2, 0, 1, 0, 0, 2, 3, 0), (0, 1, 2, 0, 2, 0, 0, 0, 3, 0),
             (0, 1, 2, 0, 2, 1, 0, 1, 1, 0), (0, 1, 2, 0, 2, 2, 0, 1, 0, 2), (0, 1, 2, 1, 4, 0, 0, 0, 0, 1),
             (0, 1, 2, 2, 0, 0, 0, 2, 0, 0), (0, 1, 3, 0, 0, 0, 0, 1, 1, 0), (0, 1, 3, 0, 0, 1, 0, 1, 0, 1),
             (0, 1, 3, 2, 0, 0, 0, 1, 0, 0), (0, 1, 4, 0, 0, 1, 0, 0, 0, 1), (0, 1, 4, 2, 0, 0, 1, 0, 0, 0),
             (0, 2, 0, 0, 0, 0, 0, 1, 2, 3), (0, 2, 0, 0, 0, 0, 1, 2, 0, 4), (0, 2, 0, 0, 0, 0, 1, 2, 1, 2),
             (0, 2, 0, 0, 0, 1, 0, 1, 0, 4), (0, 2, 0, 0, 0, 1, 1, 1, 0, 0), (0, 2, 0, 0, 0, 2, 0, 3, 1, 1),
             (0, 2, 0, 0, 1, 0, 0, 1, 1, 2), (0, 2, 0, 0, 2, 2, 0, 0, 0, 0), (0, 2, 0, 0, 3, 0, 0, 0, 2, 0),
             (0, 2, 0, 0, 3, 1, 0, 3, 0, 0), (0, 2, 0, 1, 0, 0, 1, 3, 0, 1), (0, 2, 0, 1, 1, 0, 0, 0, 0, 1),
             (0, 2, 0, 1, 1, 0, 1, 0, 1, 0), (0, 2, 0, 1, 2, 0, 1, 3, 0, 0), (0, 2, 1, 0, 0, 0, 0, 0, 4, 3),
             (0, 2, 1, 0, 0, 0, 0, 3, 2, 0), (0, 2, 1, 0, 0, 1, 0, 4, 1, 0), (0, 2, 1, 0, 0, 2, 0, 2, 0, 0),
             (0, 2, 1, 0, 1, 0, 0, 3, 0, 0), (0, 2, 1, 0, 1, 2, 0, 0, 1, 2), (0, 2, 1, 0, 3, 1, 0, 0, 1, 1),
             (0, 2, 1, 0, 4, 1, 0, 2, 1, 0), (0, 2, 1, 1, 0, 0, 0, 1, 3, 1), (0, 2, 1, 2, 0, 0, 0, 4, 0, 0),
             (0, 2, 1, 3, 0, 0, 0, 1, 0, 4), (0, 2, 2, 0, 0, 0, 0, 4, 1, 0), (0, 2, 2, 0, 2, 1, 0, 0, 0, 0),
             (0, 2, 2, 0, 3, 0, 0, 0, 0, 0), (0, 2, 2, 2, 0, 0, 0, 0, 3, 1), (0, 2, 3, 0, 0, 0, 0, 3, 0, 2),
             (0, 2, 4, 0, 0, 0, 0, 1, 0, 3), (0, 2, 4, 1, 0, 0, 0, 2, 0, 1), (0, 2, 4, 1, 0, 0, 1, 1, 0, 0),
             (0, 3, 0, 0, 0, 0, 1, 0, 0, 1), (0, 3, 0, 0, 0, 3, 0, 0, 1, 0), (0, 3, 0, 0, 0, 4, 0, 1, 0, 2),
             (0, 3, 0, 0, 2, 0, 0, 0, 2, 3), (0, 3, 0, 0, 2, 1, 0, 0, 0, 4), (0, 3, 0, 1, 0, 0, 0, 0, 3, 4),
             (0, 3, 0, 1, 3, 0, 1, 1, 0, 0), (0, 3, 1, 0, 0, 0, 0, 1, 0, 1), (0, 3, 1, 0, 0, 0, 0, 1, 1, 0),
             (0, 3, 1, 0, 0, 1, 1, 0, 0, 0), (0, 3, 1, 0, 1, 4, 0, 0, 0, 0), (0, 3, 1, 0, 2, 1, 0, 0, 0, 1),
             (0, 3, 1, 1, 0, 0, 0, 0, 2, 0), (0, 3, 1, 1, 1, 0, 1, 1, 0, 0), (0, 3, 1, 3, 0, 0, 0, 0, 0, 0),
             (0, 3, 2, 0, 0, 1, 0, 0, 0, 0), (0, 3, 3, 1, 0, 0, 0, 1, 0, 0), (0, 3, 3, 1, 1, 0, 1, 0, 0, 0),
             (0, 4, 0, 0, 0, 0, 0, 1, 0, 0), (0, 4, 0, 0, 3, 0, 1, 0, 0, 0), (0, 4, 0, 0, 4, 0, 0, 0, 1, 2),
             (0, 4, 1, 0, 0, 3, 0, 0, 0, 1), (0, 4, 1, 0, 1, 0, 0, 0, 0, 1), (0, 4, 1, 1, 0, 0, 0, 0, 0, 2),
             (0, 4, 1, 1, 3, 0, 0, 0, 0, 0), (0, 4, 3, 0, 0, 0, 0, 0, 3, 1), (1, 0, 0, 0, 0, 0, 1, 4, 1, 0),
             (1, 0, 0, 0, 0, 0, 3, 0, 1, 4), (1, 0, 0, 0, 0, 1, 1, 3, 1, 3), (1, 0, 0, 0, 0, 1, 2, 0, 1, 1),
             (1, 0, 0, 0, 0, 2, 1, 2, 0, 3), (1, 0, 0, 0, 1, 0, 0, 0, 1, 2), (1, 0, 0, 0, 1, 0, 0, 2, 1, 1),
             (1, 0, 0, 0, 1, 0, 2, 1, 1, 1), (1, 0, 0, 0, 1, 0, 3, 1, 0, 2), (1, 0, 0, 0, 1, 1, 0, 1, 0, 0),
             (1, 0, 0, 0, 2, 0, 1, 0, 0, 0), (1, 0, 0, 0, 3, 1, 0, 0, 0, 3), (1, 0, 0, 0, 4, 0, 0, 2, 0, 4),
             (1, 0, 0, 1, 4, 0, 0, 3, 4, 0), (1, 0, 0, 2, 0, 0, 0, 1, 4, 1), (1, 0, 0, 2, 0, 0, 1, 3, 0, 0),
             (1, 0, 0, 2, 0, 0, 2, 0, 1, 1), (1, 0, 0, 2, 4, 0, 0, 0, 1, 0), (1, 0, 0, 3, 1, 0, 1, 0, 0, 0),
             (1, 0, 0, 4, 0, 0, 1, 0, 1, 0), (1, 0, 0, 4, 3, 0, 0, 2, 1, 0), (1, 0, 0, 4, 4, 0, 0, 1, 0, 4),
             (1, 0, 1, 0, 0, 1, 0, 3, 0, 2), (1, 0, 1, 0, 0, 1, 1, 2, 0, 3), (1, 0, 1, 0, 0, 2, 1, 1, 0, 1),
             (1, 0, 1, 0, 0, 2, 1, 4, 0, 0), (1, 0, 1, 0, 2, 0, 0, 0, 0, 3), (1, 0, 1, 0, 2, 1, 0, 0, 0, 2),
             (1, 0, 1, 0, 3, 0, 0, 1, 3, 0), (1, 0, 1, 0, 4, 0, 0, 1, 0, 0), (1, 0, 1, 1, 0, 0, 0, 2, 2, 4),
             (1, 0, 1, 1, 0, 0, 0, 4, 3, 1), (1, 0, 1, 1, 2, 0, 0, 1, 2, 0), (1, 0, 1, 1, 3, 0, 0, 3, 0, 1),
             (1, 0, 1, 1, 4, 0, 0, 0, 2, 1), (1, 0, 1, 2, 0, 0, 2, 1, 0, 2), (1, 0, 1, 2, 2, 0, 0, 0, 1, 1),
             (1, 0, 1, 3, 3, 0, 0, 2, 0, 1), (1, 0, 1, 4, 0, 0, 0, 1, 1, 1), (1, 0, 2, 0, 0, 1, 0, 0, 0, 0),
             (1, 0, 2, 0, 1, 0, 1, 4, 0, 1), (1, 0, 2, 0, 3, 0, 0, 3, 0, 1), (1, 0, 2, 1, 0, 0, 0, 3, 3, 1),
             (1, 0, 2, 2, 0, 0, 0, 1, 1, 0), (1, 0, 2, 2, 0, 0, 4, 0, 0, 0), (1, 0, 2, 3, 2, 0, 0, 0, 0, 0),
             (1, 0, 3, 0, 0, 0, 0, 0, 3, 3), (1, 0, 3, 0, 0, 0, 0, 2, 2, 0), (1, 0, 3, 0, 0, 0, 0, 4, 1, 1),
             (1, 0, 3, 2, 0, 0, 0, 0, 0, 1), (1, 0, 4, 1, 1, 0, 0, 1, 0, 0), (1, 0, 4, 3, 0, 0, 1, 1, 0, 0),
             (1, 1, 0, 0, 0, 1, 0, 3, 1, 1), (1, 1, 0, 0, 0, 1, 1, 1, 0, 1), (1, 1, 0, 0, 0, 2, 0, 1, 0, 1),
             (1, 1, 0, 0, 1, 0, 0, 2, 3, 0), (1, 1, 0, 0, 1, 2, 0, 0, 0, 3), (1, 1, 0, 0, 1, 3, 0, 0, 1, 1),
             (1, 1, 0, 0, 3, 0, 0, 3, 1, 0), (1, 1, 0, 2, 0, 0, 1, 2, 0, 3), (1, 1, 1, 0, 0, 2, 0, 0, 0, 3),
             (1, 1, 1, 0, 0, 3, 0, 0, 1, 2), (1, 1, 1, 0, 0, 3, 1, 0, 0, 2), (1, 1, 1, 0, 1, 0, 1, 0, 0, 1),
             (1, 1, 2, 0, 0, 0, 0, 2, 1, 1), (1, 1, 2, 0, 2, 0, 0, 2, 1, 0), (1, 1, 2, 1, 3, 0, 0, 2, 0, 0),
             (1, 1, 3, 1, 2, 0, 0, 1, 0, 0), (1, 2, 0, 0, 0, 0, 0, 2, 0, 0), (1, 2, 0, 0, 0, 1, 1, 0, 1, 0),
             (1, 2, 0, 0, 0, 2, 0, 0, 1, 0), (1, 2, 0, 0, 1, 0, 0, 1, 1, 1), (1, 2, 0, 0, 2, 0, 0, 1, 0, 1),
             (1, 2, 0, 1, 0, 0, 0, 0, 0, 4), (1, 2, 0, 1, 0, 0, 0, 0, 1, 0), (1, 2, 1, 0, 1, 1, 0, 0, 0, 0),
             (1, 2, 1, 1, 0, 0, 0, 1, 0, 0), (1, 2, 2, 0, 0, 0, 0, 1, 1, 1), (1, 2, 2, 3, 0, 0, 1, 0, 0, 0),
             (1, 2, 3, 0, 0, 0, 0, 0, 3, 0), (1, 3, 0, 0, 0, 0, 0, 1, 1, 4), (1, 3, 0, 0, 0, 1, 0, 0, 1, 4),
             (1, 3, 0, 0, 2, 0, 0, 0, 3, 0), (1, 3, 1, 0, 1, 0, 0, 0, 2, 1), (1, 3, 4, 0, 0, 0, 0, 0, 1, 0),
             (1, 3, 4, 0, 0, 0, 0, 1, 0, 0), (1, 4, 0, 0, 4, 1, 0, 0, 0, 1), (2, 0, 0, 0, 0, 0, 0, 0, 4, 1),
             (2, 0, 0, 0, 0, 0, 0, 2, 3, 2), (2, 0, 0, 0, 0, 0, 1, 0, 0, 4), (2, 0, 0, 0, 0, 0, 4, 1, 1, 0),
             (2, 0, 0, 0, 0, 1, 1, 0, 0, 0), (2, 0, 0, 0, 0, 3, 0, 3, 1, 2), (2, 0, 0, 0, 1, 0, 0, 2, 0, 2),
             (2, 0, 0, 1, 0, 0, 3, 0, 2, 1), (2, 0, 0, 1, 1, 0, 0, 3, 1, 0), (2, 0, 0, 1, 1, 0, 3, 0, 1, 1),
             (2, 0, 0, 1, 4, 0, 0, 1, 4, 0), (2, 0, 0, 2, 0, 0, 1, 2, 0, 3), (2, 0, 0, 2, 0, 0, 2, 0, 0, 1),
             (2, 0, 0, 2, 0, 0, 2, 1, 2, 0), (2, 0, 0, 2, 1, 0, 0, 4, 0, 0), (2, 0, 0, 2, 2, 0, 0, 0, 3, 1),
             (2, 0, 0, 3, 3, 0, 0, 0, 1, 1), (2, 0, 0, 4, 2, 0, 0, 0, 0, 2), (2, 0, 1, 0, 0, 0, 0, 0, 0, 0),
             (2, 1, 0, 0, 0, 0, 0, 2, 0, 0), (2, 1, 0, 0, 0, 1, 0, 1, 1, 3), (2, 1, 0, 0, 1, 2, 0, 0, 1, 0),
             (2, 1, 0, 0, 2, 0, 0, 1, 3, 1), (2, 1, 0, 1, 0, 0, 0, 0, 2, 3), (2, 1, 0, 2, 0, 0, 0, 0, 1, 3),
             (2, 1, 0, 3, 0, 0, 0, 0, 1, 0), (2, 1, 0, 3, 0, 0, 1, 0, 0, 2), (2, 3, 0, 0, 2, 1, 0, 0, 1, 1),
             (2, 4, 0, 0, 0, 0, 0, 0, 0, 2), (3, 0, 0, 0, 0, 0, 1, 2, 1, 2), (3, 0, 0, 0, 0, 3, 0, 1, 0, 4),
             (3, 0, 0, 0, 0, 3, 0, 4, 0, 1), (3, 0, 0, 0, 1, 0, 1, 1, 2, 0), (3, 0, 0, 0, 2, 0, 0, 4, 3, 0),
             (3, 0, 0, 0, 2, 3, 0, 0, 0, 3), (3, 0, 0, 0, 4, 0, 0, 4, 0, 0), (3, 0, 0, 1, 1, 0, 2, 0, 0, 3),
             (3, 0, 0, 1, 4, 0, 0, 2, 2, 0), (3, 0, 0, 3, 0, 0, 0, 1, 1, 0), (3, 0, 0, 3, 0, 0, 1, 0, 0, 3),
             (3, 0, 0, 4, 0, 0, 0, 0, 1, 1), (3, 0, 0, 4, 3, 0, 0, 3, 0, 1), (3, 1, 0, 0, 0, 2, 0, 1, 0, 0),
             (3, 1, 0, 1, 0, 0, 0, 0, 0, 0), (3, 2, 0, 0, 0, 0, 0, 1, 2, 0), (3, 2, 0, 0, 0, 1, 0, 1, 0, 1),
             (3, 2, 0, 0, 1, 0, 0, 0, 3, 1), (3, 2, 0, 0, 2, 1, 0, 0, 0, 2), (3, 3, 0, 0, 2, 0, 0, 0, 0, 2),
             (3, 4, 0, 0, 1, 0, 0, 0, 1, 0), (4, 0, 0, 0, 0, 0, 0, 3, 2, 3), (4, 0, 0, 0, 3, 0, 0, 0, 3, 0),
             (4, 0, 0, 0, 3, 0, 0, 2, 2, 0), (4, 0, 0, 1, 0, 0, 0, 1, 0, 3), (4, 0, 0, 1, 0, 0, 1, 1, 1, 2),
             (4, 0, 0, 1, 1, 0, 1, 1, 0, 1), (4, 0, 0, 1, 2, 0, 0, 0, 3, 0), (4, 0, 0, 2, 0, 0, 0, 1, 2, 1),
             (4, 1, 0, 0, 1, 0, 0, 0, 4, 0), (4, 2, 0, 0, 0, 0, 0, 0, 1, 1), (4, 2, 0, 0, 1, 1, 0, 0, 0, 1),
             (4, 3, 0, 0, 3, 0, 0, 0, 1, 0)]

    # vectors with 15 components
    mpv15 = [(0, 0, 0, 3, 2, 3, 2, 2, 3, 3, 0, 3, 0, 1, 3), (0, 0, 1, 2, 2, 3, 2, 2, 2, 3, 3, 3, 0, 0, 3),
               (0, 0, 1, 3, 3, 3, 2, 2, 3, 3, 2, 2, 3, 2, 0), (0, 0, 3, 2, 3, 2, 0, 3, 3, 2, 3, 1, 3, 2, 0),
               (0, 0, 3, 2, 3, 2, 2, 3, 2, 1, 1, 3, 3, 0, 3), (0, 0, 3, 3, 0, 2, 3, 2, 2, 2, 3, 2, 2, 0, 1),
               (0, 0, 3, 3, 1, 3, 3, 3, 3, 1, 3, 3, 2, 2, 3), (0, 0, 3, 3, 2, 2, 3, 3, 1, 3, 3, 3, 3, 3, 2),
               (0, 1, 1, 1, 3, 1, 2, 0, 3, 3, 3, 3, 1, 2, 0), (0, 1, 2, 3, 1, 2, 2, 3, 3, 1, 2, 3, 0, 3, 0),
               (0, 1, 2, 3, 2, 2, 2, 1, 0, 3, 2, 3, 0, 3, 0), (0, 1, 3, 1, 0, 2, 2, 2, 3, 1, 3, 3, 2, 0, 3),
               (0, 1, 3, 2, 0, 2, 3, 2, 3, 3, 2, 2, 1, 3, 2), (0, 1, 3, 2, 3, 3, 2, 1, 3, 1, 3, 2, 3, 3, 0),
               (0, 1, 3, 3, 3, 1, 0, 3, 3, 0, 3, 0, 1, 0, 0), (0, 1, 3, 3, 3, 2, 1, 3, 3, 2, 3, 3, 0, 0, 3),
               (0, 1, 3, 3, 3, 2, 2, 3, 3, 0, 2, 0, 0, 2, 2), (0, 1, 3, 3, 3, 3, 3, 3, 2, 0, 3, 3, 0, 3, 2),
               (0, 2, 0, 3, 2, 2, 2, 3, 3, 3, 1, 1, 2, 3, 3), (0, 2, 0, 3, 2, 3, 1, 3, 3, 3, 2, 1, 2, 3, 2),
               (0, 2, 0, 3, 2, 3, 2, 3, 3, 0, 3, 3, 2, 2, 2), (0, 2, 0, 3, 3, 2, 1, 3, 2, 3, 2, 2, 1, 2, 1),
               (0, 2, 1, 1, 2, 3, 0, 0, 1, 3, 1, 0, 2, 2, 3), (0, 2, 1, 2, 2, 0, 1, 1, 3, 2, 2, 3, 0, 2, 2),
               (0, 2, 2, 1, 3, 3, 3, 3, 3, 0, 0, 2, 3, 2, 2), (0, 2, 2, 2, 2, 2, 1, 3, 1, 3, 2, 2, 0, 3, 2),
               (0, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 3, 1, 0, 3), (0, 2, 2, 2, 3, 2, 3, 0, 2, 3, 3, 1, 3, 3, 3),
               (0, 2, 2, 3, 3, 3, 3, 0, 2, 2, 3, 0, 2, 3, 1), (0, 2, 2, 3, 3, 3, 3, 3, 1, 2, 2, 1, 1, 3, 2),
               (0, 2, 3, 0, 3, 3, 3, 1, 3, 3, 0, 3, 3, 2, 2), (0, 2, 3, 1, 2, 3, 3, 2, 3, 3, 1, 2, 1, 1, 3),
               (0, 2, 3, 1, 3, 3, 2, 2, 2, 0, 3, 3, 3, 2, 2), (0, 2, 3, 2, 2, 1, 2, 2, 3, 0, 2, 3, 2, 3, 2),
               (0, 2, 3, 2, 2, 3, 1, 1, 2, 2, 1, 3, 0, 2, 1), (0, 2, 3, 2, 3, 3, 2, 0, 1, 3, 3, 3, 3, 2, 0),
               (0, 2, 3, 2, 3, 3, 3, 0, 2, 2, 2, 3, 2, 0, 0), (0, 2, 3, 3, 2, 0, 3, 0, 2, 3, 3, 0, 3, 0, 3),
               (0, 2, 3, 3, 2, 3, 0, 0, 2, 3, 0, 2, 2, 2, 2), (0, 3, 0, 2, 2, 3, 3, 3, 3, 2, 2, 0, 3, 2, 3),
               (0, 3, 0, 3, 3, 3, 3, 3, 0, 1, 2, 3, 3, 1, 2), (0, 3, 1, 1, 1, 2, 3, 1, 2, 3, 2, 3, 0, 3, 2),
               (0, 3, 1, 2, 1, 3, 2, 3, 3, 0, 3, 3, 1, 3, 2), (0, 3, 1, 2, 3, 3, 2, 3, 0, 2, 3, 0, 1, 3, 3),
               (0, 3, 1, 3, 3, 3, 2, 3, 3, 3, 2, 2, 3, 0, 1), (0, 3, 2, 0, 3, 0, 2, 2, 2, 1, 0, 1, 2, 3, 3),
               (0, 3, 2, 2, 2, 2, 3, 3, 2, 2, 1, 3, 3, 2, 1), (0, 3, 2, 3, 0, 1, 0, 2, 0, 1, 2, 2, 0, 2, 3),
               (0, 3, 2, 3, 0, 3, 2, 2, 3, 3, 3, 3, 1, 3, 1), (0, 3, 2, 3, 3, 2, 3, 3, 0, 3, 2, 3, 3, 2, 1),
               (0, 3, 3, 1, 3, 0, 2, 0, 2, 2, 3, 0, 3, 3, 3), (0, 3, 3, 1, 3, 3, 2, 3, 1, 3, 2, 0, 1, 2, 3),
               (0, 3, 3, 2, 1, 3, 2, 2, 3, 2, 0, 2, 3, 2, 2), (0, 3, 3, 2, 2, 1, 0, 3, 0, 2, 1, 3, 3, 3, 3),
               (0, 3, 3, 2, 3, 0, 1, 2, 3, 2, 2, 3, 2, 0, 2), (0, 3, 3, 2, 3, 1, 2, 3, 0, 3, 0, 1, 3, 2, 3),
               (0, 3, 3, 3, 1, 3, 3, 0, 3, 2, 2, 2, 0, 3, 2), (0, 3, 3, 3, 2, 3, 0, 0, 2, 0, 3, 3, 3, 3, 0),
               (0, 3, 3, 3, 2, 3, 3, 2, 1, 2, 3, 0, 2, 1, 3), (0, 3, 3, 3, 3, 2, 3, 2, 3, 3, 2, 2, 3, 2, 1),
               (1, 0, 0, 3, 0, 3, 2, 3, 0, 2, 1, 3, 1, 3, 2), (1, 0, 1, 2, 2, 3, 3, 2, 1, 2, 0, 2, 3, 2, 3),
               (1, 0, 2, 0, 2, 3, 2, 2, 3, 3, 2, 2, 3, 2, 0), (1, 0, 2, 3, 1, 3, 2, 3, 3, 3, 2, 2, 3, 2, 0),
               (1, 0, 2, 3, 1, 3, 3, 2, 1, 2, 2, 2, 2, 1, 3), (1, 0, 3, 3, 0, 3, 1, 3, 0, 3, 0, 1, 2, 2, 2),
               (1, 1, 1, 1, 2, 1, 1, 0, 3, 3, 3, 2, 3, 3, 2), (1, 1, 2, 2, 1, 3, 3, 2, 3, 2, 2, 1, 2, 3, 3),
               (1, 1, 2, 2, 1, 3, 3, 2, 3, 2, 3, 3, 3, 1, 2), (1, 1, 2, 3, 2, 3, 0, 2, 2, 3, 1, 3, 2, 3, 0),
               (1, 1, 2, 3, 3, 3, 2, 3, 0, 2, 1, 3, 3, 3, 0), (1, 1, 3, 0, 1, 2, 3, 3, 3, 3, 2, 2, 2, 3, 3),
               (1, 1, 3, 3, 2, 2, 3, 3, 3, 3, 2, 1, 3, 3, 2), (1, 2, 1, 0, 1, 3, 3, 2, 3, 3, 2, 1, 2, 3, 3),
               (1, 2, 1, 1, 3, 2, 3, 3, 3, 2, 1, 2, 2, 0, 2), (1, 2, 1, 2, 2, 1, 3, 3, 3, 3, 3, 3, 0, 1, 3),
               (1, 2, 1, 2, 2, 2, 3, 3, 1, 3, 0, 0, 3, 0, 2), (1, 2, 2, 0, 3, 3, 2, 3, 3, 3, 0, 2, 2, 3, 1),
               (1, 2, 2, 1, 2, 3, 2, 2, 3, 3, 2, 3, 2, 3, 2), (1, 2, 2, 2, 3, 3, 3, 2, 3, 0, 2, 3, 1, 1, 3),
               (1, 2, 2, 3, 2, 1, 1, 3, 0, 3, 3, 3, 3, 3, 3), (1, 2, 2, 3, 2, 1, 3, 2, 0, 1, 1, 3, 3, 1, 2),
               (1, 2, 2, 3, 3, 1, 2, 3, 3, 3, 2, 0, 2, 1, 3), (1, 2, 2, 3, 3, 2, 3, 3, 2, 1, 3, 1, 0, 2, 2),
               (1, 2, 2, 3, 3, 3, 0, 3, 3, 1, 2, 3, 2, 3, 3), (1, 2, 2, 3, 3, 3, 3, 3, 3, 1, 2, 0, 3, 0, 2),
               (1, 2, 3, 0, 0, 3, 1, 3, 3, 2, 3, 2, 2, 3, 3), (1, 2, 3, 0, 2, 2, 3, 2, 3, 3, 3, 3, 1, 2, 2),
               (1, 2, 3, 0, 2, 2, 3, 3, 3, 3, 2, 2, 3, 0, 2), (1, 2, 3, 1, 1, 3, 3, 3, 3, 3, 2, 3, 3, 2, 3),
               (1, 2, 3, 2, 2, 2, 2, 3, 2, 3, 3, 1, 1, 3, 3), (1, 2, 3, 3, 2, 1, 3, 1, 3, 2, 2, 0, 3, 3, 3),
               (1, 2, 3, 3, 3, 3, 2, 3, 2, 2, 2, 3, 2, 3, 0), (1, 3, 0, 0, 0, 2, 3, 1, 2, 3, 1, 3, 3, 3, 2),
               (1, 3, 0, 3, 1, 2, 3, 3, 1, 2, 0, 1, 2, 3, 1), (1, 3, 1, 0, 2, 2, 3, 2, 2, 1, 2, 3, 3, 3, 2),
               (1, 3, 1, 2, 0, 2, 2, 2, 2, 0, 1, 2, 3, 2, 3), (1, 3, 1, 2, 1, 3, 3, 2, 3, 3, 2, 0, 1, 2, 2),
               (1, 3, 1, 3, 2, 2, 3, 0, 3, 1, 2, 3, 3, 2, 1), (1, 3, 2, 0, 1, 2, 2, 3, 3, 2, 3, 3, 3, 2, 3),
               (1, 3, 2, 0, 1, 3, 0, 3, 3, 2, 1, 2, 3, 2, 3), (1, 3, 2, 0, 3, 1, 2, 1, 3, 0, 2, 0, 2, 3, 3),
               (1, 3, 2, 1, 3, 2, 2, 3, 2, 0, 3, 2, 0, 2, 2), (1, 3, 2, 2, 1, 3, 3, 0, 3, 3, 2, 3, 3, 3, 0),
               (1, 3, 2, 2, 3, 3, 1, 2, 0, 2, 3, 2, 0, 1, 2), (1, 3, 2, 3, 2, 0, 3, 3, 3, 0, 0, 2, 1, 3, 2),
               (1, 3, 2, 3, 3, 2, 3, 3, 3, 2, 0, 3, 0, 2, 0), (1, 3, 3, 1, 2, 2, 3, 3, 3, 2, 2, 3, 3, 1, 3),
               (1, 3, 3, 2, 2, 2, 1, 1, 1, 2, 1, 2, 3, 3, 1), (1, 3, 3, 2, 3, 1, 3, 1, 1, 1, 0, 3, 3, 3, 3),
               (1, 3, 3, 2, 3, 1, 3, 2, 3, 2, 0, 0, 0, 3, 3), (1, 3, 3, 2, 3, 3, 1, 3, 0, 2, 3, 0, 3, 3, 3),
               (1, 3, 3, 3, 1, 2, 3, 1, 1, 1, 0, 3, 1, 1, 3), (1, 3, 3, 3, 2, 1, 3, 2, 1, 3, 3, 1, 1, 1, 3),
               (1, 3, 3, 3, 2, 1, 3, 3, 0, 2, 2, 3, 3, 1, 0), (1, 3, 3, 3, 2, 3, 2, 3, 0, 2, 0, 0, 2, 0, 0),
               (2, 0, 0, 2, 2, 1, 2, 0, 2, 3, 3, 2, 0, 1, 3), (2, 0, 0, 2, 3, 2, 3, 3, 2, 0, 0, 3, 3, 1, 3),
               (2, 0, 1, 2, 3, 0, 1, 1, 1, 3, 3, 1, 0, 2, 2), (2, 0, 1, 2, 3, 3, 0, 0, 2, 3, 2, 3, 1, 2, 2),
               (2, 0, 1, 3, 3, 2, 0, 0, 2, 2, 3, 2, 2, 0, 0), (2, 0, 2, 1, 3, 3, 2, 0, 3, 3, 0, 1, 3, 3, 2),
               (2, 0, 2, 2, 0, 3, 2, 0, 3, 0, 3, 3, 3, 1, 0), (2, 0, 2, 3, 0, 1, 2, 0, 2, 3, 3, 3, 3, 3, 3),
               (2, 0, 3, 1, 2, 2, 3, 3, 3, 0, 3, 3, 3, 0, 3), (2, 0, 3, 1, 3, 2, 3, 2, 0, 3, 3, 2, 2, 2, 0),
               (2, 0, 3, 1, 3, 3, 3, 3, 2, 3, 2, 2, 3, 2, 2), (2, 0, 3, 2, 2, 0, 3, 2, 1, 1, 3, 2, 2, 3, 3),
               (2, 0, 3, 2, 2, 3, 3, 2, 3, 2, 2, 0, 3, 3, 3), (2, 0, 3, 2, 3, 0, 2, 3, 1, 3, 3, 2, 2, 2, 0),
               (2, 0, 3, 3, 0, 2, 0, 3, 3, 3, 3, 2, 1, 3, 1), (2, 0, 3, 3, 1, 3, 2, 3, 0, 3, 3, 2, 3, 0, 3),
               (2, 0, 3, 3, 2, 3, 2, 3, 0, 2, 2, 1, 2, 1, 0), (2, 1, 0, 0, 2, 3, 3, 3, 3, 2, 0, 3, 0, 2, 2),
               (2, 1, 0, 3, 3, 2, 1, 0, 2, 0, 0, 3, 0, 3, 3), (2, 1, 1, 3, 2, 3, 0, 3, 2, 3, 3, 3, 2, 3, 2),
               (2, 1, 1, 3, 2, 3, 0, 3, 3, 2, 1, 3, 2, 3, 3), (2, 1, 2, 0, 2, 2, 3, 1, 3, 2, 3, 3, 3, 1, 0),
               (2, 1, 2, 2, 2, 3, 2, 2, 3, 2, 3, 3, 2, 3, 1), (2, 1, 2, 3, 2, 0, 2, 2, 3, 3, 3, 0, 0, 1, 3),
               (2, 1, 3, 2, 3, 1, 0, 3, 1, 3, 3, 3, 2, 3, 1), (2, 1, 3, 2, 3, 2, 2, 0, 0, 2, 0, 3, 3, 2, 3),
               (2, 1, 3, 2, 3, 3, 2, 1, 3, 2, 2, 0, 3, 0, 3), (2, 1, 3, 2, 3, 3, 2, 2, 3, 2, 1, 2, 0, 2, 2),
               (2, 1, 3, 2, 3, 3, 2, 3, 0, 0, 2, 0, 2, 3, 3), (2, 1, 3, 3, 0, 3, 2, 2, 3, 3, 3, 3, 2, 3, 2),
               (2, 1, 3, 3, 3, 1, 2, 3, 2, 0, 2, 0, 3, 3, 3), (2, 1, 3, 3, 3, 2, 3, 3, 3, 2, 0, 1, 1, 3, 2),
               (2, 2, 0, 1, 2, 3, 2, 3, 2, 2, 0, 2, 1, 3, 2), (2, 2, 0, 1, 2, 3, 3, 3, 0, 3, 0, 0, 0, 2, 1),
               (2, 2, 0, 1, 3, 3, 3, 3, 2, 0, 3, 3, 1, 3, 2), (2, 2, 0, 2, 0, 3, 3, 2, 2, 2, 3, 3, 3, 3, 3),
               (2, 2, 1, 1, 2, 3, 2, 0, 3, 2, 3, 2, 3, 3, 2), (2, 2, 1, 2, 2, 3, 0, 2, 3, 3, 1, 3, 3, 1, 2),
               (2, 2, 1, 3, 1, 2, 1, 1, 1, 1, 3, 3, 2, 0, 3), (2, 2, 1, 3, 3, 3, 3, 1, 3, 2, 3, 1, 2, 2, 3),
               (2, 2, 1, 3, 3, 3, 3, 3, 1, 1, 3, 1, 1, 2, 3), (2, 2, 2, 1, 2, 0, 3, 3, 2, 0, 1, 3, 2, 2, 2),
               (2, 2, 2, 1, 2, 3, 3, 3, 3, 2, 2, 1, 2, 2, 0), (2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 1, 3),
               (2, 2, 2, 2, 3, 0, 2, 3, 2, 3, 3, 2, 2, 2, 0), (2, 2, 2, 2, 3, 2, 1, 2, 3, 3, 0, 3, 0, 2, 2),
               (2, 2, 2, 3, 1, 3, 1, 1, 2, 3, 3, 2, 2, 2, 2), (2, 2, 2, 3, 2, 0, 3, 2, 1, 0, 3, 1, 1, 2, 3),
               (2, 2, 2, 3, 3, 2, 0, 3, 2, 3, 0, 2, 1, 0, 3), (2, 2, 2, 3, 3, 2, 3, 2, 3, 3, 2, 2, 1, 2, 0),
               (2, 2, 3, 0, 0, 3, 1, 2, 2, 0, 1, 2, 1, 0, 3), (2, 2, 3, 0, 1, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2),
               (2, 2, 3, 0, 2, 0, 3, 1, 3, 3, 0, 2, 2, 1, 2), (2, 2, 3, 1, 0, 1, 2, 3, 3, 2, 2, 0, 3, 3, 3),
               (2, 2, 3, 1, 2, 3, 3, 3, 2, 2, 1, 2, 3, 3, 1), (2, 2, 3, 2, 0, 3, 3, 3, 2, 2, 3, 0, 3, 0, 2),
               (2, 2, 3, 2, 3, 2, 2, 3, 1, 3, 3, 3, 3, 0, 3), (2, 2, 3, 2, 3, 3, 1, 2, 3, 0, 2, 1, 2, 2, 2),
               (2, 2, 3, 2, 3, 3, 2, 3, 1, 3, 0, 3, 2, 3, 3), (2, 2, 3, 3, 2, 0, 0, 3, 3, 1, 2, 3, 3, 3, 3),
               (2, 2, 3, 3, 2, 2, 3, 3, 1, 2, 3, 3, 2, 3, 2), (2, 2, 3, 3, 2, 3, 0, 3, 3, 3, 1, 3, 0, 3, 0),
               (2, 2, 3, 3, 3, 0, 0, 3, 3, 3, 3, 1, 1, 2, 2), (2, 2, 3, 3, 3, 1, 3, 1, 1, 2, 2, 3, 2, 0, 2),
               (2, 2, 3, 3, 3, 3, 2, 2, 1, 3, 0, 3, 3, 0, 3), (2, 3, 0, 0, 3, 1, 2, 3, 3, 2, 2, 2, 3, 2, 0),
               (2, 3, 0, 1, 3, 0, 0, 2, 0, 3, 0, 3, 1, 1, 2), (2, 3, 0, 1, 3, 3, 1, 2, 1, 3, 0, 3, 3, 2, 0),
               (2, 3, 0, 2, 2, 2, 0, 2, 3, 3, 3, 2, 2, 2, 2), (2, 3, 0, 2, 3, 0, 0, 2, 3, 2, 2, 1, 2, 2, 2),
               (2, 3, 0, 2, 3, 1, 2, 3, 2, 2, 2, 3, 3, 0, 0), (2, 3, 0, 2, 3, 3, 0, 0, 2, 0, 3, 1, 2, 3, 3),
               (2, 3, 0, 2, 3, 3, 0, 3, 0, 3, 3, 0, 0, 3, 1), (2, 3, 0, 3, 0, 3, 2, 2, 3, 3, 1, 3, 2, 2, 3),
               (2, 3, 0, 3, 1, 3, 0, 1, 2, 2, 3, 2, 3, 1, 2), (2, 3, 0, 3, 2, 0, 2, 0, 3, 2, 3, 2, 2, 3, 2),
               (2, 3, 0, 3, 2, 3, 2, 0, 2, 2, 1, 0, 3, 3, 3), (2, 3, 0, 3, 3, 3, 2, 3, 3, 2, 3, 1, 2, 2, 0),
               (2, 3, 1, 1, 1, 2, 3, 3, 2, 2, 2, 2, 1, 2, 3), (2, 3, 1, 1, 2, 3, 0, 0, 2, 2, 2, 1, 3, 2, 2),
               (2, 3, 1, 2, 2, 1, 3, 3, 3, 3, 0, 2, 2, 3, 3), (2, 3, 1, 2, 2, 3, 0, 2, 3, 1, 2, 0, 1, 1, 0),
               (2, 3, 1, 2, 3, 0, 0, 0, 3, 3, 2, 3, 3, 1, 2), (2, 3, 1, 3, 1, 3, 3, 0, 0, 3, 1, 3, 1, 2, 3),
               (2, 3, 1, 3, 2, 3, 3, 3, 2, 1, 2, 0, 3, 3, 3), (2, 3, 1, 3, 3, 0, 3, 0, 3, 2, 2, 0, 2, 0, 3),
               (2, 3, 1, 3, 3, 3, 1, 1, 2, 1, 1, 2, 3, 3, 3), (2, 3, 2, 0, 0, 2, 0, 3, 1, 2, 3, 2, 3, 2, 1),
               (2, 3, 2, 0, 0, 3, 2, 0, 3, 1, 1, 1, 3, 3, 2), (2, 3, 2, 0, 2, 1, 3, 0, 3, 0, 2, 3, 0, 3, 3),
               (2, 3, 2, 0, 2, 3, 2, 2, 3, 3, 3, 3, 2, 3, 3), (2, 3, 2, 0, 3, 0, 0, 3, 3, 0, 2, 3, 3, 3, 1),
               (2, 3, 2, 1, 3, 3, 2, 1, 0, 1, 3, 1, 2, 2, 3), (2, 3, 2, 2, 1, 2, 2, 2, 1, 3, 1, 3, 2, 2, 2),
               (2, 3, 2, 2, 1, 3, 2, 3, 1, 2, 3, 3, 3, 0, 0), (2, 3, 2, 2, 2, 0, 3, 2, 3, 3, 0, 2, 3, 0, 2),
               (2, 3, 2, 2, 3, 0, 3, 2, 3, 3, 0, 0, 2, 3, 2), (2, 3, 2, 3, 0, 0, 3, 3, 0, 0, 2, 3, 1, 0, 2),
               (2, 3, 2, 3, 1, 0, 1, 0, 3, 3, 0, 0, 0, 2, 3), (2, 3, 2, 3, 1, 2, 2, 3, 1, 0, 1, 1, 2, 0, 0),
               (2, 3, 2, 3, 2, 1, 3, 3, 2, 0, 2, 3, 0, 2, 3), (2, 3, 2, 3, 2, 2, 1, 2, 2, 2, 3, 3, 3, 2, 2),
               (2, 3, 2, 3, 2, 3, 0, 3, 1, 2, 0, 2, 3, 2, 2), (2, 3, 2, 3, 3, 0, 3, 2, 0, 2, 2, 3, 2, 3, 3),
               (2, 3, 2, 3, 3, 0, 3, 2, 2, 1, 2, 2, 3, 1, 3), (2, 3, 3, 0, 2, 2, 2, 1, 3, 2, 3, 3, 1, 3, 3),
               (2, 3, 3, 0, 2, 3, 1, 2, 3, 0, 3, 0, 3, 2, 2), (2, 3, 3, 0, 3, 0, 3, 2, 1, 3, 3, 3, 1, 2, 2),
               (2, 3, 3, 0, 3, 2, 3, 3, 3, 3, 3, 1, 3, 0, 3), (2, 3, 3, 1, 1, 2, 3, 3, 3, 2, 3, 1, 1, 2, 3),
               (2, 3, 3, 1, 2, 1, 2, 3, 2, 2, 2, 3, 3, 2, 2), (2, 3, 3, 1, 3, 0, 3, 2, 2, 3, 2, 1, 3, 2, 0),
               (2, 3, 3, 2, 0, 0, 0, 3, 3, 3, 1, 1, 2, 3, 3), (2, 3, 3, 2, 0, 3, 3, 1, 1, 3, 3, 1, 1, 3, 3),
               (2, 3, 3, 2, 0, 3, 3, 3, 2, 3, 2, 0, 3, 3, 1), (2, 3, 3, 2, 1, 0, 2, 3, 3, 3, 1, 2, 1, 2, 0),
               (2, 3, 3, 2, 1, 2, 2, 1, 3, 2, 3, 2, 1, 2, 2), (2, 3, 3, 2, 2, 1, 3, 2, 2, 2, 3, 2, 2, 3, 2),
               (2, 3, 3, 2, 2, 3, 2, 2, 0, 3, 3, 0, 3, 0, 3), (2, 3, 3, 2, 3, 2, 1, 0, 1, 3, 3, 0, 1, 2, 2),
               (2, 3, 3, 2, 3, 2, 2, 0, 3, 3, 2, 1, 2, 1, 1), (2, 3, 3, 2, 3, 3, 1, 1, 1, 3, 2, 0, 0, 2, 3),
               (2, 3, 3, 2, 3, 3, 3, 0, 1, 0, 0, 3, 3, 3, 2), (2, 3, 3, 2, 3, 3, 3, 2, 3, 2, 1, 2, 0, 1, 3),
               (2, 3, 3, 3, 1, 3, 1, 3, 0, 0, 2, 3, 3, 0, 2), (2, 3, 3, 3, 2, 0, 3, 2, 3, 1, 2, 1, 3, 2, 1),
               (2, 3, 3, 3, 3, 2, 0, 3, 2, 2, 0, 3, 1, 2, 1), (3, 0, 0, 2, 3, 3, 3, 3, 3, 2, 2, 2, 3, 3, 3),
               (3, 0, 0, 3, 2, 0, 3, 3, 0, 3, 1, 2, 3, 0, 3), (3, 0, 1, 0, 3, 2, 0, 2, 2, 3, 3, 3, 3, 0, 2),
               (3, 0, 1, 2, 0, 3, 2, 3, 1, 2, 3, 0, 3, 2, 3), (3, 0, 1, 2, 3, 2, 0, 2, 2, 0, 3, 2, 1, 2, 0),
               (3, 0, 1, 3, 0, 3, 1, 0, 2, 0, 2, 1, 0, 3, 0), (3, 0, 1, 3, 3, 0, 3, 3, 1, 0, 2, 3, 3, 2, 3),
               (3, 0, 2, 1, 1, 2, 3, 2, 2, 2, 3, 2, 2, 2, 3), (3, 0, 2, 1, 3, 3, 2, 2, 0, 3, 0, 0, 1, 3, 2),
               (3, 0, 2, 2, 0, 2, 0, 3, 3, 0, 3, 2, 2, 2, 1), (3, 0, 2, 2, 2, 3, 3, 2, 3, 1, 3, 3, 1, 3, 3),
               (3, 0, 2, 2, 3, 1, 2, 1, 3, 1, 2, 2, 3, 3, 2), (3, 0, 2, 3, 2, 2, 2, 1, 0, 2, 3, 3, 2, 1, 3),
               (3, 0, 2, 3, 2, 3, 2, 2, 3, 1, 1, 2, 3, 3, 0), (3, 0, 3, 0, 2, 3, 1, 3, 3, 1, 3, 3, 3, 1, 3),
               (3, 0, 3, 2, 3, 3, 0, 3, 0, 3, 1, 3, 2, 3, 2), (3, 0, 3, 3, 0, 1, 3, 3, 2, 3, 3, 3, 0, 3, 3),
               (3, 0, 3, 3, 1, 2, 1, 3, 1, 0, 2, 2, 2, 3, 0), (3, 0, 3, 3, 1, 2, 2, 0, 3, 3, 3, 1, 2, 3, 3),
               (3, 0, 3, 3, 3, 3, 3, 3, 3, 1, 3, 0, 2, 0, 1), (3, 1, 0, 1, 2, 3, 3, 0, 2, 2, 3, 3, 1, 2, 3),
               (3, 1, 0, 3, 2, 3, 3, 3, 3, 2, 0, 2, 2, 3, 2), (3, 1, 1, 2, 2, 2, 1, 2, 3, 1, 3, 0, 2, 3, 3),
               (3, 1, 1, 3, 2, 3, 3, 3, 0, 1, 3, 2, 3, 2, 2), (3, 1, 1, 3, 3, 3, 3, 3, 0, 0, 2, 3, 3, 2, 3),
               (3, 1, 2, 0, 2, 3, 0, 2, 3, 2, 3, 1, 1, 3, 0), (3, 1, 2, 0, 2, 3, 2, 3, 1, 2, 1, 0, 0, 1, 1),
               (3, 1, 2, 3, 2, 2, 3, 3, 3, 2, 2, 3, 2, 2, 3), (3, 1, 2, 3, 2, 3, 1, 0, 0, 3, 1, 3, 2, 3, 3),
               (3, 1, 3, 0, 2, 3, 3, 2, 3, 0, 0, 3, 0, 2, 3), (3, 1, 3, 2, 0, 0, 3, 2, 0, 3, 3, 3, 3, 0, 0),
               (3, 1, 3, 2, 0, 1, 1, 3, 0, 2, 3, 2, 0, 3, 3), (3, 1, 3, 2, 0, 2, 1, 0, 3, 3, 3, 3, 3, 2, 0),
               (3, 1, 3, 3, 3, 2, 3, 2, 3, 2, 1, 3, 1, 1, 3), (3, 2, 0, 1, 2, 2, 1, 1, 3, 2, 3, 2, 3, 2, 0),
               (3, 2, 0, 2, 1, 2, 3, 2, 3, 1, 2, 0, 2, 3, 2), (3, 2, 0, 2, 1, 3, 2, 0, 0, 3, 3, 3, 3, 2, 3),
               (3, 2, 0, 2, 2, 3, 1, 0, 1, 2, 3, 2, 1, 2, 3), (3, 2, 0, 2, 3, 1, 2, 3, 3, 2, 3, 3, 3, 2, 0),
               (3, 2, 1, 1, 0, 2, 0, 3, 3, 2, 1, 1, 2, 3, 3), (3, 2, 1, 2, 0, 3, 2, 2, 0, 0, 0, 2, 3, 0, 3),
               (3, 2, 1, 2, 2, 3, 2, 3, 3, 3, 3, 1, 1, 2, 0), (3, 2, 1, 2, 3, 2, 2, 2, 1, 2, 2, 3, 3, 3, 2),
               (3, 2, 1, 3, 3, 3, 2, 3, 0, 3, 2, 1, 1, 3, 1), (3, 2, 2, 0, 3, 2, 1, 2, 2, 3, 3, 3, 0, 3, 2),
               (3, 2, 2, 1, 1, 3, 3, 3, 3, 2, 3, 2, 0, 2, 3), (3, 2, 2, 2, 2, 1, 2, 2, 1, 3, 3, 3, 0, 3, 2),
               (3, 2, 2, 2, 2, 2, 0, 0, 3, 1, 3, 3, 3, 2, 0), (3, 2, 2, 2, 3, 2, 3, 3, 0, 3, 2, 3, 3, 2, 3),
               (3, 2, 2, 3, 0, 0, 3, 1, 3, 2, 1, 1, 3, 2, 2), (3, 2, 2, 3, 0, 3, 3, 1, 0, 3, 3, 2, 0, 3, 3),
               (3, 2, 2, 3, 1, 2, 3, 2, 3, 3, 2, 2, 1, 0, 2), (3, 2, 2, 3, 2, 2, 2, 0, 2, 3, 1, 0, 2, 3, 2),
               (3, 2, 2, 3, 2, 3, 0, 2, 1, 3, 3, 3, 3, 1, 0), (3, 2, 2, 3, 3, 2, 0, 0, 0, 3, 3, 3, 3, 0, 3),
               (3, 2, 2, 3, 3, 2, 2, 2, 3, 2, 1, 3, 3, 2, 1), (3, 2, 2, 3, 3, 3, 0, 3, 1, 1, 3, 2, 3, 2, 3),
               (3, 2, 2, 3, 3, 3, 2, 1, 3, 2, 2, 0, 3, 1, 3), (3, 2, 2, 3, 3, 3, 3, 1, 2, 1, 3, 3, 1, 2, 1),
               (3, 2, 3, 0, 2, 2, 2, 0, 3, 1, 2, 3, 2, 2, 2), (3, 2, 3, 0, 2, 3, 3, 3, 3, 0, 3, 3, 0, 3, 1),
               (3, 2, 3, 0, 3, 3, 2, 0, 2, 3, 3, 2, 1, 3, 2), (3, 2, 3, 1, 0, 0, 3, 3, 3, 0, 2, 2, 0, 0, 3),
               (3, 2, 3, 1, 2, 0, 2, 3, 3, 1, 3, 2, 3, 0, 3), (3, 2, 3, 1, 2, 0, 3, 2, 2, 3, 3, 3, 1, 0, 3),
               (3, 2, 3, 1, 3, 1, 2, 1, 2, 0, 2, 3, 0, 2, 3), (3, 2, 3, 2, 0, 3, 3, 2, 2, 2, 3, 3, 3, 3, 0),
               (3, 2, 3, 2, 1, 2, 3, 2, 3, 3, 3, 3, 0, 2, 0), (3, 2, 3, 2, 1, 3, 3, 3, 3, 3, 3, 1, 0, 2, 3),
               (3, 2, 3, 2, 2, 3, 1, 3, 0, 1, 2, 3, 3, 3, 3), (3, 2, 3, 2, 3, 2, 3, 2, 2, 3, 3, 0, 3, 2, 3),
               (3, 2, 3, 3, 0, 2, 1, 1, 3, 2, 0, 0, 1, 3, 3), (3, 2, 3, 3, 0, 2, 2, 2, 2, 3, 0, 2, 0, 3, 0),
               (3, 2, 3, 3, 1, 3, 3, 1, 0, 2, 3, 2, 2, 2, 3), (3, 2, 3, 3, 2, 1, 2, 2, 3, 2, 3, 1, 2, 1, 0),
               (3, 2, 3, 3, 3, 1, 1, 2, 2, 1, 3, 3, 3, 1, 3), (3, 2, 3, 3, 3, 3, 3, 3, 2, 1, 0, 3, 1, 2, 0),
               (3, 3, 0, 0, 3, 2, 2, 3, 3, 3, 0, 3, 3, 2, 1), (3, 3, 0, 1, 2, 3, 3, 0, 2, 2, 3, 3, 0, 3, 2),
               (3, 3, 0, 2, 1, 2, 3, 2, 2, 1, 3, 3, 3, 0, 2), (3, 3, 0, 2, 2, 2, 3, 2, 1, 1, 3, 0, 3, 2, 2),
               (3, 3, 0, 2, 3, 2, 2, 2, 0, 3, 2, 0, 3, 3, 2), (3, 3, 0, 3, 0, 0, 3, 2, 3, 0, 2, 3, 3, 2, 3),
               (3, 3, 0, 3, 1, 2, 3, 0, 3, 1, 1, 3, 2, 3, 3), (3, 3, 0, 3, 3, 2, 1, 2, 2, 2, 0, 1, 3, 3, 3),
               (3, 3, 0, 3, 3, 2, 3, 2, 1, 0, 3, 3, 3, 3, 1), (3, 3, 1, 0, 1, 0, 3, 0, 0, 3, 3, 2, 3, 3, 3),
               (3, 3, 1, 0, 2, 1, 3, 3, 2, 2, 3, 2, 3, 1, 2), (3, 3, 1, 0, 3, 2, 1, 3, 3, 1, 2, 3, 2, 3, 3),
               (3, 3, 1, 0, 3, 2, 3, 0, 1, 2, 2, 3, 3, 2, 2), (3, 3, 1, 0, 3, 3, 1, 3, 2, 3, 0, 3, 0, 3, 3),
               (3, 3, 1, 0, 3, 3, 3, 2, 0, 1, 0, 3, 2, 2, 3), (3, 3, 1, 2, 2, 0, 3, 3, 3, 3, 0, 1, 0, 3, 1),
               (3, 3, 1, 2, 3, 3, 3, 3, 1, 2, 3, 3, 2, 1, 2), (3, 3, 1, 3, 0, 1, 2, 1, 2, 2, 0, 2, 2, 2, 0),
               (3, 3, 1, 3, 2, 2, 1, 3, 2, 2, 2, 3, 1, 3, 2), (3, 3, 1, 3, 3, 0, 3, 1, 3, 1, 2, 2, 3, 0, 3),
               (3, 3, 1, 3, 3, 2, 3, 2, 1, 1, 0, 0, 2, 2, 0), (3, 3, 2, 0, 2, 2, 1, 3, 3, 2, 2, 3, 1, 2, 3),
               (3, 3, 2, 0, 3, 3, 1, 3, 3, 3, 0, 2, 0, 3, 1), (3, 3, 2, 0, 3, 3, 3, 3, 2, 0, 2, 2, 0, 1, 3),
               (3, 3, 2, 1, 2, 3, 3, 0, 3, 3, 0, 3, 0, 2, 1), (3, 3, 2, 1, 3, 3, 3, 3, 3, 0, 0, 3, 2, 0, 2),
               (3, 3, 2, 2, 0, 0, 1, 2, 2, 3, 0, 2, 3, 3, 1), (3, 3, 2, 2, 1, 3, 0, 1, 2, 1, 2, 3, 3, 3, 3),
               (3, 3, 2, 2, 2, 2, 0, 3, 2, 2, 1, 0, 0, 3, 1), (3, 3, 2, 2, 3, 0, 3, 3, 0, 3, 3, 2, 2, 2, 3),
               (3, 3, 2, 2, 3, 1, 2, 1, 2, 1, 0, 0, 3, 1, 1), (3, 3, 2, 2, 3, 2, 2, 0, 3, 3, 3, 1, 3, 3, 3),
               (3, 3, 2, 2, 3, 3, 3, 3, 2, 0, 3, 0, 0, 3, 2), (3, 3, 2, 3, 0, 1, 0, 2, 2, 2, 3, 0, 2, 0, 3),
               (3, 3, 2, 3, 1, 0, 1, 3, 1, 3, 2, 2, 2, 3, 0), (3, 3, 2, 3, 1, 2, 0, 2, 3, 3, 1, 0, 1, 2, 0),
               (3, 3, 2, 3, 2, 3, 0, 2, 2, 2, 2, 1, 1, 0, 0), (3, 3, 2, 3, 2, 3, 1, 3, 2, 0, 2, 2, 3, 2, 3),
               (3, 3, 2, 3, 2, 3, 2, 1, 2, 3, 3, 3, 1, 0, 3), (3, 3, 2, 3, 2, 3, 3, 1, 3, 3, 1, 0, 3, 0, 2),
               (3, 3, 2, 3, 3, 3, 3, 1, 3, 1, 1, 0, 1, 0, 3), (3, 3, 3, 0, 1, 3, 3, 3, 0, 3, 1, 0, 3, 0, 3),
               (3, 3, 3, 0, 2, 0, 3, 2, 2, 3, 3, 1, 2, 3, 3), (3, 3, 3, 0, 2, 2, 2, 3, 3, 3, 3, 3, 3, 1, 3),
               (3, 3, 3, 0, 3, 0, 3, 2, 3, 2, 3, 0, 0, 3, 0), (3, 3, 3, 0, 3, 3, 2, 1, 2, 2, 2, 3, 3, 0, 3),
               (3, 3, 3, 0, 3, 3, 2, 3, 2, 0, 0, 2, 3, 2, 0), (3, 3, 3, 0, 3, 3, 3, 1, 1, 2, 3, 2, 1, 3, 0),
               (3, 3, 3, 1, 0, 2, 3, 1, 2, 1, 2, 3, 3, 3, 3), (3, 3, 3, 1, 0, 3, 3, 2, 2, 3, 2, 3, 2, 3, 2),
               (3, 3, 3, 1, 1, 3, 0, 1, 3, 0, 3, 0, 3, 3, 0), (3, 3, 3, 1, 2, 1, 2, 2, 3, 3, 2, 1, 2, 1, 0),
               (3, 3, 3, 1, 2, 3, 1, 3, 3, 3, 2, 2, 1, 3, 0), (3, 3, 3, 1, 3, 3, 3, 1, 2, 2, 1, 3, 1, 3, 3),
               (3, 3, 3, 2, 0, 2, 3, 1, 0, 2, 3, 3, 0, 3, 2), (3, 3, 3, 2, 0, 2, 3, 1, 0, 2, 3, 3, 3, 0, 2),
               (3, 3, 3, 2, 0, 3, 2, 3, 2, 1, 2, 1, 2, 3, 3), (3, 3, 3, 2, 1, 2, 2, 3, 0, 3, 0, 3, 3, 1, 3),
               (3, 3, 3, 2, 1, 2, 3, 3, 2, 1, 3, 1, 3, 0, 3), (3, 3, 3, 2, 1, 3, 2, 3, 1, 1, 3, 2, 1, 0, 1),
               (3, 3, 3, 2, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 0), (3, 3, 3, 2, 3, 1, 2, 2, 3, 1, 3, 3, 2, 0, 3),
               (3, 3, 3, 2, 3, 3, 0, 3, 3, 3, 3, 2, 0, 1, 2), (3, 3, 3, 2, 3, 3, 1, 2, 3, 2, 2, 0, 3, 0, 3),
               (3, 3, 3, 3, 1, 3, 3, 2, 3, 2, 3, 0, 0, 1, 3), (3, 3, 3, 3, 2, 1, 2, 0, 0, 3, 3, 3, 3, 3, 0),
               (3, 3, 3, 3, 2, 3, 0, 3, 2, 0, 3, 2, 0, 1, 2), (3, 3, 3, 3, 2, 3, 3, 1, 0, 0, 2, 0, 1, 1, 3),
               (3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 2, 1, 0, 2, 3), (3, 3, 3, 3, 3, 3, 1, 2, 3, 0, 0, 2, 3, 3, 3)]

    mpv6_10 = mpv6[:10]
    mpv6_20 = mpv6[:20]
    mpv6_24 = mpv6[:]

    mpv7_10 = mpv7[:10]
    mpv7_20 = mpv7[:20]
    mpv7_45 = mpv7[:]

    mpv8_10 = mpv8[:10]
    mpv8_20 = mpv8[:20]
    mpv8_30 = mpv8[:30]
    mpv8_45 = mpv8[:45]
    mpv8_63 = mpv8[:]

    mpv10_100 = mpv10[:100]
    mpv10_200 = mpv10[:200]
    mpv10_300 = mpv10[:300]
    mpv10_355 = mpv10[:]

    mpv15_100 = mpv15[:100]
    mpv15_200 = mpv15[:200]


    # lists containing lists with different number of components
    lista6 = [mpv6_10, mpv6_20, mpv6_24]
    lista7 = [mpv7_10, mpv7_20, mpv7_45]
    lista8 = [mpv8_10, mpv8_20, mpv8_30, mpv8_45, mpv8_63]
    lista10 = [mpv10_100, mpv10_200]
    lista15 = [mpv15_100, mpv15_200]

    # probability matrices
    # probability matrix for vectors with 6 components
    component_probs_6 = [
        [0.1, 0.1, 0, 0.8],
        [0.1, 0, 0, 0.9],
        [0.2, 0, 0.8, 0],
        [0.1, 0, 0.9, 0],
        [0.1, 0.05, 0.05, 0.8],
        [0.1, 0.2, 0.3, 0.4]
    ]

    # probability matrix for vectors with 7 components
    component_probs_7 = [
        [0.1, 0.1, 0, 0.8],
        [0.1, 0, 0, 0.9],
        [0.2, 0, 0.8, 0],
        [0.1, 0, 0.9, 0],
        [0.1, 0.05, 0.05, 0.8],
        [0.1, 0.2, 0.3, 0.4],
        [0.05, 0.4, 0.05, 0.5]
    ]

    # probability matrix for vectors with 8 components
    component_probs_8 = [
        [0.1, 0.1, 0, 0.8],
        [0.1, 0, 0, 0.9],
        [0.2, 0, 0.8, 0],
        [0.1, 0, 0.9, 0],
        [0.1, 0.05, 0.05, 0.8],
        [0.1, 0.2, 0.3, 0.4],
        [0.05, 0.4, 0.05, 0.5],
        [0.2, 0.3, 0.2, 0.3]
    ]

    # probability matrix for vectors with 10 components
    component_probs_10 = [
        [0.1, 0.1, 0, 0.1, 0.7],
        [0.1, 0, 0, 0.8, 0.1],
        [0.2, 0, 0.7, 0, 0.1],
        [0.1, 0, 0.2, 0, 0.7],
        [0.1, 0.05, 0.05, 0.6, 0.2],
        [0.1, 0.1, 0, 0.1, 0.7],
        [0.1, 0, 0, 0.8, 0.1],
        [0.2, 0, 0.7, 0, 0.1],
        [0.1, 0, 0.2, 0, 0.7],
        [0.1, 0.05, 0.05, 0.6, 0.2]
    ]

    # probability matrix for vectors with 15 components
    component_probs_15 = [
        [0.1, 0.1, 0, 0.1, 0.7],
        [0.1, 0, 0, 0.8, 0.1],
        [0.2, 0, 0.7, 0, 0.1],
        [0.1, 0, 0.2, 0, 0.7],
        [0.1, 0.05, 0.05, 0.6, 0.2],
        [0.1, 0.1, 0, 0.1, 0.7],
        [0.1, 0, 0, 0.8, 0.1],
        [0.2, 0, 0.7, 0, 0.1],
        [0.1, 0, 0.2, 0, 0.7],
        [0.1, 0.05, 0.05, 0.6, 0.2],
        [0.1, 0.1, 0, 0.1, 0.7],
        [0.1, 0, 0, 0.8, 0.1],
        [0.2, 0, 0.7, 0, 0.1],
        [0.1, 0, 0.2, 0, 0.7],
        [0.1, 0.05, 0.05, 0.6, 0.2]
    ]

    triplets = [(lista6, component_probs_6, 6), (lista7, component_probs_7, 7), (lista8, component_probs_8, 8), (lista10, component_probs_10, 10), (lista15, component_probs_15, 15)]

    # changing number of components
    for t in triplets:
        print(f"MEASURING FOR {t[2]}-COMPONENT VECTORS:")
        lista = t[0]
        component_probs = t[1]

        # changing method
        for k in range(len(values)):
            print(f"METHOD #{k+1}:")

            method = values[k][0]
            metric = values[k][1]
            centroid_method = values[k][2]
            algorithm = values[k][3]

            # changing test case
            for j in range(len(lista)):
                mpv_1_tuples = [tuple(abs(x) for x in v) for v in lista[j]]

                # removing global variables from previous iterations
                reliability_cache.clear()
                LIST.clear()

                rel = recursive_reliability(method, metric, centroid_method, algorithm, mpv_1_tuples, component_probs)

                num_calls = max(LIST)

                # printing maximum depth
                print(f"{num_calls}")
