import numpy as np

"""[Recreates the adjacency matrix with which the steady state probabilities get multiplied iteratively
    The adjacency matrix A of a set of pages (nodes) defines the linking structure]

Returns:
    [numpy Matrix] -- [The matrix with which the steady state probabilities will get multipled]
"""
def recreate_adjacency_matrix(marchov_chain):
    # since the size of the matrix is n x n getting the size of the row is enough
    num_urls = marchov_chain.shape[0]

    # the probabilty of visiting a url, which is the 1 / total number of urls
    probability = 1 / num_urls

    probability_matrix = np.zeros((num_urls, num_urls))

    # probability_matrix will contain all similar values which is the probability of visiting a particular page
    probability_matrix[:] = probability
    
    '''
        In assigning a PageRank score to each node of the web graph, we use the teleport operation in two ways: 
        (1) When at a node with no out-links, the surfer invokes the teleport operation. 
        (2) At any node that has outgoing links, the surfer invokes the teleport operation with a probability of alpha.
        Typical value of alpha is 0.1 
    '''
    alpha = 0.1
    
    adjacency_matrix = alpha * marchov_chain + ((1 - alpha) * probability_matrix)

    return adjacency_matrix

"""[Computes the page rank of a given marchov chain url]

Returns:
    [list] -- [A list of n page rank score where n is the total number of urls]
"""
def compute_page_rank(marchov_chain):
    # create the adjacency matrix with which the steady state probability will get multiplied iteratively
    adjacency_matrix = recreate_adjacency_matrix(marchov_chain)

    num_urls = adjacency_matrix.shape[0]

    # initial vector for the steady state probabilities will always be <1, 0, 0.....n>
    steady_state_probabilities = np.zeros((1, num_urls))
    # this creates a vector of <1, 0, 0...n>
    steady_state_probabilities[0][0] = 1 
    
    # the steady_state probabilities always needs to be transposed
    steady_state_probabilities = np.transpose(steady_state_probabilities)
    
    previous_state_probabilities = steady_state_probabilities
    while True:
        steady_state_probabilities = adjacency_matrix * steady_state_probabilities
        
        # we stop when the values have converged and no longer change over the iterations
        if (previous_state_probabilities == steady_state_probabilities).all():
            # if the values converge then the steady_state_probabilities are returned which is the page rank score of the urls
            return steady_state_probabilities
        
        # otherwise the current steady state probabilities becomes the previous steady state probabilities as we are about to begin another iterations
        previous_state_probabilities = steady_state_probabilities
    

if __name__ == '__main__':

    '''
        NOTE - This marchov chain is transpose of the modified adjacency matrix of the graph.
        In an adjacency matrix the rows represent an individual url in the graph and columns
        represent the urls that the graph has already visited. This adjacency matrix will be 
        transposed and modified to recreate the adjacency matrix. If a url has visited other urls then 
        the columns will get replaced by 1 / n, where n is the total number of urls visited by that url. 
        The matrix will essentially be a transpose of the adjacency matrix with the columns divided by total non zero entries.
    '''
    marchov_chain = np.matrix([[0, 0, 1],
            [1, 0.5, 0],
            [0, 0.5, 0]])

    
    page_rank = compute_page_rank(marchov_chain)

    print(page_rank)

    print(np.sum(page_rank))