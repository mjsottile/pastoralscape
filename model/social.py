import numpy as np

class SocialNetwork:
    def __init__(self, n):
        """ Adjacency matrix contains connectivity of agents. 
            Non-zero entries are the strengths of influence. """
        self.adj = np.zeros((n,n))

        """ Identifier counter initialized to zero. """
        self.id = 0

        """ Collection of individuals starts empty. """
        self.individuals = {}

    def add_individual(self, i):
        """ Associate a given individual with their index in the adjacency
            matrix. Also store a list of adjacent individuals"""
        self.individuals[i] = (self.id, [])

        """ Increment identifier. """
        self.id += 1

    def add_relationship(self, i, j, wij, wji):
        """ Add relationship between two individuals.  Individuals are
            passed in as objects, and their adjacency matrix indices are
            looked up.  Weights are not necessarily symmetric. 
            
            Important: Don't add i,j and then j,i! 
        """
        (id_i, i_adj) = self.individuals[i]
        (id_j, j_adj) = self.individuals[j]
        i_adj.append(j)
        j_adj.append(i)
        self.adj[id_i,id_j] = wij
        self.adj[id_j,id_i] = wji

    def w(self, i, j):
        """ Get the weight associated with the directed relationship
            ij.  """
        return self.adj[self.individuals[i][0], self.individuals[j][0]]

    def neighbors(self, i):
        """ Neighbors are the second element of the tuple associated
            with i in the self.individuals map. """
        return self.individuals[i][1]
