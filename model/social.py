###########################################################################
# MIT License
#
# Copyright (c) 2020 Matthew Sottile
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###########################################################################
"""
Code for representing social networks connecting agents.
"""

import numpy as np

class SocialNetwork:
    """ Represent a social network as a weighted directed graph. """
    def __init__(self, n):
        """ Adjacency matrix contains connectivity of agents.
            Non-zero entries are the strengths of influence. """
        self.adj = np.zeros((n, n))

        """ Identifier counter initialized to zero. """
        self.current_id = 0

        """ Collection of individuals starts empty. """
        self.individuals = {}

    def add_individual(self, i):
        """ Associate a given individual with their index in the adjacency
            matrix. Also store a list of adjacent individuals"""
        self.individuals[i] = (self.current_id, [])

        # Increment identifier.
        self.current_id += 1

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
        self.adj[id_i, id_j] = wij
        self.adj[id_j, id_i] = wji

    def weight(self, i, j):
        """ Get the weight associated with the directed relationship
            ij.  """
        return self.adj[self.individuals[i][0], self.individuals[j][0]]

    def neighbors(self, i):
        """ Neighbors are the second element of the tuple associated
            with i in the self.individuals map. """
        return self.individuals[i][1]
