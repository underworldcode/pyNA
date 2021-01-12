import numpy as np
import matplotlib.pyplot as plt

class Sampler(object):
    
    def __init__(self, 
                 objective_function,
                 lower_bounds = (0, 0.),
                 upper_bounds = (1.0, 1.0),
                 n_initial = 10,
                 n_samples = 100,
                 n_resample = 10,
                 n_iterations = 3):
        
        """
        References:
            Sambridge, M. (1999). Geophysical inversion with a neighbourhood
            algorithm - I. Searching a parameter space. Geophysical Journal
            International, 138(2), 479–494.
        """
        
        self.ni = n_initial
        self.ns = n_samples
        self.nr = n_resample
        self.lower_bounds = np.array(lower_bounds)
        self.upper_bounds = np.array(upper_bounds)
        self.n_iterations = n_iterations
        
        # Number of dimensions
        self.nd = self.lower_bounds.size
        # Size of ensemble
        self.ne = self.ns * (self.n_iterations - 1) + self.ni
        # Array to hold samples models
        self.models = np.zeros((self.ne, self.nd))
        # Array to hold misfits
        self.misfits = np.zeros(self.ne)
        
        # This is the function we aim to minimise
        self.objective_function = objective_function
        
        self.random_generator = np.random.RandomState()
        
        self.np = 0 # Current number of models generated.
        
        self.lower_bounds_non_dim = 0.
        self.higher_bounds_non_dim = 1.0
        
    def generate_ensemble(self):
        # Generate the ensemble of models following Sambridge
        # Neighborhood Algorithm
        
        idx = 0
              
        for it in range(self.n_iterations):
            
            if it == 0:
                ns = self.ni
                self.queue = self.generate_random_models(ns) 
            else:
                ns = self.ns
                self.queue = self.na_sampling(ns)                                 
           
            self.models[idx:idx+ns] = self.queue
            # Dimensionalise...
            self.queue = self.lower_bounds + (self.upper_bounds - self.lower_bounds) * self.queue
            self.misfits[idx:idx+ns] = self.objective_function(self.queue)
            idx += ns
            self.np += ns
            print("iter %i, %i models" % (it, idx))
            
    def na_sampling(self, ns):
        
        if ns is None:
            ns = self.ns
            
        # Get the best models so far, They will define
        # the voronoi cells to resample.
        new_models = np.zeros((ns, self.nd))
        bests_so_far = self.get_bests_models(self.nr)
        m = all_models_so_far = self.models[:self.np, :]
        idx = 0
        
        # Loop through all the voronoi cells
        for k, vk in enumerate(bests_so_far):
            
            # We Calculate the walk length for the cell.
            # The Walk length should be the same for all cells
            # except if ns % nr != 0. In that case we oversample
            # the cell with the lowest misfit so far.
            walk_length = int(np.floor(ns / self.nr))
            if k == 0: # Best model so far
                # Prioritize best model so far in terms of number of
                # new models being generated per cell
                # This is required as the number of samples (ns) might be greater than
                # the number of cells to resample (nr).
                walk_length += int(np.floor(ns % self.nr))
            
            # Squared distance between vk and the previous models.
            # This will be used to calculate dk2 which is the
            # perpendicular distance to the i-axis.
            d2 = np.sum((m - vk) ** 2, axis=1)
           
            # Initialise distance to previous axis to 0.
            d2_prev_axis = 0.
            
            # This is the actual random walk inside the cell around vk
            for step in range(walk_length):
                # Initialize the walk location to the center of the cell
                xA = vk.copy()
                # Iterate through axes in random order
                # That is just an extra precaution. We could probably do
                # without it.
                axes = self.random_generator.permutation(self.nd)
                for id_ax, i in enumerate(axes):
                    # Current squared distance ALONG the current axis
                    d2_current_axis = (m[:, i] - xA[i]) ** 2
                    # Calculate dk2, remove the distance along the current axis
                    # and add the distance along the previous axis as it was removed
                    # at the previous step. This ensure that the squared perpendicular 
                    # distance between the sample k and the current axis is correct.
                    d2 += d2_prev_axis - d2_current_axis
                    dk2 = d2[k]
                    
                    # Find distance along axis i to all cell edges (see eq19 in Sambridge 1999)
                    # To find the required boundaries of the Voronoi cell,
                    # eq. (19) must be evaluated for all n cells and the two closest
                    # points either side of x retained.
                    vji = m[:, i]
                    
                    # I am not a big fan of those 3 lines, but it's a way to handle divide by
                    # zero...
                    a = (dk2 - d2)
                    b = (vk[i] - vji)
                    xji = 0.5 *(vk[i] + vji + np.divide(a, b, out=np.zeros_like(a), where=b!=0))
                    
                    # Because of the above "trick" we still have the departure point in xji
                    # so we need to use strict > and < conditions.
                    li = np.nanmax(
                             np.hstack((self.lower_bounds_non_dim, xji[xji < xA[i]])))
                    ui = np.nanmin(
                             np.hstack((self.higher_bounds_non_dim, xji[xji > xA[i]])))
                        
                    # random move within voronoi polygon
                    xA[i] = (ui - li) * self.random_generator.random_sample() + li
                    d2_prev_ax = d2_current_axis
                        
                new_models[idx] = xA.copy()
                idx += 1
                        
        return new_models         
    
    def get_dimensionalised_models(self, models):
        return self.lower_bounds + (self.upper_bounds - self.lower_bounds) * models
        
    def generate_random_models(self, n):
        # Generate a random set of n models.
        return self.random_generator.random_sample((n, self.nd))
        
    def get_bests_models(self, nr):
        # Return the list of best models so far
        if nr is None:
            nr = self.nr
            
        best_models_ids = np.argsort(self.misfits[:self.np])[:nr]
        return self.models[best_models_ids]
    
    def plot(self):
        
        from scipy.spatial import Voronoi, voronoi_plot_2d
        models = self.get_dimensionalised_models(self.models)
        vors = Voronoi(models)
        fig = voronoi_plot_2d(vors, show_vertices=False)
        plt.xlim(self.lower_bounds[0], self.upper_bounds[0])
        plt.ylim(self.lower_bounds[1], self.upper_bounds[1])
        return