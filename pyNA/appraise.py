import numpy as np

def logPPDNA(node, misfits):
    """ Calculate the log-posterior probability density function of model
    
    This routine converts the input data array to a log posteriori probability density
    function.
    
    For example if the input data for each model is a simple sum of squares of residuals weighted by
    a priori data covariances (i.e standard least squares). Then the Log-PPD is just a factor of -.5 times this.
    This rouine allows the user to use other Likelihood functions (or forms of posterirori probability density)
    and if necessary rescale them, or include a priori PDFs
    
    """
    return -0.5 * misfits[node]


class Appraise(object):
    
    def __init__(self, models, model, misfits):
        """

        models: ensemble of models
        x: starting point of the NA-Walk
        misfit: misfit values.
        nsample: number of samples to be collected from the NA-walk
        """
        self.models = models
        self.id_model_start = model
        self.misfits = np.random.rand(len(models))
        self.ne = len(models) # Number of models in the ensemble
        self.nd = models.shape[-1] # Dimension of the parameter space

    @staticmethod
    def NNcalc_dlist(dim, m, pt):
        """ Calculates square of distance from all models to new axis dim
            (defined by dimension dim through point pt)
            It also updates the nearest model and distance to pt.

        Returns
        -------

        dk2: numpy.ndarray, perpendicular distance to axis dim through point pt.
        dnodex: float, distance to nearest model.
        nodex: int, index of nearest model to point pt.
        """
        d = (m - pt) ** 2
        d2 = np.sum(d, axis=1)
        nodex = np.argmin(d2)
        dnodex = d2[nodex]
        dk2 = np.sum(np.delete(d, dim, axis=1), axis=1)
        return dk2, dnodex, nodex

    @staticmethod
    def NNupdate_dlist(dim, dimlast, m, pt, dk2):
        """ Calculates square of distance from all base points to new
            axis, assuming dlist contains square of all distances to
            previous axis dimlast.
            
            It also updates the nearest node to the point pt through which the axes
            pass.
        
        Returns
        -------

        dk2: numpy.ndarray, perpendicular distance to axis dim through point pt.
        dnodex: float, distance to nearest model.
        nodex: int, index of nearest model to point pt.
        """
        d1 = (m[:, dimlast] - pt[dimlast]) ** 2
        d = dk2 + d1
        nodex = np.argmin(d)
        dnodex = d[nodex]
        d2 = (m[:, dim] - pt[dim]) ** 2
        dk2 = dk2 + d1 - d2
        return dk2, dnodex, nodex

    @staticmethod
    def NNaxis_int(axis, models, model_id, dk2, left=True, right=True, root=True):
        """Find intersections of Voronoi cells with current 1D

        Returns:
        --------

        xp: numpy.ndarray, positions of voronoi boundaries along axis
        nodes: indices of voronoi cells at xp.

        This method uses a simple formula to exactly calculate the
        intersections of the Voronoi cells with the 1D axis.
        It makes use of the perpendicular distances of all models to
        the current axis.

        """
        xp = []
        nodes = []
        left_node = None
        right_node = None
        left_intersect = None
        right_intersect = None
        range_min = 0.
        range_max = 1.0
            
        x0 = models[model_id, axis]
        dp0 = dk2[model_id]
        
        if root:
            nodes += [model_id]
        
        xc = models[:, axis]
        dpc = dk2
        dx = x0 - xc
        
        xi = 0.5 * (x0 + xc + np.divide((dp0 - dpc), dx, out=np.zeros_like((dp0 - dpc)), where=dx!=0))

        if left:
            x1 = np.ma.array(xi, mask=(x0 <= xc))
            if x1.count() > 1:
                left_node = x1.argmax()
                left_intersect = x1[left_node]
            
        if right:
            x2 = np.ma.array(xi, mask=(x0 >= xc))
            if x2.count():
                right_node = x2.argmin()
                right_intersect = x2[right_node]
                    
        if left_node is not None and (left_intersect >= range_min):
            xp = [left_intersect] + xp
            nodes = [left_node] + nodes
            out1, out2 = Appraise.NNaxis_int(
                axis, models, left_node, dk2, 
                left=True, right=False, root=False)
            xp = out1 + xp
            nodes = out2 + nodes
            
        if right_node is not None and (right_intersect <= range_max):
            xp = xp + [right_intersect]
            nodes = nodes + [right_node]
            out1, out2 = Appraise.NNaxis_int(
                axis, models, right_node, dk2,
                left=False, right=True, root=False)
            xp = xp + out1
            nodes = nodes + out2
            
        return xp, nodes

    @staticmethod
    def NA_randev(xvals, nodes, misfits, n):
        """ Generate a random deviate according to a 1D NA PDF
            using a rejection method.

        This routine generates a random deviate distributed according to a
        1D conditional PPD using a rejection method.

        At each state two uniform deviates are generated. One is transformed
        to a r, v along the axis and the other is used to accept or reject it,
        The 1D PDF is determined from the NA approximation.
        """
        
        def pdf(xp, x):
            i = 0
            while x > xp[i] and i < len(xp) - 1:
                i += 1
            return i
        
        # Maximum of conditional PPD
        pmax = np.max(misfits[nodes])
        
        # Counters  
        naccept=0  
        ntrial=0  
        
        ran=[] # output list of random numbers  
        while naccept<n:  
            x=np.random.uniform(0.,1.0) # x'  
            y=np.random.uniform(0.,pmax) # y'
            
            if y < misfits[pdf(xvals, x)]:
            #if y < logPPDNA(pdf(xvals, x), misfits):
                ran.append(x)
                naccept += 1
            ntrial += 1
                
        return np.array(ran), ntrial
        
    def na_walk(self, nsample=100, nsleep=10):
        """ Generate a random walk using the Neighborhood approximation
            of the PPD built from the input ensemble of models.

        The total number of models generated in the random walk is nsample * nsleep.
        """
        
        new_models = np.zeros((nsample * nsleep, 2))
        idx = 0
        x = self.models[self.id_model_start]
        # Step of walk
        for walk_step in range(nsample): # Need to add burnin
            print(f"Doing walk step {walk_step}")
            # Sleep step
            for sleep_step in range(nsleep):
                print(4*"==" + f"Doing sleep step {sleep_step}")
                istep = sleep_step + (walk_step - 1) * nsample
                # axis loop
                for axis in range(self.nd):
                    print(10*" " + f"Doing axis {axis}")
                    # Calculate dlists
                    dk2, _, nodex = self.NNcalc_dlist(axis, self.models, x)
                    # calculate intersection of voronoi cells with current 1D axis
                    xp, nodesx = self.NNaxis_int(axis, self.models, model_id=nodex, dk2=dk2)
                    # generate random deviate (update x) according to neighborhood approximation
                    # of conditional probability distribution
                    x[axis], _ = self.NA_randev(xp, nodesx, self.misfits, 1)
                new_models[idx] = x
                idx += 1

        return new_models