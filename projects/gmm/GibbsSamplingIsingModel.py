################################################################################
# Implementation of Gibbs sampling for both prior and posterior in a two-state
# Ising model.
################################################################################

import numpy as np
from matplotlib import pyplot as plt
import os
import copy as cp
from scipy import misc

class GibbsDenoising(object):

    img = None
    evidence = None
    posterior = None
    prior = None

    def __init__(self, img_path, ising_J, ev_sigma):
        self.load_img(img_path)
        self.nrows = self.img.shape[0]
        self.ncols = self.img.shape[1]
        self.J = ising_J
        self.sigma = ev_sigma
        self.set_evidence()


    def nbrs(self, x, y, arr):
        return np.array([arr[x, (y + 1)] if y < (self.ncols - 1) else 0,
                        arr[x, (y - 1)] if y > 0 else 0,
                        arr[(x - 1), y] if x > 0 else 0,
                        arr[(x + 1), y] if x < (self.nrows - 1) else 0])



    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def agree(self, x, y, arr):
        curr_state = arr[x,y]
        curr_nbrs = self.nbrs(x,y, arr)
        return np.sum([np.abs(s) for s in curr_nbrs if s == curr_state])

    def disagree(self, x, y, arr):
        curr_state = arr[x,y]
        curr_nbrs = self.nbrs(x,y, arr)
        return np.sum([np.abs(s) for s in curr_nbrs if s != curr_state])

    def sign(self, x):
        if x > 0: return 1
        elif x <= 0: return -1

    @staticmethod
    def gauss_density(x, loc, scale):
        norm = 1 / np.sqrt(2 * np.pi * scale)
        return norm * np.exp(-((x-loc) ** 2)/(2*scale))

    def update_prior(self, x, y):
        assert (x < self.nrows) and (y < self.ncols), \
            "Coordinates (%d, %d) not within bounds (%d, %d)" %(x,y, self.nrows, self.ncols)
        eta = self.prior[x,y] * (self.agree(x,y, self.prior) - self.disagree(x,y, self.prior))
        p_state1 = self.sigmoid(2 * self.J * eta)
        self.prior[x,y] = self.sign(np.random.binomial(1, p_state1))


    def update_posterior(self,x,y):
        assert (x < self.nrows) and (y < self.ncols), \
            "Coordinates (%d, %d) not within bounds (%d, %d)" %(x,y, self.nrows, self.ncols)

        state = self.posterior[x,y]

        ev = self.evidence[x,y]
        sig = self.sigma
        eta = state * (self.agree(x,y, self.posterior) - self.disagree(x,y, self.posterior))
        ev_term = np.log(self.gauss_density(ev, -1, sig) / self.gauss_density(ev, 1, sig))

        p_state1 = self.sigmoid(2 * self.J * eta - ev_term)
        self.posterior[x,y] = self.sign(np.random.binomial(1, p_state1))


    def sample_prior(self, n_iter):
        self.prior = np.random.binomial(n=1, p=0.5, size=(self.nrows,self.ncols))
        self.prior[self.prior == 0] = -1
        xdim = self.prior.shape[1]
        ydim = self.prior.shape[0]
        for n in range(n_iter):
            print("Iteration %d" % n)
            for x in range(xdim):
                for y in range(ydim):
                    self.update_prior(x,y)


    def sample_posterior(self, n_iter):
        evidence = self.evidence
        assert evidence is not None, "First call evidence() to get the observations."
        assert len(evidence.shape) == 2, "Expected evidence as 2d np-array."

        self.posterior = cp.deepcopy(evidence)

        for n in range(n_iter):
            print("Iteration %d" % n)
            for x in range(evidence.shape[0]):
                for y in range(evidence.shape[1]):
                    self.update_posterior(x,y)


    def load_img(self, path):
        assert os.path.exists(path), "Path %s does not exist." % path
        img = misc.imread(path, flatten=True)
        img = np.asarray(img)
        img[img > 200] = -1
        img[img > 0] = 1
        self.img = img
        self.n_svar = self.img.shape[0] * self.img.shape[1]

    def set_evidence(self):
        assert self.img is not None, "First load image."
        noise = np.random.normal(loc=0, scale=self.sigma, size=(self.nrows, self.ncols))
        noise = np.reshape(noise, (self.nrows, self.ncols))
        assert noise.shape == self.img.shape, \
            "Noise shape (%d, %d) does not match image shape (%d,%d)" % \
            (noise.shape[0],noise.shape[1],self.img.shape[0],self.img.shape[1])
        f = np.vectorize(self.sign)
        self.evidence = f(self.img + noise)

    def show_gibbs_denoising(self):
        self.sample_posterior(n_iter=10)
        self.sample_prior(n_iter=10)

        f, axarr = plt.subplots(1,3)
        axarr[2].matshow(gibbs.prior)
        axarr[2].set_title("Prior")
        axarr[2].get_xaxis().set_visible(False)
        axarr[2].get_yaxis().set_visible(False)
        axarr[1].matshow(gibbs.posterior)
        axarr[1].get_xaxis().set_visible(False)
        axarr[1].get_yaxis().set_visible(False)
        axarr[1].set_title("Posterior")
        axarr[0].matshow(gibbs.evidence)
        axarr[0].set_title("Evidence")
        axarr[0].get_xaxis().set_visible(False)
        axarr[0].get_yaxis().set_visible(False)
        plt.show()



if __name__ == "__main__":
    gibbs = GibbsDenoising(img_path="Bootstrap/img/ImgPlain.png", ising_J=4, ev_sigma=0.5)
    gibbs.show_gibbs_denoising()


