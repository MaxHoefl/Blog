import numpy as np
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import invwishart
import matplotlib.animation as animation
import time

class cluster(object):
    def __init__(self, prior_mean, prior_m_cov, prior_cov_scale, prior_cov_df, prior_mixing):
        assert prior_cov_scale.shape[0] == prior_cov_scale.shape[1]
        ### Mean:
        self.post_mean = prior_mean # muk
        self.post_m_cov = prior_m_cov # Vk
        self.prior_m_mean = prior_mean # m0
        self.prior_m_cov = prior_m_cov # V0
        self.prior_m_prec = np.linalg.inv(self.prior_m_cov) # V0^{-1}

        ### Cov:
        self.post_cov = prior_cov_scale / (prior_cov_df - prior_cov_scale.shape[0] - 1)   # Sigma_k
        self.prior_cov_scale = prior_cov_scale # S_0
        self.post_cov_scale = prior_cov_scale # S_k
        self.prior_cov_df = prior_cov_df # nu_0
        self.post_cov_df = prior_cov_df # nu_k

        ### Responsibility:
        self.prior_mixing = prior_mixing # alpha_k
        self.post_mixing = prior_mixing # pi_k
        self.post_count = 0

class Gibbs_GMM(object):
    def __init__(self, nClusters, data=None, N=None, prior_means=None, prior_m_covs=None, prior_cov_scale=None, prior_cov_df=None, prior_mixings=None):

        self.nClusters = nClusters

        ##########################################
        if data is None:
            assert N is not None, "Specify size of synthetic dataset (N)."
            self.data = self.generate_data(N)
        else:
            self.data = data
        assert self.data.shape[0] > nClusters, "Data size must be larger than number of clusters."
        self.nObs = self.data.shape[0]

        ##########################################
        if prior_means is None:
            prior_means = []
            for k in range(nClusters):
                rdmRows = np.random.randint(low=0, high=self.data.shape[0]-1, size=int(self.data.shape[0]/nClusters))
                prior_means.append(np.mean(self.data[rdmRows,:], axis=0))
        else:
            assert isinstance(prior_means, list), "Provide prior means as list"
            assert len(prior_means) == nClusters
        self.prior_means = prior_means
        self.init_prior_means = prior_means

        ##########################################
        if prior_m_covs is None:
            prior_m_covs = []
            for k in range(nClusters):
                prior_m_covs.append(np.array([[10,0],[0,10]]))

        ##########################################
        if prior_cov_scale is None:
            prior_cov_scale = np.array([[10,0],[0,10]])
        self.prior_covs = prior_cov_scale

        ##########################################
        if prior_cov_df is None:
            prior_cov_df = 5
        self.prior_cov_df = prior_cov_df

        ##########################################
        if prior_mixings is None:
            prior_mixings = [1.0/float(nClusters) for k in range(nClusters)]
        assert len(prior_mixings) == nClusters, "There must be as many mixing parameters as clusters"
        self.prior_mixings = prior_mixings

        ##########################################
        self.clusters = dict([(k, cluster(prior_means[k], prior_m_covs[k], prior_cov_scale, prior_cov_df, prior_mixings[k]))
                              for k in range(nClusters)])
        self.indicators = np.zeros((self.nObs, nClusters))

        ##########################################
        plt.ion()
        self.f, self.ax = plt.subplots()
        self.ims = []
        #self.update_plot()

    def update_plot(self, count):

        delta = 0.025
        x = np.arange(-10.0, 10.0, delta)
        y = np.arange(-10.0, 10.0, delta)
        X, Y = np.meshgrid(x, y)

        Ztot = None
        im = None

        for k in range(self.nClusters):
            cluster = self.clusters[k]
            mean = cluster.post_mean
            cov = cluster.post_cov
            Z = mlab.bivariate_normal(X, Y, cov[0,0], cov[1,1], mean[0], mean[1], cov[0,1])
            if Ztot is None:
                Ztot = Z
            else:
                Ztot += Z
        im = self.ax.imshow(Ztot, animated=True ,aspect='auto',origin='lower',extent=(X.min(),X.max(),Y.min(),Y.max()))
        im2 = self.ax.scatter(x=self.data[:,0], y=self.data[:,1], c='black', s=20, alpha=0.5)
        self.ims.append([im,im2])


    def _fc_indicators(self):
        indicators = self.indicators
        clusters = self.clusters
        data = self.data
        for i in range(indicators.shape[0]):
            x = data[i,:]
            p_indicator = [clusters[k].post_mixing * self.gaussian_2d(x, clusters[k].post_mean, clusters[k].post_cov) \
                            for k in range(self.nClusters)]
            norm = np.sum(p_indicator)
            p_indicator = [p_indicator[k]/norm for k in range(self.nClusters)]
            indicators[i,:] = np.random.multinomial(n=1, pvals=p_indicator)
        self.indicators = indicators
        for k in range(self.nClusters):
            clusters[k].post_count = len(indicators[indicators[:,k] == 1,0])
        self.clusters = clusters

    def _fc_mixings(self):
        clusters = self.clusters
        alpha = [clusters[k].prior_mixing + clusters[k].post_count for k in range(self.nClusters)]
        post_mixings = np.random.dirichlet(alpha)

        checkSum = 0
        for k in range(self.nClusters):
            clusters[k].post_mixing = post_mixings[k]
            checkSum += clusters[k].post_mixing
        assert np.abs(checkSum - 1) < 1e-3, "Mixing proportions dont sum up to one."
        self.clusters = clusters


    def _fc_mean(self):
        clusters = self.clusters
        data = self.data
        for k in range(self.nClusters):
            cluster = clusters[k]
            post_prec = np.linalg.inv(cluster.post_cov) # Sigma_k^-1
            cluster.post_m_cov = np.linalg.inv(cluster.prior_m_prec + cluster.post_count * post_prec) # V_k

            tmp = np.linalg.solve(cluster.post_cov, np.sum(data[self.indicators[:,k]==1,:],axis=0)) + \
                  np.linalg.solve(cluster.prior_m_cov, cluster.prior_m_mean)
            cluster.post_mean = np.dot(cluster.post_m_cov, tmp)
            clusters[k] = cluster
        self.clusters = clusters


    def _fc_cov(self):
        data = self.data
        indicators = self.indicators
        clusters = self.clusters
        for k in range(self.nClusters):
            cluster = clusters[k]
            tmp = data[indicators[:,k] == 1,:]
            tmp = tmp - np.array([cluster.post_mean,]*len(tmp))
            cluster.post_cov_scale = cluster.prior_cov_scale + np.dot(tmp.transpose(), tmp)
            cluster.post_cov_df = cluster.prior_cov_df + cluster.post_count
            cluster.post_cov = invwishart.rvs(cluster.post_cov_df, cluster.post_cov_scale, size=1)
            clusters[k] = cluster

        self.clusters = clusters

    @staticmethod
    def gaussian_2d(x, mean, cov):
        dim = cov.shape[0]
        det = np.linalg.det(cov)
        norm = (det * (2 * np.pi) ** dim) ** (-0.5)
        dev = x - mean
        return norm * np.exp(-0.5 * np.dot(dev, np.linalg.solve(cov, dev)))

    @staticmethod
    def generate_data(N):
        # Create 2d data from 3 clusters
        c1_mean = np.array([-5,3])
        c2_mean = np.array([0,0])
        c3_mean = np.array([6,-2])

        c1_cov = np.array([[1,0],[0,1]])
        c2_cov = np.array([[1,0],[0,1]])
        c3_cov = np.array([[1,0],[0,1]])

        c1_obs = np.random.multivariate_normal(mean=c1_mean, cov=c1_cov, size=N)
        c2_obs = np.random.multivariate_normal(mean=c2_mean, cov=c2_cov, size=N)
        c3_obs = np.random.multivariate_normal(mean=c3_mean, cov=c3_cov, size=N)

        res =  np.concatenate([c1_obs, c2_obs, c3_obs])
        np.random.shuffle(res)
        return res

    def train(self, nIter, plot=False):
        for t in range(nIter):
            print "Iteration %d" % t
            self._fc_indicators()
            self._fc_mixings()
            self._fc_mean()
            self._fc_cov()

            if plot:
                self.update_plot(t)
        if plot:
            ani = animation.ArtistAnimation(self.f, self.ims, interval=500, blit=True,
                            repeat_delay=100)

            self.ax.set_xlim(-10,10)
            self.ax.set_ylim(-10,10)
            plt.show()
            #ani.save('GibbsGMM.mp4', writer = 'mencoder', fps=2)
            #ani.save('GibbsGMM.mp4', writer=writer)

if __name__ == "__main__":
    prior_means = [np.array([3,0]), np.array([0,-2]), np.array([-3,0])]
    gmm = Gibbs_GMM(nClusters=3, N=200, prior_means=prior_means)
    gmm.train(nIter=15, plot=True)
    plt.show()

