import numpy as np
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import invwishart
import matplotlib.animation as animation
import time
import math
import logging
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/Cellar/ffmpeg/3.3.4/bin/ffmpeg'

DEFAULT_PRIOR_COV_SCALE = 9
DEFAULT_PRIOR_MEAN_STRENGTH = 2
DEFAULT_PRIOR_SCALE_DF = 10

LOG_FILENAME = 'gibbsConjugate.log'
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG, filemode='w')

class gaussian_cluster(object):
    def __init__(self, prior_mean, prior_mean_strength, prior_cov_scale, prior_cov_df, prior_mixing):
        assert len(prior_cov_scale.shape) == 2, "Scale matrix must be two-dimensional"
        assert prior_cov_scale.shape[0] == prior_cov_scale.shape[1], "Scale matrix must be square"
        assert len(prior_mean) == len(prior_cov_scale), "Prior mean vector not same dimension as prior scale"
        assert prior_mean_strength > 0, "Prior mean strength (kappa) must be positive"
        assert prior_cov_df >= prior_cov_scale.shape[0], "Prior degrees of freedom must be at least data dimension."

        ### Mean and Covariance of Gaussian cluster
        self.cov = invwishart.rvs(df=prior_cov_df, scale=prior_cov_scale, size=1)
        self.mean = np.random.multivariate_normal(mean=prior_mean, cov=1.0/float(prior_mean_strength) * self.cov)

        self.prior_mean = prior_mean
        self.prior_mean_strength = prior_mean_strength
        self.prior_cov_scale = prior_cov_scale
        self.prior_cov_df = prior_cov_df

        ### Responsibility:
        self.prior_mixing = prior_mixing # alpha_k
        self.post_mixing = prior_mixing # pi_k
        self.post_count = 0

class Gibbs_GMM(object):
    def __init__(self, nClusters, data=None, N=None, prior_means=None, prior_mean_strength=None,
                 prior_cov_scale=None, prior_cov_df=None, prior_mixings=None):

        self.nClusters = nClusters

        ############### DATA INIT #############
        if data is None:
            assert N is not None, "Specify size of synthetic dataset (N)."
            self.data = self.generate_data(N)
        else:
            self.data = data
        assert self.data.shape[0] > nClusters, "Data size must be larger than number of clusters."
        self.nObs = self.data.shape[0]

        ############## CLUSTER MEAN INITS ###############
        if prior_means is None:
            prior_means = []
            for k in range(nClusters):
                rdmRows = np.random.randint(low=0, high=self.data.shape[0]-1, size=int(self.data.shape[0]/nClusters))
                prior_means.append(np.mean(self.data[rdmRows,:], axis=0))
        else:
            assert isinstance(prior_means, list), "Provide prior means as list"
            assert len(prior_means) == nClusters

        if prior_mean_strength is None:
            prior_mean_strength = DEFAULT_PRIOR_MEAN_STRENGTH

        ############### CLUSTER SCALE INITS ################
        if prior_cov_scale is None:
            prior_cov_scale = np.array([[DEFAULT_PRIOR_COV_SCALE,0],[0,DEFAULT_PRIOR_COV_SCALE]])

        if prior_cov_df is None:
            prior_cov_df = DEFAULT_PRIOR_SCALE_DF

        ################ CLUSTER PREVALENCE INITS #########
        if prior_mixings is None:
            prior_mixings = [1.0/float(nClusters) for k in range(nClusters)]
        assert len(prior_mixings) == nClusters, "There must be as many mixing parameters as clusters"

        ##########################################
        self.clusters = dict([(k, gaussian_cluster(prior_mean=prior_means[k], prior_mean_strength=prior_mean_strength,
                                                   prior_cov_scale=prior_cov_scale, prior_cov_df=prior_cov_df,
                                                   prior_mixing=prior_mixings[k])) \
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

        for cluster in self.clusters.values():
            mean = cluster.mean
            cov = cluster.cov
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
            p_indicator = [clusters[k].post_mixing * self.gaussian_2d(x, clusters[k].mean, clusters[k].cov) \
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

    def _fc_gaussians(self):
        """
        Samples the next mean and covariance for each cluster from the Normal Inverse Wishart with posterior parameters
        :return:
        """
        clusters = self.clusters
        data = self.data
        indicators = self.indicators

        for k in range(self.nClusters):
            cluster = clusters[k]
            data_in_cluster = data[indicators[:,k] == 1.0]
            mean_data_in_cluster = np.mean(data_in_cluster, axis=0)
            assert cluster.post_count == len(data_in_cluster), \
                "Cluster count incorrect: %d <> %d" % (cluster.post_count, len(data_in_cluster))

            # Update parameters of normal inverse wishart distribution
            post_strength = cluster.prior_mean_strength + cluster.post_count
            post_mean = (cluster.prior_mean_strength*cluster.prior_mean +
                         cluster.post_count*np.mean(data_in_cluster, axis=0))/float(post_strength)
            post_scale = cluster.prior_cov_scale + self.emp_cov(data_in_cluster) + \
                         (cluster.prior_mean_strength*cluster.post_count)/(cluster.prior_mean_strength+cluster.post_count)*\
                         np.outer((mean_data_in_cluster-cluster.prior_mean),(mean_data_in_cluster-cluster.prior_mean))

            #post_scale = cluster.prior_cov_scale + (cluster.post_count-1.0)*self.emp_cov(data_in_cluster)+\
            #                cluster.prior_mean_strength*np.outer(cluster.prior_mean, cluster.prior_mean)-\
            #                post_strength*np.outer(post_mean, post_mean)
            post_df = cluster.prior_cov_df + cluster.post_count

            post_mean, post_strength, post_scale, post_df = self._post_niw_params(cluster=cluster,
                                                                                  data_in_cluster=data_in_cluster)

            # Sample covariance
            cluster.cov = invwishart.rvs(df=post_df, scale=post_scale, size=1)

            # Sample mean
            cluster.mean = np.random.multivariate_normal(mean=post_mean, cov=1.0/float(post_strength)*cluster.cov)


            logging.info("-----------------")
            logging.info("\t\tCluster {}".format(k))
            logging.info("\t\t\tCount: {}".format(cluster.post_count))
            logging.info("\t\t\tMean: {}".format(cluster.mean))
            logging.info("\t\t\tCov: {}".format(cluster.cov))
            logging.info("\t\t\tPost cov scale:{}".format(post_scale))
            logging.info("\t\t\tPost cov df:{}".format(post_df))

            assert len(cluster.mean) == len(cluster.cov),"len(mean) <> len(cov) for cluster %d" % k
            self.clusters[k] = cluster

    @staticmethod
    def emp_cov(data):
        """
        Computes the empirical covariance matrix for an observation matrix of shape N x D
        :param data: numpy array of shape N x D where N is number of observations and D is dimension of each obs
        :return: (data - mean(data))^T (data - mean(data))
        """
        assert len(data.shape) == 2
        assert data.shape[0] > 1
        mean_data = np.vstack((np.mean(data,axis=0) for _ in range(data.shape[0])))
        data -= mean_data
        return 1.0/float(data.shape[0] - 1) * np.dot(np.transpose(data), data)

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

    def _update_log_lik(self):
        """
        Compute log p(X, z| alpha, beta) to see how good the fit is after the current iteration
        """
        data = self.data
        clusters = self.clusters
        indicators = self.indicators

        log_lik = 0
        for k in range(self.nClusters):
            data_in_cluster = data[indicators[:,k] == 1.0]
            cluster = clusters[k]

            # Compute p(X_k| beta)
            d = data_in_cluster.shape[1]
            kappa_N = cluster.prior_mean_strength + cluster.post_count
            kappa_0 = cluster.prior_mean_strength
            nu_N = cluster.prior_cov_df + cluster.post_count
            nu_0 = cluster.prior_cov_df
            cov_scale_N =
            self._niw_normalisation(dim=data_in_cluster.shape[1], prior_strength=cluster.prior_mean_streng)


    def _post_niw_params(self, cluster, data_in_cluster):
        """
        Computes Normal Inverse Wishart parameters for
        p(mean_of_cluster, Cov_of_cluster| data_in_cluster) = NIW(mean_of_cluster, Cov_of_cluster| params)
        for a given cluster.
        :param cluster: a Cluster object
        :param data_in_cluster: the data assigned to that cluster of size (N_k) x D
        :return: m_N, kappa_N, scale_N, nu_N
        """
        mean_data_in_cluster = np.mean(data_in_cluster, axis=0)
        post_strength = cluster.prior_mean_strength + cluster.post_count
        post_mean = (cluster.prior_mean_strength*cluster.prior_mean +
                     cluster.post_count*np.mean(data_in_cluster, axis=0))/float(post_strength)
        post_scale = cluster.prior_cov_scale + self.emp_cov(data_in_cluster) + \
                     (cluster.prior_mean_strength*cluster.post_count)/(cluster.prior_mean_strength+cluster.post_count)*\
                     np.outer((mean_data_in_cluster-cluster.prior_mean),(mean_data_in_cluster-cluster.prior_mean))
        return post_mean, \
               cluster.prior_mean_strength + cluster.post_count, \
               post_scale, \
               cluster.prior_cov_df + cluster.post_count

    def _niw_normalisation(self, dim, prior_strength, cov_df, cov_scale):
        """
        Computes the normalisation constant of the Normal Inverse Wishart distribtuion
        See e.g. Murphy, K.: Machine Learning - A Probabilistic Perspective, p.133
        :param dim: dimension of mean (D)
        :param prior_strength: strength of prior mean (kappa)
        :param cov_df: degrees of freedom (nu)
        :param cov_scale: scale matrix (S)
        """
        dim = float(dim)
        cov_df = float(cov_df)
        prior_strength = float(prior_strength)
        return 2.0**(dim*cov_df/2.0) * \
                math.gamma(cov_df/2.0)*(2.0*np.pi/prior_strength)**(dim/2.0)* \
                np.linalg.det(cov_scale)**(-cov_df/2.0)

    def train(self, nIter, plot=False):
        for t in range(nIter):
            logging.info("----------------------------------------------------")
            logging.info("Iteration %d" % t)
            self._fc_indicators()
            self._fc_mixings()
            self._fc_gaussians()

            self._update_log_lik()

            if plot:
                self.update_plot(t)
        if plot:
            ani = animation.ArtistAnimation(self.f, self.ims, interval=500, blit=True,
                            repeat_delay=100)

            self.ax.set_xlim(-10,10)
            self.ax.set_ylim(-10,10)
            plt.show()
            #ani.save('GibbsGMM_2.mp4', fps=2)
            #ani.save('GibbsGMM_2.mp4', writer = 'mencoder', fps=2)
            FFwriter = animation.FFMpegWriter()
            ani.save('GibbsGMM_2.mp4', writer=FFwriter, fps=2)

if __name__ == "__main__":
    prior_means = [np.array([3,0]), np.array([0,-2]), np.array([-3,0])]
    gmm = Gibbs_GMM(nClusters=3, N=200, prior_means=prior_means)
    gmm.train(nIter=25, plot=True)
    plt.show()

