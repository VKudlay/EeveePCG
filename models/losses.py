import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

'''
Pool of functions related to loss evaluation. 
NOTE: Latents assumed to be in form [z, mu, sd].
'''

def dist_matrix(pnts_t, pnts_p):
    """Gets matrix of squared distances
        Modified from this answer: https://stackoverflow.com/a/54706262/5003309
    
    :param pnts_t: list of true points
    :paran obts_p: list of predicted points
    """
    num_t = K.int_shape(pnts_t)[0]
    num_p = K.int_shape(pnts_p)[0]

    # if 1024 samples, expand both to (1024^2, 3) dimensions for easy element-wise product
    pnts_t = tf.tile(tf.expand_dims(pnts_t, 1), [1, 1, num_p])
    pnts_t = tf.cast(tf.reshape(pnts_t, [-1, 3]), tf.float64)
    pnts_p = tf.cast(tf.tile(pnts_p, [num_t, 1]), tf.float64)

    dists_mat = K.sum(K.square(tf.subtract(pnts_t, pnts_p)), axis=1) # compute element-wise L2 norm
    dists_mat = tf.reshape(dists_mat, [num_t, num_p])

    dists_mat_upper = tf.linalg.band_part(dists_mat, 0, -1)
    dists_mat_symm = dists_mat_upper + tf.transpose(dists_mat_upper)
    dists_mat_symm = tf.linalg.set_diag(dists_mat_symm, tf.linalg.diag_part(dists_mat))  

    return dists_mat_symm


def kl_loss(y_true, y_pred, latents):
    '''Computes the Kullback-Leibler Divergence (KLD) based on latent distributions
        Assumes (zs, means, variances) latents format
    '''
    _, mu, var = latents
    kl_loss = 1 + var - tf.square(mu) - tf.exp(var)
    out = tf.reduce_sum(kl_loss, axis=1) * -0.5
    return K.mean(out)


def chamfer_dists(pnts, op1 = K.sum, op2 = lambda a,b: a+b):
    '''Computes chamfer distance 
        D = op2(op1(true_to_pred_min_dists), op1(pred_to_true_min_dists))
        By default, (sum of min dists from p to t) + (sum of min dists from t to p)
        where p is set of predictions and t is the set of true points.

    :param pnts: [list of true points, list of predicted points]
    Modified from this answer: https://stackoverflow.com/a/54706262/5003309
    '''
    pnts_t, pnts_p = pnts
    dists_mat = dist_matrix(pnts_t, pnts_p)
    dist_t_to_p = op1(K.min(dists_mat, axis=0))
    dist_p_to_t = op1(K.min(dists_mat, axis=1))
    return op2(dist_p_to_t, dist_t_to_p)

def chamfer_loss(y_true, y_pred, latents=None):
    '''Calculate the chamfer distance, use euclidean metric
        Modified from this answer: https://stackoverflow.com/a/54706262/5003309
    '''    
    # Don't know why, but outputting to f64 and casting to f32 seems to fix some conversion bug...
    out = tf.map_fn(chamfer_dists, elems=(y_true, y_pred), fn_output_signature=tf.float64)
    out = tf.cast(out, tf.float32) 
    return K.mean(out)
    

def coverage(pnts, varc = False):
    '''[NOT USED] Computes custom coverage heuristic'''
    pnts_t, pnts_p = pnts
    dists_mat = dist_matrix(pnts_t, pnts_p)
    map_p_to_t_unq = tf.unique(K.argmin(dists_mat, axis=0)).y
    coverage = len(map_p_to_t_unq)/K.int_shape(pnts_t)[0]
    if varc: 
        dists_mat_p = dist_matrix(pnts_p, pnts_p)
        dists_mat_p = tf.linalg.set_diag(dists_mat_p, tf.ones(dists_mat_p.shape[0:-1]))
        dist_p_to_p = K.min(dists_mat_p, axis=0)
        vc_factor = K.std(dist_p_to_p) / K.mean(dist_p_to_p)
    else: 
        vc_factor = 1
    return vc_factor * (0.1/coverage - 0.1 )


def coverage_loss(y_true, y_pred, latents=None):
    '''[NOT USED] Computes custom coverage loss heuristic: 
        what % of the closest points to the prediction set are unique?
    '''
    out = tf.map_fn(coverage, elems=(y_true, y_pred), fn_output_signature=tf.float64)
    return K.mean(out)

def consistency_loss(y_true, y_pred, latents):
    '''[NOT USED] MSE-like operation on latents: https://arxiv.org/pdf/1901.09394.pdf'''
    return np.var(latents[0])
    

def gradient_penalty(f, real, fake, mode):
    '''Gradient penalty function; can be used to get wgan and dragan gradient penalty
        https://colab.research.google.com/drive/1zAUGSNFENZ_iU7m8YkiniG8seUNqbYT5
    
    :param mode: One of 'dragan' and 'wgan'. Returns 0 otherwise
    '''
    def _gradient_penalty(f, real, fake=None):
        def _interpolate(a, b=None):
            if b is None:   # interpolation in DRAGAN
                beta = tf.random.uniform(shape=tf.shape(a), minval=0., maxval=1.)
                b = a + 0.5 * tf.math.reduce_std(a) * beta
            shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
            alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.shape)
            return inter

        x = _interpolate(real, fake)
        with tf.GradientTape() as t:
            t.watch(x)
            pred = f(x)
        grad = t.gradient(pred, x)
        norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
        return tf.reduce_mean((norm - 1.)**2)
    
    if mode == 'dragan': return _gradient_penalty(f, real)
    elif mode == 'wgan': return _gradient_penalty(f, real, fake)
    return 0


def dis_loss(y_fake, y_true, D):
    """Discriminator (critic) loss
    - encourages true data predictions to tend towards 1
    - encourages fake data predictions to tend towards 0
    """
    return 1 - K.mean(y_true) + K.mean(y_fake)

def gen_loss(y_fake, y_goal, G):
    """Generator loss
    - encourages fake data predictions to tend towards 1
    """
    return 1 - K.mean(y_fake)