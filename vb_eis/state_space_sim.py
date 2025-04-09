import jax.numpy as jnp
import jax

def gammasgn(x):
    """Sign of the gamma function.
    JAX implementation of :func:`scipy.special.gammasgn`.
    Args:
    x: arraylike, real valued.
    Returns:
    array containing 1.0 where gamma(x) is positive, and -1.0 where
    gamma(x) is negative.
    See Also:
    :func:`jax.scipy.special.gamma`
    """
    floor_x = jax.lax.floor(x)
    return jnp.where((x > 0) | (x == floor_x) | (floor_x % 2 == 0), 1.0, -1.0)

def tf_binom(x, y):
  """
  Calculates the binomial coefficient using gamma functions.

  Args:
    x: A jax.numpy array of floats.
    y: A jax.numpy array of floats.

  Returns:
    A jax.numpy array of floats containing the binomial coefficients.
  """
  r = gammasgn(x + 1) / gammasgn(y + 1) / gammasgn(x - y + 1)
  return r * jnp.exp(jax.lax.lgamma(x + 1) - (jax.lax.lgamma(y + 1) + jax.lax.lgamma(x - y + 1)))

def recurrent_tf_binom(prev_coef, x, y):
  """
  Calculates the next binomial coefficient in a recursive manner.

  Args:
    prev_coef: The previous binomial coefficient.
    x: The first parameter of the binomial coefficient.
    y: The second parameter of the binomial coefficient.

  Returns:
    The next binomial coefficient.
  """
  return (x - y) / (y + 1) * prev_coef

def generate_state_space_tensor(Rs, R, C, alfa, fs, num_obs):
    """
    Generates the state-space matrices for the system.
    
    Args:
    Rs: A list of resistances.
    R: A resistance value.
    C: A list of capacitances.
    alfa: A list of poles.
    fs: The sampling frequency.
    num_obs: The number of observations.
    
    Returns:
    A tuple containing the state-space matrices (A, bl, m, d, num_obs).
    """
    # assert len(R) == len(C), "Same shape for R and C"
    # assert len(alfa) == len(C), "Same shape for alfa and C"
    
    Ts = 1.0 / fs
    n = len(R)  # Length of elements

    TsA = Ts**alfa
    
    # Define A using slicing and vectorized operations
    A_0 = alfa - TsA / (R * C)

    # js = jnp.empty((1,num_obs-1))
    
    # # Prepare a vector that corresponds to j+1 at each j Eq. (9)
    # js.at[0].set(jnp.arange(1,num_obs) + 1)
    
    # #Expand the vector j+1 to have columns equal to the number of poles - Expand does NOT allocate extra mem
    # # jj = js.expand(len(alfa),-1).T
    # jj = jnp.expand_dims(js, axis=0)
    jj = jnp.tile(jnp.expand_dims(jnp.arange(0,num_obs,dtype=jnp.float32) + 1 ,1),[1,len(alfa)])
    
    
    
    # alf = jnp.empty((1,len(alfa)))
    # alf.at[0] = alfa
    # aa = alf.expand(num_obs-1,-1)
    aa = jnp.tile(jnp.expand_dims(alfa,1).T,[num_obs,1])
    # print(aa.shape,aa)
    # print(aa.shape)
    # print(jnp.expand_dims(alfa,1).shape)

    X = ( (-1)**(jj-1) )*( tf_binom(aa, (jj) ) )

    mask = jnp.expand_dims(jnp.eye(1,num_obs)[0],-1)
    inv_mask = 1 - mask
    A = mask * A_0 + inv_mask * X

    # ppp = mask * A_0
    # print(mask.shape, A_0.shape, ppp.shape, X.shape, inv_mask.shape)
    
    # X = tf_binom(jnp.expand_dims(alfa, axis=0), jnp.arange(1, num_obs,dtype=jnp.float32)[:, None])
    # return inv_mask * X
    # A_rest = jnp.expand_dims(-1, axis=0) * jnp.outer(jnp.arange(1, num_obs) + 1, tf_binom(jnp.expand_dims(alfa, axis=0), jnp.arange(1, num_obs,dtype=jnp.float32)[:, None]))
    
    # print(A_0.shape, A_rest.shape)
    # A = jnp.concatenate((A_0[None, :], A_rest), axis=0)
    
    # Define bl using vectorized operations
    bl = TsA / C
    
    # Define m and d using broadcasting
    m = jnp.ones(n)
    d = jnp.array(Rs)
    
    return A, bl, m, d, num_obs


def generate_state_space_tensor_rtau(Rs, R, taus, alfa, fs, num_obs):
    """
    Generates the state-space matrices for the system.
    
    Args:
    Rs: A list of resistances.
    R: A resistance value.
    taus: A list of time constants.
    alfa: A list of poles.
    fs: The sampling frequency.
    num_obs: The number of observations.
    
    Returns:
    A tuple containing the state-space matrices (A, bl, m, d, num_obs).
    """
    # assert len(R) == len(C), "Same shape for R and C"
    # assert len(alfa) == len(C), "Same shape for alfa and C"
    
    Ts = 1.0 / fs
    n = len(R)  # Length of elements

    TsA = Ts**alfa
    
    # Define A using slicing and vectorized operations
    A_0 = alfa - TsA / (jnp.exp(taus))

    # js = jnp.empty((1,num_obs-1))
    
    # # Prepare a vector that corresponds to j+1 at each j Eq. (9)
    # js.at[0].set(jnp.arange(1,num_obs) + 1)
    
    # #Expand the vector j+1 to have columns equal to the number of poles - Expand does NOT allocate extra mem
    # # jj = js.expand(len(alfa),-1).T
    # jj = jnp.expand_dims(js, axis=0)
    jj = jnp.tile(jnp.expand_dims(jnp.arange(0,num_obs,dtype=jnp.float32) + 1 ,1),[1,len(alfa)])
    
    
    
    # alf = jnp.empty((1,len(alfa)))
    # alf.at[0] = alfa
    # aa = alf.expand(num_obs-1,-1)
    aa = jnp.tile(jnp.expand_dims(alfa,1).T,[num_obs,1])
    # print(aa.shape,aa)
    # print(aa.shape)
    # print(jnp.expand_dims(alfa,1).shape)

    X = ( (-1)**(jj-1) )*( tf_binom(aa, (jj) ) )

    mask = jnp.expand_dims(jnp.eye(1,num_obs)[0],-1)
    inv_mask = 1 - mask
    A = mask * A_0 + inv_mask * X

    # ppp = mask * A_0
    # print(mask.shape, A_0.shape, ppp.shape, X.shape, inv_mask.shape)
    
    # X = tf_binom(jnp.expand_dims(alfa, axis=0), jnp.arange(1, num_obs,dtype=jnp.float32)[:, None])
    # return inv_mask * X
    # A_rest = jnp.expand_dims(-1, axis=0) * jnp.outer(jnp.arange(1, num_obs) + 1, tf_binom(jnp.expand_dims(alfa, axis=0), jnp.arange(1, num_obs,dtype=jnp.float32)[:, None]))
    
    # print(A_0.shape, A_rest.shape)
    # A = jnp.concatenate((A_0[None, :], A_rest), axis=0)
    
    # Define bl using vectorized operations
    bl = TsA * R / jnp.exp(taus)
    
    # Define m and d using broadcasting
    m = jnp.ones(n)
    d = jnp.array(Rs)
    
    return A, bl, m, d, num_obs

def generate_mask(shape):
    rows, cols = shape
    # print(rows,cols)
    ones_row = jnp.ones(cols)
    zeros_matrix = jnp.zeros((rows - 1, cols))
    return jnp.vstack((ones_row, zeros_matrix))

jgen = jax.jit(generate_state_space_tensor, static_argnames=['num_obs'])
jgen_taus = jax.jit(generate_state_space_tensor_rtau, static_argnames=['num_obs'])


@jax.jit
def forward_sim(A, bl, m, d, x_init, I, mask):
    val = jnp.copy(x_init)
    # j = 167
    # val = x_init
    # xkp1 = jnp.sum(A*jnp.roll(val,j,axis=0),axis=0) + bl*I[j]
    # xkp1 = jnp.roll(mask,j+1, axis=0) *  xkp1
    # val = val + xkp1
    
    def step(j, val):
    #     xkp1 = jnp.sum(A*jnp.roll(val,j,axis=0),axis=0) + bl*I[j]
        xkp1 = jnp.sum(A*jnp.roll(jnp.flipud(val),j+1,axis=0),axis=0) + bl*I[j]
        xkp1 = jnp.roll(mask,j+1, axis=0) *  xkp1
        return val + xkp1
    
    
    
    res = jax.lax.fori_loop(0,A.shape[0]-1,step, x_init)
    yk_lim = jnp.expand_dims(m,1).T * res
    # print(yk_lim.sum(axis=-1).shape)
    return yk_lim.sum(axis=-1) + d*I
    # yk_lim =  m*res + jnp.expand_dims(d*I, -1)

    return yk_lim, res