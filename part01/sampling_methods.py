# sampling

# FOR
# GENERATION: sampling algorithm directly controls creativity, coherence, and diversity
# TRAINING: sampling reweights samples to reduce gradient variance in reinforcement learning
# ESTIMATION: Monte Carlo estimation approximates all of these by averaging over samples
# EXPLORATION: explore posterior distributions or other stuffs



# Uniform Random Sampling
    # all intervals have the same probability

# Inverse CDF Method
    # U = generate uniform random number
    # apply the inverse CDF (cumulative distribution fucntion)
    # res = x = follows the desired target probability distribution

# Rejection Sampling = a Monte Carlo technique
    # target distribution = p(x)
    # proposal distribution = q(x)
    # Bound: M such that p(x) <= M * q(x) for all x

    # sample x ~ q(x)
    # sample u ~ Uniform(0, 1)
    # if u < p(x) / (M * q(x)), accept x
    # otherwise, reject and go to step 1

    # Acceptance rate = 1 / M

# Importance Sampling = a Monte Carlo technique
    # Goal: estimate E_p[f(x)] = integral of f(x) * p(x) dx
    # Rewrite: E_p[f(x)] = integral of f(x) * (p(x) / q(x)) * q(x) dx = E_q[f(x) * w(x)]
    # whre w(x) = p(x) / q(x) are the importance weights
    # Estimator: E_p[f(x)] ~ (1 / N) * sum(f(x_i) * w(x_i)) where x_i ~ q(x)


# Monte Carlo Estimation = discovered by Stanislaw Ulm at Los Alamos
    # Goal: estiamte I = integral of g(x) dx over domain D

    # Method: Sample x_1, x_2, ... x_N uniformly from D
    # I ~ (Volume of D / N) * sum(g(x_i))

    # Error: O(1 / sqrt(N))


# O(n^2)
def solve_pi(points):
    len = 2 / points
    x = -1
    y = -1


    count = 0
    while x <= 1:
        y = -1
        while y <= 1:
            if x ** 2 + y ** 2 <= 1:
                count += 1
            y += len
        x += len

    pi = 4 * (count / (points ** 2))
    return pi

print(f"pi = {solve_pi(1000)}")


# Markov Chain Monte Carlo (MCMC)
    # start with some x_0
    # for t = 1, 2, .. T
        # propose x' ~ q(x' | x_t)
        # compute acceptance ratio: alpha = [p'(x) * q(x_t | x')] / [p(x_t) * q(x' | x_t)]
        # accept with probability min(1, alpha)

    # discard first B samples (burn-in)
    # return remaining samples


# Gibbs Sampling (MCMC algorithm)
    # Input: initial state x = (x_1, ..., x_d), number of iterations T,
    #        burn-in length B, and the full conditional distributions.
    # for t = 1, 2, ..., T:
        # for j = 1, 2, ..., d:
            # Sample x_j from p(x_j | x_1, ..., x_(j-1), x_(j+1), ..., x_d)
        # if t > B:
            # Save the current state x as a sample.
    # Return the saved states. Each coordinate is updated using the latest
    # available values, producing samples from the joint distribution.
        
# Temperature Sampling (used in llms)
    # p_i = exp(z_i / T) / sum(exp(z_j / T))

    # T = 1.0: standard softmax (original distribution)
    # T -> 0:  argmax (deterministic, always picks highest logit)
    # T -> inf: uniform (all tokens equally likely)
    # T < 1.0: sharpens the distribution (more confident, less diverse)
    # T > 1.0: flattens the distribution (less confident, more diverse)

# Top-k Sampling
    