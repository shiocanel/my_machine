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