# sampling

# FOR
# GENERATION: sampling algorithm directly controls creativity, coherence, and diversity
# TRAINING: sampling reweights samples to reduce gradient variance in reinforcement learning
# ESTIMATION: Monte Carlo estimation approximates all of these by averaging over samples
# EXPLORATION: explore posterior distributions or other stuffs


import math
import random

# Uniform Random Sampling
    # all intervals have the same probability

def sample_uniform(a, b):
    return a + (b - a) * random.random() # random from [0.0, 1.0]


# Inverse CDF Method
    # U = generate uniform random number
    # apply the inverse CDF (cumulative distribution fucntion)
    # res = x = follows the desired target probability distribution

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam

# Rejection Sampling = a Monte Carlo technique
    # target distribution = p(x)
    # proposal distribution = q(x)
    # Bound: M such that p(x) <= M * q(x) for all x

    # sample x ~ q(x)
    # sample u ~ Uniform(0, 1)
    # if u < p(x) / (M * q(x)), accept x
    # otherwise, reject and go to step 1

    # Acceptance rate = 1 / M

def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf):
            return x

# Importance Sampling = a Monte Carlo technique
    # Goal: estimate E_p[f(x)] = integral of f(x) * p(x) dx
    # Rewrite: E_p[f(x)] = integral of f(x) * (p(x) / q(x)) * q(x) dx = E_q[f(x) * w(x)]
    # whre w(x) = p(x) / q(x) are the importance weights
    # Estimator: E_p[f(x)] ~ (1 / N) * sum(f(x_i) * w(x_i)) where x_i ~ q(x)

def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_pdf()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w

    return total / n

# Monte Carlo Estimation = discovered by Stanislaw Ulm at Los Alamos
    # Goal: estiamte I = integral of g(x) dx over domain D

    # Method: Sample x_1, x_2, ... x_N uniformly from D
    # I ~ (Volume of D / N) * sum(g(x_i))

    # Error: O(1 / sqrt(N))

def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)

        if x**2 + y**2 <= 1:
            inside += 1

    return 4 * inside / n


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

def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0

    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))

        if math.log(random.random()) < log_alpha:
            x = x_new

        if i >= burn_in:
            samples.append(x)

    return samples


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

def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)

        if i > burn_in:
            samples.append((x, y))
    
    return samples

    
# Temperature Sampling (used in llms)
    # p_i = exp(z_i / T) / sum(exp(z_j / T))

    # T = 1.0: standard softmax (original distribution)
    # T -> 0:  argmax (deterministic, always picks highest logit)
    # T -> inf: uniform (all tokens equally likely)
    # T < 1.0: sharpens the distribution (more confident, less diverse)
    # T > 1.0: flattens the distribution (less confident, more diverse)


def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)

    return [e / total for e in exps]

def sample_from_probs(probs):
    r = random.random()
    cumsum = 0.0
    for i, p in enumerate(probs):
        cumsum += p
        if r <= cumsum:
            return i
    return len(probs) - 1

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)

# Top-k Sampling
    # compute softmax probabilities for all V tokens
    # sort tokens by probability
    # keep only top k tokens
    # renormalize: p_i' = p_i / sum(p_j for j in top_k)
    # sample from the renormalized distribution

# Top-p (Nucleus) Sampling
    # compute softmax probabilities for all V tokens
    # sort tokens by probability
    # find smallest k such that sum of top-k probabilities >= p
    # keep only those
    # renormalize and sample

def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[: k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    consum = 0
    selected = []

    for token_idx, prob in indexed:
        consum += prob
        selected.append((token_idx, prob))

        if consum >= p:
            break

    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)

    return selected[idx][0]

