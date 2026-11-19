# Read the concept (20 min)
# Rebuild the core from notes (30 min)
# Do one exercise (20 min)
# Write 3-5 flashcards (10 min)


# logistic regression = linear regression + sigmoid => values from 0.0 to 1.0
# Binary Cross-Entropy Loss = The loss to train
# Do Gradient descent on it
# take a descision

import random
import math


# 1. sigmoid fucntion
def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


# 2. generate 200 random samples with 2 centers on 2 and 5 where 2 is 0 and 5 is 1
N = 200
X = []
y = []

for _ in range(N // 2):
    X.append([random.gauss(2, 1), random.gauss(2, 1)])
    y.append(0)

for _ in range(N // 2):
    X.append([random.gauss(5, 1), random.gauss(5, 1)])
    y.append(1)

# 3. shuffle data
combined = list(zip(X, y))
random.shuffle(combined)

X, y = zip(*combined)
X = list(X)
y = list(y)

# 4. print data
print(f"Generated {N} samples")
print(f"Class 0 with center (2, 2) and class 1 in (5, 5)")
for i in range(5):
    print(f"Features: [{X[i][0]:.2f}, {X[i][1]:.2f}], Label: {y[i]}")

# Logistic Regression
# 1. weights, bias = 0.0, lr, loss_history
# 2. predict_proba(): z = W + X and predict() with 1 and 0
# 3. compute_loss(): binary cross entropy loss
# 4. fit()
#    for each epoch:
#       for each sample, compute z = w·x + b, then p = sigmoid(z)
#       error = p - y
#       accumulate the gradient: dw += error * x, db += error
#       average the gradients, then update w -= lr*dw, b -= lr*db
