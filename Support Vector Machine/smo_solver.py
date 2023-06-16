import numpy as np


def random_index(i, n_samples):
    j = i
    while j == i:
        j = np.random.randint(0, n_samples)
    return j


def compute_error(xi, yi, X, y, alphas, b, kernel):
    fx = np.sum(alphas * y * kernel(X, xi)) + b
    return fx - yi


def SMO(X, y, C, kernel, tol=1e-3, max_passes=5):
    n_samples = X.shape[0]

    # Initialize alphas and b
    alphas = np.zeros(n_samples)
    b = 0.0

    passes = 0
    while passes < max_passes:
        num_changed_alphas = 0
        for i in range(n_samples):
            Ei = compute_error(X[i], y[i], X, y, alphas, b, kernel)
            if (y[i]*Ei < -tol and alphas[i] < C) or (y[i]*Ei > tol and alphas[i] > 0):
                j = random_index(i, n_samples)
                Ej = compute_error(X[j], y[j], X, y, alphas, b, kernel)

                old_alpha_i = alphas[i]
                old_alpha_j = alphas[j]

                # Compute L and H
                if y[i] != y[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[i] + alphas[j] - C)
                    H = min(C, alphas[i] + alphas[j])
                if L == H:
                    continue

                eta = 2 * kernel(X[i], X[j]) - kernel(X[i],
                                                      X[i]) - kernel(X[j], X[j])
                if eta >= 0:
                    continue

                alphas[j] -= y[j]*(Ei - Ej)/eta
                alphas[j] = min(H, max(L, alphas[j]))

                if abs(alphas[j] - old_alpha_j) < tol:
                    continue
                alphas[i] += y[i]*y[j]*(old_alpha_j - alphas[j])

                b1 = b - Ei - y[i]*(alphas[i] - old_alpha_i)*kernel(X[i],
                                                                    X[i]) - y[j]*(alphas[j] - old_alpha_j)*kernel(X[i], X[j])
                b2 = b - Ej - y[i]*(alphas[i] - old_alpha_i)*kernel(X[i],
                                                                    X[j]) - y[j]*(alphas[j] - old_alpha_j)*kernel(X[j], X[j])

                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0

                num_changed_alphas += 1

        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0

    # Set support vectors
    is_sv = alphas > 1e-4
    support_vectors = X[is_sv]
    support_alphas = alphas[is_sv]
    support_ys = y[is_sv]

    return support_vectors, support_alphas, support_ys, b
