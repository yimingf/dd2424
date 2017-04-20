function J = ComputeCostComputeCost(X, Y, W1, b1, W2, b2, lambda, K)

[~, N] = size(X);
P = EvaluateClassifier(X, W1, b1, W2, b2, K);
J = -sum(log(sum(Y.*P, 1)))/N+lambda*(sumsqr(W1)+sumsqr(W2));