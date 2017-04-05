function J = ComputeCost(X, Y, W, b, lambda, K)

J = 0;
[~, N] = size(X);
P = EvaluateClassifier(X, W, b, K); 
J = -sum(log(sum(Y.*P, 1)))/N+lambda*sumsqr(W);