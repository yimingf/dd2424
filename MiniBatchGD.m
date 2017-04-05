function [Wstar, bstar] = MiniBatchGD(X, Y, W, b, lambda, K, d, eta)

P = EvaluateClassifier(X, W, b, K);
[grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda, K, d);

Wstar = W-eta*grad_W;
bstar = b-eta*grad_b;