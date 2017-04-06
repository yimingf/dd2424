function P = EvaluateClassifier(X, W, b, K)

Y = bsxfun(@plus, W*X, b);
P = bsxfun(@rdivide, exp(Y), sum(exp(Y), 1));