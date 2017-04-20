function acc = ComputeAccuracy(X, y, W1, b1, W2, b2, K)

[~, N] = size(X);
P = EvaluateClassifier(X, W1, b1, W2, b2, K);

[~, k] = max(P);
acc = sum((k-1)'==y)/N;