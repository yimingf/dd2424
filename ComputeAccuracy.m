function acc = ComputeAccuracy(X, y, W, b, K)

[~, N] = size(X);
P = EvaluateClassifier(X, W, b, K);

[~, k] = max(P);
acc = sum((k-1)'==y)/N;