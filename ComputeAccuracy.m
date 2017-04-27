function acc = ComputeAccuracy(X, y, W, b, hp)

[~, N] = size(X);
[P, ~] = EvaluateClassifier(X, W, b, hp);

k = cell2mat(cellfun(@(x) find(x==max(x)), P, 'UniformOutput', false));
acc = sum((k-1)'==y)/N;