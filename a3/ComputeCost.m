function J = ComputeCost(X, Y, W, b, hp, ma)

[~, N] = size(X);
[P, ~, ~, ~] = EvaluateClassifier(X, W, b, hp, ma);
J = -sum(log(sum(cell2mat(cellfun(@(x, y) x.*y, Y, P, 'UniformOutput', false)), 1)))/N+hp.lambda*sum(cellfun(@sumsqr, W));