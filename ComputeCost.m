function J = ComputeCost(X, Y, W, b, hp)

[~, N] = size(X);
[P, ~] = EvaluateClassifier(X, W, b, hp);
J = -sum(log(sum(cell2mat(cellfun(@(x, y) x.*y, Y, P, 'UniformOutput', false)), 1)))/N+hp.lambda*sum(cellfun(@sumsqr, W));