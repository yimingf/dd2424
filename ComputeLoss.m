function J = ComputeLoss(X, Y, RNN, h0)

[~, N] = size(X);
[P, H, ~] = synthesizeText(RNN, h0, X(:, 1), N);
J = -sum(log(sum(Y.*P, 1)))/N;
