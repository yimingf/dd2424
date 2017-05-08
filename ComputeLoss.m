function J = ComputeLoss(X, Y, RNN, h0)

[~, N] = size(X);
[P, H, ~] = synthesizeText(RNN, X, Y, h0);
J = -sum(log(sum(Y.*P, 1)+RNN.epsilon));
