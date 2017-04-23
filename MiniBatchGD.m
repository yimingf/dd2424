function [W1, b1, W2, b2] = MiniBatchGD(X, Y, W1, b1, W2, b2, hyper_parameters)%lambda, K, d, m, eta, rho, n_batch)

[~, N] = size(X);

lambda   = hyper_parameters.lambda;
K        = hyper_parameters.K;
d        = hyper_parameters.d;
m        = hyper_parameters.m;
eta      = hyper_parameters.eta;
rho      = hyper_parameters.rho;
n_batch  = hyper_parameters.n_batch;

% the momentum.
v_W1 = zeros(m, d);
v_b1 = zeros(m, 1);
v_W2 = zeros(K, m);
v_b2 = zeros(K, 1);

for j=1:N/n_batch
  j_start = (j-1)*n_batch + 1;
  j_end = j*n_batch;
  inds = j_start:j_end;
  Xbatch = X(:, inds);
  Ybatch = Y(:, inds);
  % get the mini-batch of X and Y.

  [P, H] = EvaluateClassifier(Xbatch, W1, b1, W2, b2, K);
  [grad_W1, grad_b1, grad_W2, grad_b2] = ComputeGradients(Xbatch, Ybatch, H, P, W1, W2, b1, b2, lambda, K, d, m);

  % update the momentum.
  v_W1 = rho*v_W1+eta*grad_W1;
  v_b1 = rho*v_b1+eta*grad_b1;
  v_W2 = rho*v_W2+eta*grad_W2;
  v_b2 = rho*v_b2+eta*grad_b2;
  % update the weights and the bias.
  W1 = W1-v_W1;
  b1 = b1-v_b1;
  W2 = W2-v_W2;
  b2 = b2-v_b2;

  % % update w/o momentum.
  % W1 = W1-eta*grad_W1;
  % b1 = b1-eta*grad_b1;
  % W2 = W2-eta*grad_W2;
  % b2 = b2-eta*grad_b2;
end % for j