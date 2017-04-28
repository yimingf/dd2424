function [W, b] = MiniBatchGD(X, Y, W, b, hp)

[~, N] = size(X);

% initialize the momentum.
v_W = cell(hp.n_layers, 1);
v_b = cell(hp.n_layers, 1);
for i=1:hp.n_layers
  v_W{i} = zeros(size(W{i}));
  v_b{i} = zeros(size(b{i}));
end

for j=1:N/hp.n_batch
  j_start = (j-1)*hp.n_batch+1;
  j_end = j*hp.n_batch;
  inds = j_start:j_end;
  Xbatch = X(:, inds);
  Ybatch = Y(:, inds);
  % get the mini-batch of X and Y.

  [P, s, mu, v] = EvaluateClassifier(Xbatch, W, b, hp);
  [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, s, P, mu, v, W, b, hp);

  % update the momentum.
  v_W = cellfun( @(x, y) hp.rho*x+hp.eta*y, ...
    v_W, grad_W, 'UniformOutput', false);
  v_b = cellfun( @(x, y) hp.rho*x+hp.eta*y, ...
    v_b, grad_b, 'UniformOutput', false);
  W = cellfun( @(x, y) x-y, ...
    W, v_W, 'UniformOutput', false);
  b = cellfun( @(x, y) x-y, ...
    b, v_b, 'UniformOutput', false);
end % for j