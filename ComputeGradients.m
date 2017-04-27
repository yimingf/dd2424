function [grad_W, grad_b] = ComputeGradients(X, Y, s, P, W, b, hp)

[~, N] = size(X);

% initialize the gradients.
grad_W = cell(hp.n_layers, 1);
grad_b = cell(hp.n_layers, 1);
for i=1:hp.n_layers
  grad_W{i} = zeros(size(W{i}));
  grad_b{i} = zeros(size(b{i}));
end

g = cell(N, 1);

% the last layer.
x = cellfun(@(x) max(0, x), ...
  s{hp.n_layers-1}, 'UniformOutput', false);
g = cellfun(@(y, p) -y'/(y'*p)*(diag(p)-p*p'), ...
  Y, P, 'UniformOutput', false);
% 昨天写到这里 今天接着写

for i=1:N
  x = max(0, s{hp.n_layers-1}(:, i));
  y = Y(:, i);
  p = P(:, i);
  g{i} = -y'/(y'*p)*(diag(p)-p*p');
  % 昨天模仿到这里 今天接着写
  grad_b{hp.n_layers} = grad_b{hp.n_layers}+g{i}';
  grad_W{hp.n_layers} = grad_W{hp.n_layers}+g{i}'*x';

  g{i} = g{i}*W{hp.n_layers};
  g{i} = g{i}*diag(x>0);
end
% g = cellfun( @(x) x*W{hp.n_layers}, g, 'UniformOutput', false);
% g = cellfun( @(x) x*diag(x>0), g, 'UniformOutput', false);

% update the gradients.
grad_W{hp.n_layers} = grad_W{hp.n_layers}/N+2*hp.lambda*W{hp.n_layers};
grad_b{hp.n_layers} = grad_b{hp.n_layers}/N;

for j=(hp.n_layers-1):(-1):2
  % insert BatchNormBackPass(...arg) here.
  for i=1:N
    x = max(0, s{j-1}(:, i));
    grad_b{j} = grad_b{j}+g{i}';
    grad_W{j} = grad_W{j}+g{i}'*x';

    g{i} = g{i}*W{j};
    g{i} = g{i}*diag(x>0);
  end % for i
  grad_W{j} = grad_W{j}/N+2*hp.lambda*W{j};
  grad_b{j} = grad_b{j}/N;
end % for j

for i=1:N
  x = X(:, i);
  grad_b{1} = grad_b{1}+g{i}';
  grad_W{1} = grad_W{1}+g{i}'*x';
end

grad_W{1} = grad_W{1}/N+2*hp.lambda*W{1};
grad_b{1} = grad_b{1}/N;