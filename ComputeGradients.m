function [grad_W, grad_b] = ComputeGradients(X, Y, s, P, mu, v, W, b, hp)

[~, N] = size(X);
g = cell(N, 1);
% initialize the gradients.
grad_W = cell(hp.n_layers, 1);
grad_b = cell(hp.n_layers, 1);
for i=1:hp.n_layers
  grad_W{i} = zeros(size(W{i}));
  grad_b{i} = zeros(size(b{i}));
end

% the last layer.
x = cellfun(@(x) max(0, x), s{hp.n_layers-1}, 'UniformOutput', false);
g = cellfun(@(y, p) -y'/(y'*p)*(diag(p)-p*p'), Y, P, 'UniformOutput', false);
grad_b{hp.n_layers} = mean(reshape(cell2mat(g), [hp.K, N]), 2);
grad_W{hp.n_layers} = (reshape(sum(cell2mat(cellfun(@(g, x) reshape(g'*x', [hp.K*hp.h_nodes(hp.n_layers), 1]), g, x, 'UniformOutput', false)), 2), [hp.K, hp.h_nodes(hp.n_layers)]))/N+2*hp.lambda*W{hp.n_layers};
g = cellfun(@(g, x) g*W{hp.n_layers}*diag(x>0), g, x, 'UniformOutput', false);

for j=(hp.n_layers-1):(-1):1
  g = batchNormBackPass(g, s{j}, mu{j}, v{j});
  if (j == 1)
    x = X;
  else
    x = cellfun(@(x) max(0, x), s{j-1}, 'UniformOutput', false);
  end
  grad_b{j} = mean(reshape(cell2mat(g), [hp.h_nodes(j+1), N]), 2);
  grad_W{j} = (reshape(sum(cell2mat(cellfun(@(g, x) reshape(g'*x', [hp.h_nodes(j+1)*hp.h_nodes(j), 1]), g, x, 'UniformOutput', false)), 2), [hp.h_nodes(j+1), hp.h_nodes(j)]))/N+2*hp.lambda*W{j};
  if (j>1)
    g = cellfun(@(g, x) g*W{j}*diag(x>0), g, x, 'UniformOutput', false);
  end
end % for j