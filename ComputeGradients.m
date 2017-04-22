function [grad_W1, grad_b1, grad_W2, grad_b2] = ComputeGradients(X, Y, H, s1, P, W1, W2, b1, b2, lambda, K, d, m)

[~, N] = size(X);
grad_W1 = zeros(m, d);
grad_b1 = zeros(m, 1);
grad_W2 = zeros(K, m);
grad_b2 = zeros(K, 1);

for i=1:N
  x = X(:, i);
  y = Y(:, i);
  p = P(:, i);
  h = H(:, i);
  g = -y'/(y'*p)*(diag(p)-p*p');
  grad_b2 = grad_b2+g';
  grad_W2 = grad_W2+g'*h';

  g = g*W2;
  % g = g*diag(h>0); % ReLU
  g = g*diag((exp(-s1)./((1+exp(-s1)).^2)));
  grad_b1 = grad_b1+g';
  grad_W1 = grad_W1+g'*x';
end

% update the gradients
grad_W2 = grad_W2/N+2*lambda*W2;
grad_b2 = grad_b2/N;
grad_W1 = grad_W1/N+2*lambda*W1;
grad_b1 = grad_b1/N;