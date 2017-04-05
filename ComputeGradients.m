function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda, K, d)

[~, N] = size(X);
grad_W = zeros(K, d);
grad_b = zeros(K, 1);

for i=1:N
  x = X(:, i);
  y = Y(:, i);
  p = P(:, i);
  g = -y'/(y'*p)*(diag(p)-p*p');
  grad_b = grad_b + g';
  grad_W = grad_W + g'*x';
end
  
grad_W = grad_W/N+2*lambda*W;
grad_b = grad_b/N;