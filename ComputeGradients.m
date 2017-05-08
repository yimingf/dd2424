function [grads] = ComputeGradients(X, Y, RNN, P, H);

[~, N] = size(X);
grads.b = zeros(RNN.m, 1);
grads.c = zeros(RNN.K, 1);
grads.V = zeros(RNN.K, RNN.m);
grads.U = zeros(RNN.m, RNN.K);
grads.W = zeros(RNN.m, RNN.m);
dO = zeros(RNN.K, N);
dH = zeros(RNN.m, N);
dA = zeros(RNN.m, N);
h0 = zeros(RNN.m, 1);

for i=1:N
  y = Y(:, i);
  h = H(:, i);
  p = P(:, i);

  g = (-y'/(y'*p)*(diag(p)-p*p'))';
  grads.c = grads.c+g;
  grads.V = grads.V+g*h';
  dO(:, i) = g;
end

dH(:, N) = dO(:, N)'*RNN.V;
dA(:, N) = dH(:, N)'*(diag(1-(H(:, N)).^2));
for i=(N-1):(-1):1
  dH(:, i) = dO(:, i)'*RNN.V+dA(:, i+1)'*RNN.W;
  dA(:, i) = dH(:, i)'*(diag(1-(H(:, i)).^2));
end

for i=1:N
  x = X(:, i);
  h = H(:, i);
  if (i==1)
    grads.W = grads.W+dA(:, i)*h0';
  else
    grads.W = grads.W+dA(:, i)*h';
  end

  grads.b = grads.b+dA(:, i);
  grads.U = grads.U+dA(:, i)*x';
end