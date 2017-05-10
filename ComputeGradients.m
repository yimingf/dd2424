function [grads] = ComputeGradients(X, Y, RNN, P, H, hprev);

f = fieldnames(RNN)';
for i=RNN.g
  grads.(f{i}) = zeros(size(RNN.(f{i})));
end
dO = zeros(RNN.seq_length, RNN.K);
dH = zeros(RNN.seq_length, RNN.m);
dA = zeros(RNN.seq_length, RNN.m);

for i=1:RNN.seq_length
  y = Y(:, i);
  h = H(:, i);
  p = P(:, i);

  g = (-y'/(y'*p)*(diag(p)-p*p'));
  grads.c = grads.c+g';
  grads.V = grads.V+g'*h';
  dO(i, :) = g;
end

dH(RNN.seq_length, :) = dO(RNN.seq_length, :)*RNN.V;
dA(RNN.seq_length, :) = dH(RNN.seq_length, :)*(diag(1-(H(:, RNN.seq_length)).^2));
for i=(RNN.seq_length-1):(-1):1
  dH(i, :) = dO(i, :)*RNN.V+dA(i+1, :)*RNN.W;
  dA(i, :) = dH(i, :)*(diag(1-(H(:, i)).^2));
end

for i=1:RNN.seq_length
  x = X(:, i);
  if (i==1)
    grads.W = grads.W+dA(i, :)'*hprev';
  else
    h = H(:, i-1);
    grads.W = grads.W+dA(i, :)'*h';
  end
  grads.b = grads.b+dA(i, :)';
  grads.U = grads.U+dA(i, :)'*x';
end