function [P, H, Y] = synthesizeText (RNN, h, x, N)

P = zeros(RNN.K, N);
H = zeros(RNN.m, N);
Y = zeros(RNN.K, N);
for i=1:N
  a = RNN.W*h+RNN.U*x+RNN.b;
  h = tanh(a);
  o = RNN.V*h+RNN.c;

  foo = exp(o);
  p = bsxfun(@rdivide, foo, sum(foo, 1)); % softmax

  a = rand;
  ixs = find(cumsum(p)-a>0);
  ii = ixs(1);
  % [~, k] = max(p);
  Y(ii, i) = 1;
  x = Y(:, i); % xnext.

  P(:, i) = p;
  H(:, i) = h;
end