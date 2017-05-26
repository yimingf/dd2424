function [Y] = synthesizeText (RNN, X_batch, Y_batch, h0)

[~, N] = size(X_batch);
P = zeros(RNN.K, N);
H = zeros(RNN.m, N);
Y = zeros(size(Y_batch));
h = h0;
x = zeros(RNN.K, 1);
x(11) = 1; % encoding of '.', the default character.
for i=1:N
  a = RNN.W*h+RNN.U*x+RNN.b;
  h = tanh(a);
  o = RNN.V*h+RNN.c;
  foo = exp(o);
  p = bsxfun(@rdivide, foo, sum(foo, 1)); % softmax

  cp = cumsum(p);
  a = rand;
  ixs = find(cp-a>0);
  ii = ixs(1); % corrected pick-up method. respect the randomness.

  Y(ii, i) = 1;
  x = Y(:, i);
  P(:, i) = p;
  H(:, i) = h;
end
hafter = H(:, N);
loss = -sum(log(sum(Y_batch.*P, 1)+RNN.epsilon));