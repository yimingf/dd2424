function [P, H, hafter, loss] = forwardPass (RNN, X_batch, Y_batch, h0)

[~, N] = size(X_batch);
P = zeros(RNN.K, N);
H = zeros(RNN.m, N);
Y_predict = zeros(size(Y_batch));
h = h0;
for i=1:N
  x = X_batch(:, i);
  a = RNN.W*h+RNN.U*x+RNN.b;
  h = tanh(a);
  o = RNN.V*h+RNN.c;
  foo = exp(o);
  p = bsxfun(@rdivide, foo, sum(foo, 1)); % softmax

  P(:, i) = p;
  H(:, i) = h;
end
hafter = H(:, N);
loss = -sum(log(sum(Y_batch.*P, 1)+RNN.epsilon));