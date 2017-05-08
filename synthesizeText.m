function [P, H, Y_predict, hafter, loss] = synthesizeText (RNN, X_batch, Y_batch, h0)

P = zeros(RNN.K, RNN.seq_length);
H = zeros(RNN.m, RNN.seq_length);
Y_predict = zeros(size(Y_batch));
h = h0;
for i=1:RNN.seq_length
  x = X_batch(:, i);
  a = RNN.W*h+RNN.U*x+RNN.b;
  h = tanh(a);
  o = RNN.V*h+RNN.c;
  foo = exp(o);
  p = bsxfun(@rdivide, foo, sum(foo, 1)); % softmax

  [~, k] = max(p);
  Y_predict(k, i) = 1;
  P(:, i) = p;
  H(:, i) = h;
end
hafter = H(:, RNN.seq_length);
loss = -sum(log(sum(Y_batch.*P, 1)+RNN.epsilon));