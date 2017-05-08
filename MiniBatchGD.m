function [RNN] = MiniBatchGD(book_data, book_chars, RNN)

[~, N] = size(book_data);

X_chars = book_data(1:RNN.seq_length);
Y_chars = book_data(2:RNN.seq_length+1);

X = zeros(RNN.K, RNN.seq_length);
Y = zeros(RNN.K, RNN.seq_length);

for i=1:RNN.seq_length
  X(char_to_int(X_chars(i)), i) = 1;
  Y(char_to_int(Y_chars(i)), i) = 1;
end

h0 = zeros(RNN.m, 1);
[P, H, ~] = synthesizeText(RNN, h0, X(:, 1), RNN.seq_length);
grads = ComputeGradients(X, Y, RNN, P, H);