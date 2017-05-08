% kth dd2424 deepl17 (deep learning in data science) assignment 4.
clc, clear

% 1.1
book_fname = './Datasets/goblet_book.txt';
fid = fopen(book_fname, 'r');
book_data = fscanf(fid, '%c');
fclose(fid);
book_chars = unique(book_data);
[~, RNN.K] = size(book_chars);

char_to_int = containers.Map('KeyType', 'char', 'ValueType', 'int32');
int_to_char = containers.Map('KeyType', 'int32', 'ValueType', 'char');

for i=1:RNN.K
  char_to_int(book_chars(i)) = i;
  int_to_char(int32(i)) = book_chars(i);
end

% 1.2
RNN.m           = 5; % #hidden states
RNN.eta         = 0.1; % learning rate
RNN.seq_length  = 25; % length of sequence
RNN.b           = zeros(RNN.m, 1);
RNN.c           = zeros(RNN.K, 1);
RNN.sig         = 0.01;
RNN.U           = randn(RNN.m, RNN.K)*RNN.sig;
RNN.W           = randn(RNN.m, RNN.m)*RNN.sig;
RNN.V           = randn(RNN.K, RNN.m)*RNN.sig;
RNN.n           = 10; % depth of the network
RNN.n_epochs    = 10;
% J_train         = zeros(hp.n_epochs, 1);
% J_validation    = zeros(hp.n_epochs, 1);

% 1.3
h0 = zeros(RNN.m, 1);
x0 = zeros(RNN.K, 1);
x0(1) = 1; % one-hot encoding

[P, H, ~] = synthesizeText(RNN, h0, x0, RNN.seq_length);

for k=1:RNN.n_epochs
  RNN = MiniBatchGD(book_data, book_chars, RNN);
  % foo = ComputeCost(X, Y, W, b, hp, ma)
  % J_train(k) = foo;
  % J_validation(k) = ComputeCost(X_validation, Y_validation, W, b, hp, ma);
  % if (mod(k, 10) == 0)
  %   hp.eta = hp.eta * hp.decay_rate;
  % end
end % for k



for i=1:n
  chars(i) = int_to_char(find(Y(:, i) == 1));
end
chars