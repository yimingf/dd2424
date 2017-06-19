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

[~, RNN.N] = size(book_data);
X = zeros(RNN.K, RNN.N);
for i=1:RNN.N
  X(char_to_int(book_data(i)), i) = 1;
end

% 1.2
RNN.m           = 100; % #hidden states
RNN.eta         = 0.1; % learning rate
RNN.seq_length  = 25; % length of sequence
RNN.sig         = 0.01;
RNN.b           = zeros(RNN.m, 1);
RNN.c           = zeros(RNN.K, 1);
RNN.U           = randn(RNN.m, RNN.K)*RNN.sig;
RNN.W           = randn(RNN.m, RNN.m)*RNN.sig;
RNN.V           = randn(RNN.K, RNN.m)*RNN.sig; % 7 8 9 10 11
RNN.n           = 10; % depth of the network
RNN.n_epochs    = 10;
RNN.epsilon     = 1e-8; % AdaGrad
RNN.g           = [7 8 9 10 11]; % b c U W V
RNN.int_to_char = int_to_char;
RNN.char_to_int = char_to_int;
% J_train         = zeros(hp.n_epochs, 1);
% J_validation    = zeros(hp.n_epochs, 1);

% 1.3
h0 = zeros(RNN.m, 1);
RNN = MiniBatchGD(X, book_chars, RNN);
e = 1;
X_batch = X(:, e:e+1000-1);
Y_batch = X(:, e+1:e+1000);
Y = synthesizeText(RNN, X_batch, Y_batch, h0);

for i=1:1000
  chars(i) = int_to_char(find(Y(:, i) == 1));
end
chars