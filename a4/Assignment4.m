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
RNN.eta         = 0.001; % learning rate
RNN.seq_length  = 25; % length of sequence
RNN.sig         = 0.01;
RNN.b           = zeros(RNN.m, 1);
RNN.c           = zeros(RNN.K, 1);
RNN.U           = randn(RNN.m, RNN.K)*RNN.sig;
RNN.W           = randn(RNN.m, RNN.m)*RNN.sig;
RNN.V           = randn(RNN.K, RNN.m)*RNN.sig; % 7 8 9 10 11
RNN.n           = 10; % depth of the network
RNN.n_epochs    = 15;
RNN.epsilon     = 1e-6; % AdaGrad
RNN.g           = [7 8 9 10 11]; % b c U W V
RNN.int_to_char = int_to_char;
RNN.char_to_int = char_to_int;
% J_train         = zeros(hp.n_epochs, 1);
% J_validation    = zeros(hp.n_epochs, 1);

% 1.3
h0 = zeros(RNN.m, 1);
[RNN] = MiniBatchGD(X, book_chars, RNN);
e = 1;
X_batch = X(:, e:e+1000-1);
Y_batch = X(:, e+1:e+1000);
[~, ~, Y, ~, ~] = synthesizeText(RNN, X_batch, Y_batch, h0);

for i=1:1000
  chars(i) = int_to_char(find(Y(:, i) == 1));
end
chars

% MiniBatchGD.m
function [RNN] = MiniBatchGD(X, book_chars, RNN)

foo = 1;
smooth_loss = 0;
f = fieldnames(RNN)';
% for i=RNN.g
%   m.(f{i}) = zeros(size(RNN.(f{i})));
% end
loss = zeros(100, 1);

for epoch = 1:RNN.n_epochs
  e = 1;
  hprev = zeros(RNN.m, 1);
  while (e+RNN.seq_length<RNN.N)
    X_batch = X(:, e:e+RNN.seq_length-1);
    Y_batch = X(:, e+1:e+RNN.seq_length);

    [P, H, ~, hafter, l] = synthesizeText(RNN, X_batch, Y_batch, hprev);
    grads = ComputeGradients(X_batch, Y_batch, RNN, P, H, hprev);
    hprev = hafter;

    for i=RNN.g % AdaGrad
      % m.(f{i}) = m.(f{i})+grads.(f{i}).^2;
      RNN.(f{i}) = RNN.(f{i})-RNN.eta*grads.(f{i});%./sqrt(m.(f{i})+RNN.epsilon);
    end

    if (foo == 1)
      smooth_loss = l;
    else
      smooth_loss = 0.999*smooth_loss+0.001*l;
    end

    if (mod(foo, 10000) == 0) % every 100 iterations.
      foo
      smooth_loss
      X_batch = X(:, e:e+199);
      Y_batch = X(:, e+1:e+200);
      [~, ~, Y, ~, ~] = synthesizeText(RNN, X_batch, Y_batch, hprev);
      for i=1:200
        chars(i) = RNN.int_to_char(find(Y(:, i) == 1));
      end
      chars
    end % if
    e = e+RNN.seq_length;
    foo = foo+1;
  end
end

% ComputeLoss.m
function J = ComputeLoss(X, Y, RNN, h0)

[~, N] = size(X);
[P, H, ~] = synthesizeText(RNN, X, Y, h0);
J = -sum(log(sum(Y.*P, 1)+RNN.epsilon));

% synthesizeText.m
function [P, H, Y_predict, hafter, loss] = synthesizeText (RNN, X_batch, Y_batch, h0)

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

  [~, k] = max(p);
  Y_predict(k, i) = 1;
  P(:, i) = p;
  H(:, i) = h;
end
hafter = H(:, N);
loss = -sum(log(sum(Y_batch.*P, 1)+RNN.epsilon));

% ComputeGradients.m
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