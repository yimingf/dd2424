% codes for Assignment 1

%% Assignment1.m
addpath ~/dd2424/Datasets/cifar-10-batches-mat/

% exercise 2
clear

filename = {'data_batch_1.mat', 'data_batch_2.mat', 'data_batch_3.mat', 'data_batch_4.mat', 'data_batch_5.mat'};
X = [];
Y = [];
y = [];
N = 0;
for i=1:5
  [XX, YY, yy, NN, K, d] = LoadBatch(filename{i}); % 1
  X = [X XX];
  Y = [Y YY];
  y = [y; yy];
  N = N + NN;
end

a = 0.01; % variance
eta = 0.02; % learning rate
lambda = 0.01; % regularization rate
n_batch = 500; % number of batches
n_epochs = 50; % number of epoches
W = a.*randn(K, d);
b = a.*randn(K, 1); % 2
J_train = zeros(n_epochs, 1);
J_validation = zeros(n_epochs, 1);

split = N-1000;
X_train = X(:, 1:split);
X_validation = X(:, split+1:N);
Y_train = Y(:, 1:split);
Y_validation = Y(:, split+1:N);
N = split;

% the training process
for i=1:n_epochs

  for j=1:N/n_batch
    j_start = (j-1)*n_batch + 1;
    j_end = j*n_batch;
    inds = j_start:j_end;
    Xbatch = X(:, inds);
    Ybatch = Y(:, inds);
    % get the mini-batch of X and Y.

    [W, b] = MiniBatchGD(Xbatch, Ybatch, W, b, lambda, K, d, eta);
  end % for j

  J_train(i) = ComputeCost(X_train, Y_train, W, b, lambda, K) % 4
  J_validation(i) = ComputeCost(X_validation, Y_validation, W, b, lambda, K); % 5
  if (mod(i, 10) == 0)
    eta = eta * 0.9;
  end

end % for i

% calculate the accuracy
[X, ~, y, ~, K, ~] = LoadBatch('test_batch.mat'); % 1
acc = ComputeAccuracy(X, y, W, b, K)

% % exercise 1

% filename = 'data_batch_1.mat';
% [X, Y, y, N, K, d] = LoadBatch(filename); % 1

% a = 0.01; % variance
% eta = 0.01; % learning rate
% lambda = 0.01; % regularization rate
% n_batch = 100; % number of batches
% n_epochs = 40; % number of epoches
% W = a.*randn(K, d);
% b = a.*randn(K, 1); % 2
% J_train = zeros(n_epochs, 1);
% J_validation = zeros(n_epochs, 1);

% split = 0.9*N;
% X_train = X(:, 1:split);
% X_validation = X(:, split+1:N);
% Y_train = Y(:, 1:split);
% Y_validation = Y(:, split+1:N);
% N = split;

% % the training process
% for i=1:n_epochs

%   for j=1:N/n_batch
%     j_start = (j-1)*n_batch + 1;
%     j_end = j*n_batch;
%     inds = j_start:j_end;
%     Xbatch = X(:, inds);
%     Ybatch = Y(:, inds);
%     % get the mini-batch of X and Y.

%     [W, b] = MiniBatchGD(Xbatch, Ybatch, W, b, lambda, K, d, eta);
%   end % for j

%   J_train(i) = ComputeCost(X_train, Y_train, W, b, lambda, K); % 4
%   J_validation(i) = ComputeCost(X_validation, Y_validation, W, b, lambda, K); % 5

% end % for i

% % calculate the accuracy
% [X, ~, y, ~, K, ~] = LoadBatch('test_batch.mat'); % 1
% acc = ComputeAccuracy(X, y, W, b, K)

%% ComputeAccuracy.m
function acc = ComputeAccuracy(X, y, W, b, K)

[~, N] = size(X);
P = EvaluateClassifier(X, W, b, K);

[~, k] = max(P);
acc = sum((k-1)'==y)/N;

%% ComputeCost.m
function J = ComputeCost(X, Y, W, b, lambda, K)

J = 0;
[~, N] = size(X);
P = EvaluateClassifier(X, W, b, K); 
J = -sum(log(sum(Y.*P, 1)))/N+lambda*sumsqr(W);

%% ComputeGradient.m
function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda, K, d)

[~, N] = size(X);
grad_W = zeros(K, d);
grad_b = zeros(K, 1);

for i=1:N
  x = X(:, i);
  y = Y(:, i);
  p = P(:, i);
  g = -y'/(y'*p)*(diag(p)-p*p');
  grad_b = grad_b + g';
  grad_W = grad_W + g'*x';
end
  
grad_W = grad_W/N+2*lambda*W;
grad_b = grad_b/N;

%% EvaluateClassifier.m
function P = EvaluateClassifier(X, W, b, K)

Y = bsxfun(@plus, W*X, b);
P = bsxfun(@rdivide, exp(Y), sum(exp(Y), 1));

%% LoadBatch.m
function [X, Y, y, N, K, d] = LoadBatch(filename)

A = load('data_batch_1.mat');
X = im2double(A.data');
y = A.labels;

N = length(y);
K = length(min(y):max(y));
[d, ~] = size(X);

Y = zeros(K, N);

for i = 0:(K-1)
  rows = y == i;
  Y(i+1, rows) = 1;
end

%% MiniBatchGD.m
function [Wstar, bstar] = MiniBatchGD(X, Y, W, b, lambda, K, d, eta)

P = EvaluateClassifier(X, W, b, K);
[grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda, K, d);

Wstar = W-eta*grad_W;
bstar = b-eta*grad_b;

%% visualize.m
function visualize()

mt = [];
for i=1:K
  im = reshape(W(i, :), 32, 32, 3);
  s_im{i} = (im-min(im(:)))/(max(im(:))-min(im(:)));
  s_im{i} = permute(s_im{i}, [2, 1, 3]);
  mt = [mt s_im{i}];
end
montage(mt);

inds = 1:n_epochs;
plot(inds, J_train, inds, J_validation);