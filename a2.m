addpath ~/dd2424/Datasets/cifar-10-batches-mat/

% kth dd2424 deepl17 (deep learning in data science) assignment 2.
clc, clear

filename = {'data_batch_1.mat', 'data_batch_2.mat', 'test_batch.mat'}; % training data, validation data, test data
[X, Y, y, N, ...
  hyper_parameters.K, ... % number of classes
  hyper_parameters.d, ... % number of dimensions on X
  mu] ...
  = LoadBatch(filename{1});
[X_validation, Y_validation, ~, ~, ~, ~, ~] ...
  = LoadBatch(filename{2}); % 1

% generate the hyper-parameters.
hyper_parameters.a          = 0.001; % variance
hyper_parameters.eta        = 0.02; % learning rate
hyper_parameters.lambda     = 0.001; % regularization rate
hyper_parameters.n_batch    = 500; % number of batches
hyper_parameters.n_epochs   = 150; % number of epoches
hyper_parameters.decay_rate = 0.95;
hyper_parameters.m          = 50; % hidden layer.
hyper_parameters.rho        = 0.9; % momentum parameter.

W1 = hyper_parameters.a.*randn(hyper_parameters.m, hyper_parameters.d);
b1 = zeros(hyper_parameters.m, 1);
W2 = hyper_parameters.a.*randn(hyper_parameters.K, hyper_parameters.m);
b2 = zeros(hyper_parameters.K, 1);
J_train = zeros(hyper_parameters.n_epochs, 1);
J_validation = zeros(hyper_parameters.n_epochs, 1);

split = N-1000;
X_train = X(:, 1:split);
X_validation = X(:, split+1:N);
Y_train = Y(:, 1:split);
Y_validation = Y(:, split+1:N);
N = split;

% the training process
for i=1:50

  [W1, b1, W2, b2] = MiniBatchGD(X, Y, W1, b1, W2, b2, hyper_parameters);%, lambda, K, d, m, eta, rho, n_batch);

  foo = ComputeCost(X, Y, W1, b1, W2, b2, hyper_parameters.lambda, hyper_parameters.K)
  J_train(i) = foo;
  %J_validation(i) = ComputeCost(X_validation, Y_validation, W, b, lambda, K);
  if (mod(i, 10) == 0)
    hyper_parameters.eta = hyper_parameters.eta * hyper_parameters.decay_rate;
  end

end % for i

% calculate the accuracy
[X, ~, y, ~, K, ~] = LoadBatch('test_batch.mat'); % 1
acc = ComputeAccuracy(X, y, W1, b1, W2, b2, K)

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