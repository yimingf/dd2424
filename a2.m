% function [foo] = a2(eta, lambda)

addpath ~/dd2424/Datasets/cifar-10-batches-mat/

% kth dd2424 deepl17 (deep learning in data science) assignment 2.
clc, clear

filename = {'data_batch_1.mat', 'data_batch_2.mat', 'data_batch_3.mat', 'data_batch_4.mat', 'data_batch_5.mat'};
X = [];
Y = [];
y = [];
N = 0;
for i=1:5
  [XX, YY, yy, NN, ...
    hyper_parameters.K, ...
    hyper_parameters.d ...
  ] = LoadBatch(filename{i}); % 1
  X = [X XX];
  Y = [Y YY];
  y = [y; yy];
  N = N + NN;
end

% generate the hyper-parameters.
hyper_parameters.a          = 0.001; % variance
hyper_parameters.eta        = 0.02; % learning rate
hyper_parameters.lambda     = 0.001; % regularization rate
hyper_parameters.n_batch    = 500; % number of batches
hyper_parameters.n_epochs   = 60; % number of epoches
hyper_parameters.decay_rate = 0.95;
hyper_parameters.m          = 100; % hidden layer.
hyper_parameters.rho        = 0.9; % momentum parameter.

W1 = hyper_parameters.a.*randn(hyper_parameters.m, hyper_parameters.d);
b1 = zeros(hyper_parameters.m, 1);
W2 = hyper_parameters.a.*randn(hyper_parameters.K, hyper_parameters.m);
b2 = zeros(hyper_parameters.K, 1);
J_train = zeros(hyper_parameters.n_epochs, 1);
J_validation = zeros(hyper_parameters.n_epochs, 1);

% split = N-1000;
% X_train = X(:, 1:split);
% X_validation = X(:, split+1:N);
% Y_train = Y(:, 1:split);
% Y_validation = Y(:, split+1:N);
% N = split;

% the training process
for i=1:20

  [W1, b1, W2, b2] = MiniBatchGD(X, Y, W1, b1, W2, b2, hyper_parameters);%, lambda, K, d, m, eta, rho, n_batch);
  foo = ComputeCost(X, Y, W1, b1, W2, b2, hyper_parameters.lambda, hyper_parameters.K)
  J_train(i) = foo;
  % J_validation(i) = ComputeCost(X_validation, Y_validation, W1, b1, W2, b2, hyper_parameters.lambda, hyper_parameters.K);
  if (mod(i, 10) == 0)
    hyper_parameters.eta = hyper_parameters.eta * hyper_parameters.decay_rate;
  end

end % for i

% foo = J_train(hyper_parameters.n_epochs);

% % calculate the accuracy
[X, ~, y, ~, K, ~] = LoadBatch('test_batch.mat'); % 1
acc = ComputeAccuracy(X, y, W1, b1, W2, b2, K)