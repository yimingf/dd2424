% function [foo] = a2(eta, lambda)

addpath ~/dd2424/Datasets/cifar-10-batches-mat/

% kth dd2424 deepl17 (deep learning in data science) assignment 3.
clc, clear

filename = {'data_batch_1.mat', 'data_batch_2.mat', 'data_batch_3.mat', 'data_batch_4.mat', 'data_batch_5.mat'};
[X, Y, y, hp.N, ... % X and Y have form of cells
  hp.K, ...
  hp.d ...
] = LoadBatch(filename{1}); % 1
[X_validation, ...
  Y_validation, ...
  ~, ~, ~, ~ ...
] = LoadBatch(filename{2});
[Xtest, ~, ytest, ~, ~, ~] = LoadBatch('test_batch.mat'); % 1

% generate the hyper-parameters.
hp.n_layers   = 3; % number of layers
hp.alpha      = 0.99;
hp.h_nodes    = [hp.d 50 30 hp.K]; % hidden nodes
hp.a          = 0.001; % variance
%hp.eta        = 0.02; % learning rate
%hp.lambda     = 0; % regularization rate
hp.n_batch    = 100; % number of batches
hp.n_epochs   = 10; % number of epoches
hp.decay_rate = 0.95; % decay rate of the learning rate.  
hp.rho        = 0.9; % momentum parameter.
ma.mu         = cell(hp.n_layers, 1);
ma.v          = cell(hp.n_layers, 1);

% initializing W.
W = cell(hp.n_layers, 1);
b = cell(hp.n_layers, 1);
for i=1:hp.n_layers
  W{i} = hp.a.*randn(hp.h_nodes(i+1), hp.h_nodes(i));
  b{i} = zeros(hp.h_nodes(i+1), 1);
end

J_train       = zeros(hp.n_epochs, 1);
J_validation  = zeros(hp.n_epochs, 1);

% split = N-800;
% X_train = X(:, 1:split);
% X_validation = X(:, split+1:N);
% Y_train = Y(:, 1:split);
% Y_validation = Y(:, split+1:N);
% N = split;

max_coarse = 14;
foo = linspace(1.69, 4, max_coarse);
e = 10.^(-foo); % eta
foo = linspace(4, 8, max_coarse);
l = 10.^(-foo); % lambda

for i=1:max_coarse % for eta
  for j=1:max_coarse % for lambda
    hp.eta = e(i);
    hp.lambda = l(j);
    % re-initialize the weight matrices.
    for k=1:hp.n_layers
      W{k} = hp.a.*randn(hp.h_nodes(k+1), hp.h_nodes(k));
      b{k} = zeros(hp.h_nodes(k+1), 1);
    end
    % the training process.
    for k=1:hp.n_epochs
      [W, b, ma] = MiniBatchGD(X, Y, W, b, hp, ma);
      % foo = ComputeCost(X, Y, W, b, hp, ma)
      % J_train(k) = foo;
      %J_validation(i) = ComputeCost(X_validation, Y_validation, W, b, hp, ma);
      % if (mod(k, 10) == 0)
      %   hp.eta = hp.eta * hp.decay_rate;
      % end
    end % for k
    fprintf('training process (%d, %d), accuracy: ', i, j);
    acc(i, j) = ComputeAccuracy(Xtest, ytest, W, b, hp, ma);
  end % for j
end % for i

% foo = J_train(hp.n_epochs);

% % calculate the accuracy
acc = ComputeAccuracy(Xtest, ytest, W, b, hp, ma)