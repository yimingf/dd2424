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

  J_train(i) = ComputeCost(X_train, Y_train, W, b, lambda, K)
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