% a2.m

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

% ComputeAccuracy.m
function acc = ComputeAccuracy(X, y, W1, b1, W2, b2, K)

[~, N] = size(X);
P = EvaluateClassifier(X, W1, b1, W2, b2, K);

[~, k] = max(P);
acc = sum((k-1)'==y)/N;

% ComputeCost.m
function J = ComputeCostComputeCost(X, Y, W1, b1, W2, b2, lambda, K)

[~, N] = size(X);
P = EvaluateClassifier(X, W1, b1, W2, b2, K);
J = -sum(log(sum(Y.*P, 1)))/N+lambda*(sumsqr(W1)+sumsqr(W2));

% ComputeGradients.m
function [grad_W1, grad_b1, grad_W2, grad_b2] = ComputeGradients(X, Y, H, s1, P, W1, W2, b1, b2, lambda, K, d, m)

[~, N] = size(X);
grad_W1 = zeros(m, d);
grad_b1 = zeros(m, 1);
grad_W2 = zeros(K, m);
grad_b2 = zeros(K, 1);

for i=1:N
  x = X(:, i);
  y = Y(:, i);
  p = P(:, i);
  h = H(:, i);
  g = -y'/(y'*p)*(diag(p)-p*p');
  grad_b2 = grad_b2+g';
  grad_W2 = grad_W2+g'*h';

  g = g*W2;
  % g = g*diag(h>0); % ReLU
  g = g*diag((exp(-s1)./((1+exp(-s1)).^2)));
  grad_b1 = grad_b1+g';
  grad_W1 = grad_W1+g'*x';
end

% update the gradients
grad_W2 = grad_W2/N+2*lambda*W2;
grad_b2 = grad_b2/N;
grad_W1 = grad_W1/N+2*lambda*W1;
grad_b1 = grad_b1/N;

% EvaluateClassifier.m
% dedicated for kth dd2424 deepl2017 (deep learning) assignment 2.
function [P, H, s1] = EvaluateClassifier(X, W1, b1, W2, b2, K)
  
s1 = bsxfun(@plus, W1*X, b1);
% H = max(0, s1); % ReLU
H = 1./(1+exp(-s1)); % sigmoid
s = bsxfun(@plus, W2*H, b2);
P = bsxfun(@rdivide, exp(s), sum(exp(s), 1));

% LoadBatch.m
function [X, Y, y, N, K, d, mu] = LoadBatch(filename)

A = load(filename);
X = im2double(A.data');
mu = mean(X, 2);
X = X-repmat(mu, [1, size(X, 2)]); % transfer to zero mean.
y = A.labels;

N = length(y);
K = length(min(y):max(y));
[d, ~] = size(X);

Y = zeros(K, N);

for i = 0:(K-1)
  rows = y == i;
  Y(i+1, rows) = 1;
end

% MiniBatchGD.m
function [W1, b1, W2, b2] = MiniBatchGD(X, Y, W1, b1, W2, b2, hyper_parameters)%lambda, K, d, m, eta, rho, n_batch)

[~, N] = size(X);

lambda   = hyper_parameters.lambda;
K        = hyper_parameters.K;
d        = hyper_parameters.d;
m        = hyper_parameters.m;
eta      = hyper_parameters.eta;
rho      = hyper_parameters.rho;
n_batch  = hyper_parameters.n_batch;

% the momentum.
v_W1 = zeros(m, d);
v_b1 = zeros(m, 1);
v_W2 = zeros(K, m);
v_b2 = zeros(K, 1);

for j=1:N/n_batch
  j_start = (j-1)*n_batch + 1;
  j_end = j*n_batch;
  inds = j_start:j_end;
  Xbatch = X(:, inds);
  Ybatch = Y(:, inds);
  % get the mini-batch of X and Y.

  [P, H, s1] = EvaluateClassifier(Xbatch, W1, b1, W2, b2, K);
  [grad_W1, grad_b1, grad_W2, grad_b2] = ComputeGradients(Xbatch, Ybatch, H, s1, P, W1, W2, b1, b2, lambda, K, d, m);

  % % update the momentum.
  % v_W1 = rho*v_W1+eta*grad_W1;
  % v_b1 = rho*v_b1+eta*grad_b1;
  % v_W2 = rho*v_W2+eta*grad_W2;
  % v_b2 = rho*v_b2+eta*grad_b2;
  % % update the weights and the bias.
  % W1 = W1-v_W1;
  % b1 = b1-v_b1;
  % W2 = W2-v_W2;
  % b2 = b2-v_b2;

  % update w/o momentum.
  W1 = W1-eta*grad_W1;
  b1 = b1-eta*grad_b1;
  W2 = W2-eta*grad_W2;
  b2 = b2-eta*grad_b2;
end % for j

% train.m

num_sample = 100;
e_min = 0.001, e_max = 0.003;
l_min = 2, l_max = 5;
e = e_min+(e_max-e_min)*rand(1, num_sample);
l = l_min+(l_max-l_min)*rand(1, num_sample);
lambda = 10.^(-l);

loss = zeros(1, num_sample);

for i=1:num_sample
  loss(i) = a2(eta, lambda);
end

for i=1:3
  [~, ind] = min(loss);
  eta(ind)
  lambda(ind)
  loss(ind) = [];
  eta(ind) = [];
  lambda(ind) = [];
end

% visualize.m
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