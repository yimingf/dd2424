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
hp.n_layers   = 2; % number of layers
hp.alpha      = 0.99;
hp.h_nodes    = [hp.d 50 hp.K]; % hidden nodes
hp.a          = 0.001; % variance
hp.eta        = 1e-01; % learning rate
hp.lambda     = 1e-06;% 2.0309e-08; % regularization rate
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
% for i=1:7 % for eta
%   % for j=[4 10 11 12 13] % for lambda
%   hp.eta = e(i);
%   % re-initialize the weight matrices.
  % for k=1:hp.n_layers
  %   W{k} = hp.a.*randn(hp.h_nodes(k+1), hp.h_nodes(k));
  %   b{k} = zeros(hp.h_nodes(k+1), 1);
  % end
  % the training process.
for k=1:hp.n_epochs
  [W, b, ma] = MiniBatchGD(X, Y, W, b, hp, ma);
  foo = ComputeCost(X, Y, W, b, hp, ma)
  J_train(k) = foo;
  J_validation(k) = ComputeCost(X_validation, Y_validation, W, b, hp, ma);
  if (mod(k, 10) == 0)
    hp.eta = hp.eta * hp.decay_rate;
  end
end % for k
%   fprintf('training process (%d), accuracy: \n', i);
%   acc(i) = ComputeAccuracy(Xtest, ytest, W, b, hp, ma)
%   % end % for j
% end % for i

% foo = J_train(hp.n_epochs);

% % calculate the accuracy
acc = ComputeAccuracy(Xtest, ytest, W, b, hp, ma)

% ComputeAccuracy.m

function acc = ComputeAccuracy(X, y, W, b, hp, ma)

[~, N] = size(X);
[P, ~] = EvaluateClassifier(X, W, b, hp, ma);

k = cell2mat(cellfun(@(x) find(x==max(x)), P, 'UniformOutput', false));
acc = sum((k-1)'==y)/N;

% MiniBatchGD.m

function [W, b, ma] = MiniBatchGD(X, Y, W, b, hp, ma)

[~, N] = size(X);

% initialize the momentum.
v_W = cell(hp.n_layers, 1);
v_b = cell(hp.n_layers, 1);
for i=1:hp.n_layers
  v_W{i} = zeros(size(W{i}));
  v_b{i} = zeros(size(b{i}));
end

for j=1:N/hp.n_batch
  j_start = (j-1)*hp.n_batch+1;
  j_end = j*hp.n_batch;
  inds = j_start:j_end;
  Xbatch = X(:, inds);
  Ybatch = Y(:, inds);
  % get the mini-batch of X and Y.

  [P, s, mu, v] = EvaluateClassifier(Xbatch, W, b, hp);
  if (size(ma.mu{1}, 1)==0) % initialization.
    ma.mu = mu;
    ma.v  = v;
  else % moving average.
    ma.mu = cellfun(@(x, y) hp.alpha*x+(1-hp.alpha)*y, ma.mu, mu, 'UniformOutput', false);
    ma.v  = cellfun(@(x, y) hp.alpha*x+(1-hp.alpha)*y, ma.v , v , 'UniformOutput', false);
  end
  [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, s, P, mu, v, W, b, hp);

  % update the momentum.
  v_W = cellfun( @(x, y) hp.rho*x+hp.eta*y, ...
    v_W, grad_W, 'UniformOutput', false);
  v_b = cellfun( @(x, y) hp.rho*x+hp.eta*y, ...
    v_b, grad_b, 'UniformOutput', false);
  W = cellfun( @(x, y) x-y, ...
    W, v_W, 'UniformOutput', false);
  b = cellfun( @(x, y) x-y, ...
    b, v_b, 'UniformOutput', false);
end % for j

% ComputeCost.m

function [grad_W, grad_b] = ComputeGradients(X, Y, s, P, mu, v, W, b, hp)

[~, N] = size(X);
g = cell(N, 1);
% initialize the gradients.
grad_W = cell(hp.n_layers, 1);
grad_b = cell(hp.n_layers, 1);
for i=1:hp.n_layers
  grad_W{i} = zeros(size(W{i}));
  grad_b{i} = zeros(size(b{i}));
end

% the last layer.
x = cellfun(@(x) max(0, x), s{hp.n_layers-1}, 'UniformOutput', false);
g = cellfun(@(y, p) -y'/(y'*p)*(diag(p)-p*p'), Y, P, 'UniformOutput', false);
grad_b{hp.n_layers} = mean(reshape(cell2mat(g), [hp.K, N]), 2);
grad_W{hp.n_layers} = (reshape(sum(cell2mat(cellfun(@(g, x) reshape(g'*x', [hp.K*hp.h_nodes(hp.n_layers), 1]), g, x, 'UniformOutput', false)), 2), [hp.K, hp.h_nodes(hp.n_layers)]))/N+2*hp.lambda*W{hp.n_layers};
g = cellfun(@(g, x) g*W{hp.n_layers}*diag(x>0), g, x, 'UniformOutput', false);

for j=(hp.n_layers-1):(-1):1
  g = batchNormBackPass(g, s{j}, mu{j}, v{j});
  if (j == 1)
    x = X;
  else
    x = cellfun(@(x) max(0, x), s{j-1}, 'UniformOutput', false);
  end
  grad_b{j} = mean(reshape(cell2mat(g), [hp.h_nodes(j+1), N]), 2);
  grad_W{j} = (reshape(sum(cell2mat(cellfun(@(g, x) reshape(g'*x', [hp.h_nodes(j+1)*hp.h_nodes(j), 1]), g, x, 'UniformOutput', false)), 2), [hp.h_nodes(j+1), hp.h_nodes(j)]))/N+2*hp.lambda*W{j};
  if (j>1)
    g = cellfun(@(g, x) g*W{j}*diag(x>0), g, x, 'UniformOutput', false);
  end
end % for j

% EvaluateClassifier.m

% dedicated for kth dd2424 deepl2017 (deep learning) assignment 3 (k-layer).
function [P, s, mu, v] = EvaluateClassifier(X, W, b, hp, varargin)

[~, N] = size(X);
H = X;
s = cell(hp.n_layers, 1);
flag = length(varargin) == 0; % moving average
if (flag)
  mu = cell(hp.n_layers, 1);
  v = cell(hp.n_layers, 1);
else
  mu = varargin{1}.mu;
  v = varargin{1}.v;
end

for i=1:(hp.n_layers-1)
  s{i} = cellfun(@(x) W{i}*x+b{i}, H, 'UniformOutput', false);
  if (flag)
    mu{i} = mean(cell2mat(s{i}), 2);
    v{i} = var(cell2mat(s{i})')*(N-1)/N;
  end
  % H = cellfun(@(x) max(0, x), s{i}, 'UniformOutput', false);

  H = cellfun(@(x) max(0, x), batchNormalize(s{i}, mu{i}, v{i}), ...
  'UniformOutput', false); % ReLU
end

s{hp.n_layers} = cellfun(@(x) W{hp.n_layers}*x+b{hp.n_layers}, ...
  H, 'UniformOutput', false);
foo = exp(cell2mat(s{hp.n_layers}));
P = bsxfun(@rdivide, foo, sum(foo, 1)); % softmax
P = mat2cell(P, [size(P, 1)], ones(size(P, 2), 1)); % transform into cell

% batchNormalize.m

% dedicated for kth dd2424 deepl17 (deep learning) assignment 3 (k-layer).
function [s] = batchNormalize(s, mu, v)
s = cellfun(@(x) (v.^(-0.5))'.*(x-mu), s, 'UniformOutput', false);

% batchNormBackPass.m

% dedicated for kth dd2424 deepl17 (deep learning) assignment 3 (k-layer).
function [g] = batchNormBackPass(g, s, mu, v)

[~, N] = size(g);
[m, ~] = size(mu);
dv = (-0.5)*sum(reshape(cell2mat(cellfun(@(g, s) g.*(v.^(-1.5))*diag(s-mu), g, s, 'UniformOutput', false)), [m, N]), 2);
dmu = -sum(reshape(cell2mat(cellfun(@(g) g.*(v.^(-0.5)), g, 'UniformOutput', false)), [m, N]), 2);
g = cellfun(@(g, s) g.*(v.^(-0.5))+(2/N)*dv'*diag(s-mu)+(dmu/N)', g, s, 'UniformOutput', false);

% LoadBatch.m

function [X, Y, y, N, K, d, mu] = LoadBatch(filename)

A = load(filename);
X = im2double(A.data');
mu = mean(X, 2);
X = X-repmat(mu, [1, size(X, 2)]); % transfer to zero mean.
[d, ~] = size(X);
X = mat2cell(X, [size(X, 1)], ones(size(X, 2), 1));
y = A.labels;

N = length(y);
K = length(min(y):max(y));

Y = zeros(K, N);
for i = 0:(K-1)
  rows = y == i;
  Y(i+1, rows) = 1;
end
Y = mat2cell(Y, [size(Y, 1)], ones(size(Y, 2), 1));