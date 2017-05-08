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