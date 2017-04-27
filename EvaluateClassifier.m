% dedicated for kth dd2424 deepl2017 (deep learning) assignment 3 (k-layer).
function [P, s] = EvaluateClassifier(X, W, b, hp)
  
s = cell(hp.n_layers, 1);
s{1} = cellfun(@(x) W{1}*x+b{1}, X, 'UniformOutput', false);

for i=2:hp.n_layers
  H = cellfun(@(x) max(0, x), s{i-1}, 'UniformOutput', false); % ReLU
  % H = 1./(1+exp(-s1)); % sigmoid
  s{i} = cellfun(@(x) W{i}*x+b{i}, H, 'UniformOutput', false);
end

foo = exp(cell2mat(s{hp.n_layers}));
P = bsxfun(@rdivide, foo, sum(foo, 1)); % softmax
P = mat2cell(P, [size(P, 1)], ones(size(P, 2), 1)); % transform into cell