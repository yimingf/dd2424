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