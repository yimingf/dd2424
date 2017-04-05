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