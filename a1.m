addpath ~/dd2424/Datasets/cifar-10-batches-mat/
A = load('data_batch_1.mat');
I = reshape(A.data', 32, 32, 3, 10000);
I = permute(I, [2, 1, 3, 4]);
montage(I(:, :, :, 1:500), 'size', [5,5]);