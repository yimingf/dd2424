% dedicated for kth dd2424 deepl17 (deep learning) assignment 3 (k-layer).
function [s] = batchNormalize(s, mu, v)
s = cellfun(@(x) (v.^(-0.5))'.*(x-mu), s, 'UniformOutput', false);