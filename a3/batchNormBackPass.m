% dedicated for kth dd2424 deepl17 (deep learning) assignment 3 (k-layer).
function [g] = batchNormBackPass(g, s, mu, v)

[~, N] = size(g);
[m, ~] = size(mu);
dv = (-0.5)*sum(reshape(cell2mat(cellfun(@(g, s) g.*(v.^(-1.5))*diag(s-mu), g, s, 'UniformOutput', false)), [m, N]), 2);
dmu = -sum(reshape(cell2mat(cellfun(@(g) g.*(v.^(-0.5)), g, 'UniformOutput', false)), [m, N]), 2);
g = cellfun(@(g, s) g.*(v.^(-0.5))+(2/N)*dv'*diag(s-mu)+(dmu/N)', g, s, 'UniformOutput', false);