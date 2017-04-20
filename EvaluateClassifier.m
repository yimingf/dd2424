% dedicated for kth dd2424 deepl2017 (deep learning) assignment 2.
function [P, H] = EvaluateClassifier(X, W1, b1, W2, b2, K)
  
s1 = bsxfun(@plus, W1*X, b1);
H = max(0, s1);
s = bsxfun(@plus, W2*H, b2);
P = bsxfun(@rdivide, exp(s), sum(exp(s), 1));