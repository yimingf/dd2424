function [RNN] = MiniBatchGD(X, book_chars, RNN)

e = 1;
foo = 1;
smooth_loss = 0;
hprev = zeros(RNN.m, 1);
f = fieldnames(RNN)';
for i=RNN.g
  m.(f{i}) = zeros(size(RNN.(f{i})));
end

while (e+RNN.seq_length<RNN.N)
  X_batch = X(:, e:e+RNN.seq_length-1);
  Y_batch = X(:, e+1:e+RNN.seq_length);

  [P, H, ~, hafter, l] = synthesizeText(RNN, X_batch, Y_batch, hprev);
  grads = ComputeGradients(X_batch, Y_batch, RNN, P, H, hprev);
  hprev = hafter;

  for i=RNN.g % AdaGrad
    m.(f{i}) = m.(f{i})+grads.(f{i}).^2;
    RNN.(f{i}) = RNN.(f{i})-RNN.eta*grads.(f{i})./sqrt(m.(f{i})+RNN.epsilon);
  end

  if (foo == 1)
    smooth_loss = l;
  else
    smooth_loss = 0.999*smooth_loss+0.001*l;
  end

  foo = foo+1;
  e = e+RNN.seq_length;
  if (mod(foo, 100) == 0) % every 100 iterations.
    smooth_loss
    [~, ~, Y, ~, ~] = synthesizeText(RNN, X_batch, Y_batch, hprev);
    for i=1:RNN.seq_length
      chars(i) = RNN.int_to_char(find(Y(:, i) == 1));
    end
    chars
  end
end