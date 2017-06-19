function [RNN] = MiniBatchGD(X, book_chars, RNN)

foo = 1;
smooth_loss = 0;
f = fieldnames(RNN)';
for i=RNN.g
  m.(f{i}) = zeros(size(RNN.(f{i})));
end

for epoch = 1:RNN.n_epochs
  e = 1;
  hprev = zeros(RNN.m, 1);
  while (e+RNN.seq_length<RNN.N)
    X_batch = X(:, e:e+RNN.seq_length-1);
    Y_batch = X(:, e+1:e+RNN.seq_length);

    [P, H, hafter, l] = forwardPass(RNN, X_batch, Y_batch, hprev);
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

    e = e+RNN.seq_length;

    if (mod(foo, 10000) == 0) % every 100 iterations.
      foo
      smooth_loss
      X_batch = X(:, e:e+199);
      Y_batch = X(:, e+1:e+200);
      Y = synthesizeText(RNN, X_batch, Y_batch, hprev);
      for i=1:200
        chars(i) = RNN.int_to_char(find(Y(:, i) == 1));
      end
      chars
    end % if
    foo = foo+1;
  end
end