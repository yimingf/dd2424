
num_sample = 100;
e_min = 0.001, e_max = 0.003;
l_min = 2, l_max = 5;
e = e_min+(e_max-e_min)*rand(1, num_sample);
l = l_min+(l_max-l_min)*rand(1, num_sample);
lambda = 10.^(-l);

loss = zeros(1, num_sample);

for i=1:num_sample
  loss(i) = a2(eta, lambda);
end

for i=1:3
  [~, ind] = min(loss);
  eta(ind)
  lambda(ind)
  loss(ind) = [];
  eta(ind) = [];
  lambda(ind) = [];
end