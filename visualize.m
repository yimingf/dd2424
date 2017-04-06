function visualize()

mt = [];
for i=1:K
  im = reshape(W(i, :), 32, 32, 3);
  s_im{i} = (im-min(im(:)))/(max(im(:))-min(im(:)));
  s_im{i} = permute(s_im{i}, [2, 1, 3]);
  mt = [mt s_im{i}];
end
montage(mt);

inds = 1:n_epochs;
plot(inds, J_train, inds, J_validation);