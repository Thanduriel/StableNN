function J = jacobian_layer(W, y)

Id = eye(size(W));

J = Id + diag(1-tanh(W*y').^2)*W;