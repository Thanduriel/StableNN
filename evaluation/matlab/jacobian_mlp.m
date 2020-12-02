function J = jacobian_mlp(Ws, x)

P = extend_proj(2,16);
y = x * P;
J = eye(size(Ws{1}));


for k=1:length(Ws)
    J = jacobian_layer(Ws{k}, y) * J;
    y = y + tanh(Ws{k}*y')';
end

J = P * J * P';

y = y * P'
