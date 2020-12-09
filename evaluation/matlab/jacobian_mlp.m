function [J,y] = jacobian_mlp(Ws, x)

P = extend_proj(size(x, 2),size(Ws{1}, 1));
y = x * P;
J = P';%eye(size(Ws{1}));

for k=1:length(Ws)
    J = jacobian_layer(Ws{k}, y) * J;
    y = y + tanh(y * Ws{k}');
end

y = y * P';
J = P * J;
