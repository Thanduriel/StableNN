function P = extend_proj(n,m)

P = zeros(n,m);
val = 1 / sqrt(m/2);
P(1:n/2, 1:m/2) = val;
P(n/2+1:n, m/2+1:m) = val;