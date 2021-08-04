n = 4;
A = rand(n);

d = -0.9;
W = zeros(n);
%W(1,4) = 2;
W(2,4) = -2;
%W(1,3) = 2;
%W(4,1) = -2;
%W(3,1) = -2;
W(4,2) = 2;
W = 16*(A-A')-eye(n)*d;
[U,L] = eig(W);
im = 0.1;
for k=1:2:n
    L(k,k) = -d + 1i*im;
    L(k+1,k+1) = -d - 1i*im;
end

P = extend_proj(2,n);
[Us,Ls] = eig(P*U*L*U'*P');
diag(L)
%eig(W*W)
%abs(eig(W*W))
diag(Ls)