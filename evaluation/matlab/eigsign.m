n = 3;
A = stableMat(n);
B = stableMat(n);
%max(abs(eig(A))) * max(abs(eig(B)))
%max(abs(eig( A * B)))

A = rand(n);
W = 0.5 * (A - A') - eye(n) * 0.01;
%eig(W)
v = abs(rand(n,1));
v(1) = 0.001;
%eig(diag(v)*W)

h = 1.0;
d = 0.1;
a = 0.64;
b = 0.53;
c = 0.5;
W =[-d   -b   -a;
    b   -d   -c;
    a    c  -d];

x = [0.0;
    1.0;  %0.9602;
    0.5]; % 0.4786

count = 0;
for j = 1:10000
    a = rand()*2;
    b = rand()*2;
    c = rand()*2;

    g = 0.0001;
    a = (rand() - 0.5) * 2.0;
    A = W;
%    A = [-g a; -a -g];
    g = rand();
    a = rand(); 
    B = A;
%    B = [-g a; -a -g];
    %x = rand(n,1);%[1.0; 1.0; 0.5];
%    diag(tanh(A * x).^2)
    J1 = eye(n) + h * diag(tanh(A * x).^2)*A;
    y = x + tanh(A * x);
%    diag(tanh(A * y).^2)
%diag(abs(rand(n,1)))
    J2 = eye(n) + h * diag(tanh(B * y).^2)*B;
    p1 = max(abs(eig(J1)))
    p2 = max(abs(eig(J2)))
    pT = max(abs(eig(J2 * J1)))
    break;
    if p1 < 1 && p2 < 1
        count = count + 1;
    end
    if p1 < 1 && p2 < 1 && pT > 1
        W
        x
        p1
        p2
        pT
        break;
    end
end
count

function A = stableMat(n)
    
    A = rand(n);
    A = 0.5 * (A - A') - eye(n) * 0.1;
    v = abs(rand(n,1));
    %A = diag(v) * A;
    A = A + eye(n);
    s = max(abs(eig(A)));
%    if s >= 1
        A = A * 1 / s * 0.99999; 
%    end
end