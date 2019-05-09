% Cost of the problem

function c = cost(var)

global N;

% Note that var = [y;v;m;u]
y = var(1:N+1); v = var(N+2:2*N+2); m = var(2*N+3:3*N+3); u = var(3*N+4:4*N+3);

% Put here the cost
c = ;