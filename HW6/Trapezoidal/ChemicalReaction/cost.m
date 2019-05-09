% Cost of the problem

function c = cost(var)

global N;

% Note that var = [x;y;u]
x = var(1:N+1); y = var(N+2:2*N+2); u = var(2*N+3:3*N+3);

% Put here the cost
c = ;