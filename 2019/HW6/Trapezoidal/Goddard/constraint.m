% Function providing equality and inequality constraints
% ceq(var) = 0 and c(var) \le 0

function [c,ceq] = constraint(var)

global N;
global T;

global y0;
global v0;
global m0;
global mf;

% Put here constraint inequalities
c = ;

% Note that var = [y;v;m;u]
y = var(1:N+1); v = var(N+2:2*N+2); m = var(2*N+3:3*N+3); u = var(3*N+4:4*N+4); % Note: var = [y;v;m;u]

% Computing dynamical constraints via the trapezoidal rule
h = 1.0*T/(1.0*N);
for i = 1:N
    % Provide here dynamical constraints via the trapeziodal formula
    ceq(i) = ;
    ceq(i+N) = ;
    ceq(i+2*N) = ;
end

% Put here initial and final conditions
ceq(1+3*N) = ;
ceq(2+3*N) = ;
ceq(3+3*N) = ;
ceq(4+3*N) = ;