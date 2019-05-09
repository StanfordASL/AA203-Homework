% Function providing equality and inequality constraints
% ceq(var) = 0 and c(var) \le 0 

function [c,ceq] = constraint(var)

global N;
global T;

global x0;
global y0;

% Put here constraint inequalities
c = ;

% Note that var = [x;y;u]
x = var(1:N+1); y = var(N+2:2*N+2); u = var(2*N+3:3*N+3);

% Computing dynamical constraints via the trapezoidal rule
h = 1.0*T/(1.0*N);
for i = 1:N
    % Provide here dynamical constraints via the trapeziodal formula
    ceq(i) = ;
    ceq(i+N) = ;
end

% Put here initial conditions
ceq(1+2*N) = ;
ceq(2+2*N) = ;