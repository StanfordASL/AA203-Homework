% Function F(tf) for which we seek zeros.
% This function returns v(tf) where the velocity v arises
% from integrating the rocket dynamics with the optimal control
% given in the homework, i.e., u(t) = uMax if t <= tSw and
% u(t) = 0 otherwise. The time tf satisfying v(tf) = 0 is
% the one that maximizes the height, as explained in the
% homework.

function XZero = dichotomyFunc(X)

global h0;
global v0;
global m0;

tf = X;

% Integrating the rocket dynamics
options = odeset('AbsTol',1e-9,'RelTol',1e-9);
[t,x] = ode113(@(t,x) Xdyn(t,x,tf), [0 tf], [h0;v0;m0], options);

XZero = x(end,2); % Condition v(tf) = 0