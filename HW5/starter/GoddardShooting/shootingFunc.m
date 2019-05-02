% Function F(tf,py(0),pv(0),pm(0)) for which we seek zeros.
% The implementation is the same seen in the lecture, where
% the input is X = (tf,py(0),pv(0),pm(0)) while the output
% is (m(tf) - mf,py(tf) + 1,pv(tf),H(h(tf),v(tf),m(tf),ph(tf),pv(tf),pm(tf)))
% where H is the hamiltonian of the problem, which must be equal to 0.

function XZero = shootingFunc(X)

global h0;
global v0;
global m0;
global mf;

tf = X(1);
py0 = X(2);
pv0 = X(3);
pm0 = X(4);

% Integrating the adjoint equations
options = odeset('AbsTol',1e-9,'RelTol',1e-9);
[t,z] = ode113(@(t,z) Zdyn(t,z), [0 tf], [h0;v0;m0;py0;pv0;pm0], options); % Recall that

XZero = [z(end,3) - mf; % Condition: m(tf) - mf = 0
         z(end,4) + 1.; % Condition: py(tf) + 1 = 0
         z(end,5); % Condition: pv(tf) = 0
         hamiltonianFunc(z(end,1),z(end,2),z(end,3),z(end,4),z(end,5),z(end,6))]; % Condition: H(y(tf),v(tf),m(tf),py(tf),pv(tf),pm(tf)) = 0