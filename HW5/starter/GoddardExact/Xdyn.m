% Dynamical equations related to our rocket.
% We implement the exact version of the
% switching time tSw given in the homework,
% i.e., tSw = (m0 - mf)/(b*uMax). The optimal
% control is u(t) = uMax if t <= tSw and
% u(t) = 0 otherwise.

function xdot = Xdyn(t,x,tf)
           
global g;
global b;
global uMax;
global m0;
global mf;

v = x(2);
m = x(3);

tSw = (m0 - mf)/(b*uMax); % Optimal switching time
if tSw > tf % Verifying that: 0 < tSw <= tf
    tSw = tf;
end

u = 0.; % Optimal control policy
if t <= tSw
    u = uMax;
end

xdot = [v; u/m - g; -b*u]; % Rocket dynamics