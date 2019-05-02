% Adjoints equations related to our rocket.

function zdot = Zdyn(t,z)
           
global g;
global b;
global uMax;

v = z(2);
m = z(3);
py = z(4);
pv = z(5);
pm = z(6);

phi = % TODO compute phi

% TODO use phi to compute control action

zdot = % TODO Rocket dynamics and adjoint equations