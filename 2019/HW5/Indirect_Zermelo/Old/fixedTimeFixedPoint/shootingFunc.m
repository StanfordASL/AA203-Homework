function XZero = shootingFunc(X) % Function F(z) for which we seek zeros

global l;
global M;
global T;

options = odeset('AbsTol',1e-9,'RelTol',1e-9);
[t,z] = ode113(@Rdyn,[0.0;T],[0.0;0.0;X],options); % ODE: \dot{z}(t) = R(z(t))

XZero = [z(end,1) - M; z(end,2) - l]; % Condition: F(z) = 0