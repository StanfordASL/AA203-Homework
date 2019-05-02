function zdot = Rdyn(t,z) % Dynamics for the shooting dot(x,p) = R(t,(x,p))
                          % We approximate: sin(u)~u, cos(u)~1-u^2/2

global v;

x = z(1);
y = z(2);
px = z(3);
py = z(4);

[fl,flDer] = flowFunc(y);
u = v*py/(1.0 + v*px); % Optimal controls from maximality condition

zdot = [v*(1.0 - u^2/2.0) + fl; v*u; 0.0; -px*flDer];