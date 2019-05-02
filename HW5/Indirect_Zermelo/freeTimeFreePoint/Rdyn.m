function zdot = Rdyn(t,z) % Dynamics for the shooting dot(x,p) = R(t,(x,p))
                          % No approximations are used

global v;

x = z(1);
y = z(2);
px = z(3);
py = z(4);

[fl,flDer] = flowFunc(y);
if px ~= 0
    u = atan(py/px); % Optimal controls from minimality condition
                     % Note: we need to manage the case px=0 !
else
    if py < 0
        u = pi/2.0; % Optimal controls from minimality condition: px=0
    else
        u = -pi/2.0;
    end
end

zdot = [v*cos(u) + fl; v*sin(u); 0.0; -px*flDer];