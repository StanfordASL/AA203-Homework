function [A, B, c] = linearize_dynamics(x_ref, u_ref, dt)

xk = x_ref(1);
thk = x_ref(2);
xdk = x_ref(3);
thdk = x_ref(4);

mc = 10; mp = 2.; l = 1.; g= 9.81; I = 0.25;
s = sin(thk);
c = cos(thk);

denom = mc + mp * (s)^2;

B = dt*[0; 0; 1/denom; -c/(l*denom)];

% building jacobian df/dx:
J = zeros(4,4);
J(1,3) = 1.;
J(2,4) = 1.;

dxdd_dth = (mp*c*(l*thdk^2 + g*c) - mp*g*(s^2))/denom - (u_ref + mp*s*(l*thdk^2 + g*c))*(2*mp*s*c)/(denom^2);
dxdd_dthd = 2*mp*s*l*thdk/denom;
dthdd_dth = (u_ref*s - mp*l*(thdk^2)*(-s^2 + c^2) - (mc+mp)*g*c)/(l*denom) - (-u_ref*c - mp*l*(thdk^2)*c*s-(mc+mp)*g*s)*(2*l*mp*s*c)/((l*denom)^2);
dthdd_dthd = -(2*mp*l*thdk*c*s)/(l*denom);

J(3,2) = dxdd_dth;
J(3,4) = dxdd_dthd;
J(4,2) = dthdd_dth;
J(4,4) = dthdd_dthd;

A = eye(4) + dt*J;
c = zeros(4,1);
end

