function x1 = sim_cartpole(x0, u, dt)

DT =0.1; t=0;
while t < dt
    current_dt = min(DT, dt-t);
    x0 = x0 + current_dt*dynamics(x0,u);
    t = t+current_dt;
end
x1 = x0;



    function xdot = dynamics(x,u)
  
        %global cartpole
        %
        %mc=cartpole.mc;  mp=cartpole.mp;  l=cartpole.l;  g=cartpole.g;
        mc = 10; mp = 2.; l = 1.; g= 9.81; I = 0.25;
        s = sin(x(2,:)); c = cos(x(2,:));
      
        xddot = [u + mp*s.*(l*x(4,:).^2 + g*c)]./[mc+mp*s.^2];
        tddot = [-u.*c - mp*l*x(4,:).^2.*c.*s - (mc+mp)*g.*s]./[l*(mc+mp.*s.^2)];
        xdot = [x(3:4,:); xddot; tddot];
      