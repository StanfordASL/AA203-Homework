function [x,u] = scp_solution(f,d, Q, R, Qf, u_lb, u_ub, goal_state, x0, u_old, num_steps, dt)
    
n = size(Q,1);
m = size(R,1); 

x_old = zeros(n*num_steps,1);
x_old(1:4) = x0;
%u_old = ones(m*num_steps,1);

% initial forward pass
for i=1:(num_steps-1)
    x_old(i*n+1:i*n+n) = f(x_old((i-1)*n+1:i*n),u_old((i-1)*m+1:i*m),dt);
end

[x,u] = scp(x_old, u_old, u_lb, u_ub, f, d, Q,R, Qf, goal_state, x0, num_steps, dt);
ctr = 1;

while (norm(x-x_old,"inf") + norm(u-u_old,"inf") > 0.5)
    ctr = ctr+1;
    fprintf("Round %i: ", ctr)
    x_old = x;
    u_old = u;
    [x,u] = scp(x_old, u_old, u_lb, u_ub, f, d, Q,R, Qf, goal_state, x0, num_steps, dt);
end

end