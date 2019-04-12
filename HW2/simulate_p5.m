% We will implement an iterative LQR controller to do a swing up
% maneuver. After the swing up is complete, we stabilize with the LQR
% controller.
% What you need to do:
% Implement iLQR in ilqr_solution.m

% This assignment is based in part on a problem set from Berkeley CS287,
% and the cartpole visualization code was written by Russ Tedrake

clear all; clc; close all;

animate = true; % change to visualize animation
noise = false; % change to toggle stochastic disturbance

f = @sim_cartpole;
dt = 0.1; % we work with discrete time

%% LQR Swing up

% our reference point is x=0, theta=pi, xdot = 0; thetadot=0; this has the pole vertically UP
% our reference control input is 0; this is a fixed point of the system

x_ref = [0; pi; 0; 0];
u_ref = 0;

clip_val = 150;

if noise
    noise_variance = [0;0;0.02;0.02];
else
    noise_variance = [0.;0.;0.;0.];
end

Q = eye(4); R = eye(1); 

[K_inf, P_inf] = lqr_infinite_horizon_solution(Q, R); 

num_swingup_steps = 75;
ep_length = 200;

u_bar = randn(1,num_swingup_steps); %initialize with random actions

d = @linearize_dynamics;

Qf = 1000*eye(4); 
Q = diag([10,10,2,2]);
R = eye(1);

goal_state=[0; pi; 0; 0];
start_state=[0; 0; 0; 0];

[x_bar,u_bar,l,L] = ilqr_solution(f,d, Q, R, Qf, goal_state, start_state, u_bar, num_swingup_steps, dt); %YOURS to implement

x = zeros(4,ep_length+1);
x(:,1) = start_state;

for t=1:ep_length
    if t<num_swingup_steps
        % swing up maneuver is planned for fixed horizon, execute actions
        % from swing up, up until that horizon
        u(:,t) = u_bar(:,t) + l(:,t) + L(:,:,t) * (x(:,t) - x_bar(:,t));
    else 
        % after swing up is over, switch to infinite horizon LQR
        u(:,t) =  K_inf * (x(:,t) - goal_state) + u_ref;
    end
        
    u(:,t) = max(min(u(:,t),clip_val),-clip_val);
    w = noise_variance.*randn(4,1);
    x(:,t+1) = f(x(:,t), u(:,t), dt) + w;
end

figure; 
plot(x');

figure; 
plot(u,'--');

if animate
    for t=1:ep_length
        cartpole_draw(t,x(:,t));
    end
end
