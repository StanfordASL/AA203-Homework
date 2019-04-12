% We will implement an infinite horizon LQR controller to stabilize
% the cartpole around vertical. 
% What you need to do:
% Implement the Riccati recursion for the infinite horizon LQR in
% lqr_infinite_horizon_solution.m

% This assignment is based in part on a problem set from Berkeley CS287,
% and the cartpole visualization code was written by Russ Tedrake

clear all; clc; close all;

animate = true; % change to visualize animation
noise = false; % change to toggle stochastic disturbance

f = @sim_cartpole;
dt = 0.1; % we work with discrete time

%% LQR Stabilization

% our reference point is x=0, theta=pi, xdot = 0; thetadot=0; this has the pole vertically UP
% our reference control input is 0; this is a fixed point of the system

x_ref = [0; pi; 0; 0];
u_ref = 0;

if noise
    noise_variance = [0;0;0.02;0.02];
else
    noise_variance = [0.;0.;0.;0.];
end
    
% now let's find the infinite horizon controller
% for the linearized version of the cartpole balancing problem:

Q = eye(4); R = eye(1); 

[K_inf, P_inf] = lqr_infinite_horizon_solution(Q, R); %YOURS to implement

disturbance = (2*rand()-1.)*pi/5; % randomized starting angle
starting_state = [0; pi-disturbance; 0; 0];

ep_length = 200;
clip_val = 150;

x = zeros(4,ep_length+1);
x(:,1) = starting_state;

% x(:,1) = starting_state;
for t=1:ep_length
    w = noise_variance.*randn(4,1);
    u(:,t) =  (K_inf * ( x(:,t) - x_ref ) ) + u_ref;
    u(:,t) = max(min(u(:,t),clip_val),-clip_val);
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