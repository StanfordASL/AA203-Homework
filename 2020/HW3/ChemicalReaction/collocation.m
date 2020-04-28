% Problem (OCP)_1 from Pset 3

clear all; clf; clc; format long;

% Parameters and scenario
global N; N = 10;
global T; T = 1.;
global uMax; uMax = 1.0;
global x0; x0 = 1.;
global y0; y0 = 0.;

% Random initialization
uInit = 0.5*uMax*ones(N+1,1);
xInit = ones(N+1,1); yInit = zeros(N+1,1);
varInit = [xInit; yInit; uInit];

% Lower and upper bounds.
lb = zeros(3*N+3,1); ub = uMax*ones(3*N+3,1); % For the control: 0 \le u \le uMax
ub(1:N+1) = 1.; % For the state x : 0 \le x \le 1
ub(N+2:2*N+2) = 1.; % For the state y : 0 \le y \le 1

% Solving the problme via fmincon
options=optimoptions('fmincon','Display','iter','Algorithm','sqp','MaxFunctionEvaluations',100000,'MaxIterations',10000);
[var,Fval,convergence] = fmincon(@cost,varInit,[],[],[],[],lb,ub,@constraint,options); % Solving the problem
convergence % = 1, good

% Collecting the solution. Note that var = [x;y;u]
x = var(1:N+1); y = var(N+2:2*N+2); u = var(2*N+3:3*N+3);
tState = zeros(N+1,1);
t = zeros(N+1,1);
for i = 1:N
    t(i+1) = t(i) + (1.0*T/(1.0*N));
end

% Plotting
fprintf('Optimal Final Quantity for the Second Substance = %f\n\n',Fval);
subplot(131); plot(t,x,'linewidth',3);
title('\textbf{a) First Substance}','interpreter','latex','FontSize',22,'FontWeight','bold');
xlabel('\boldmath{$t$} \ \textbf{(s)}','interpreter','latex','FontSize',20,'FontWeight','bold');
ylabel('\boldmath{$x$} \ \textbf{(\%)}','interpreter','latex','FontSize',20,'FontWeight','bold');
xlim([-inf inf]);
ylim([-inf inf]);
grid on;
subplot(132); plot(t,y,'linewidth',3) ;
title('\textbf{b) Second Substance}','interpreter','latex','FontSize',22,'FontWeight','bold');
xlabel('\boldmath{$t$} \ \textbf{(s)}','interpreter','latex','FontSize',20,'FontWeight','bold');
ylabel('\boldmath{$y$} \ \textbf{(\%)}','interpreter','latex','FontSize',20,'FontWeight','bold');
xlim([-inf inf]);
ylim([-inf inf]);
grid on;
subplot(133); plot(t,u,'linewidth',3);
title('\textbf{c) Optimal Control}','interpreter','latex','FontSize',22,'FontWeight','bold');
xlabel('\boldmath{$t$} \ \textbf{(s)}','interpreter','latex','FontSize',20,'FontWeight','bold');
ylabel('\boldmath{$u$}','interpreter','latex','FontSize',20,'FontWeight','bold');
xlim([-inf inf]);
ylim([-inf inf]);
grid on;