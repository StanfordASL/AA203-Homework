% Problem (OCP)_2 from Pset 6 - Trapezoidal Rule

clear all; clf; clc; format long;

% Parameters
global N; N = ; % Choose here the number of discretization points
global mu; mu = 3.9915e14;
global rE; rE = 6378145;
global h0; h0 = 7500;
global D; D = 5e-3;
global b; b = 1e-3;
global uMax; uMax = 1.2e5;

% Scenario
global T; T = 258.;
global y0; y0 = 0.;
global v0; v0 = 0.;
global m0; m0 = 12000;
global mf; mf = 1000;

% Bound on the state: better conditioning the formulation (see below)
global yMax; yMax = 5e6;
global vMax; vMax = 2000;

% Since this optimal control problem is highly nonlinear, without
% an appropriate intialization direct methods unlikely converge.
% In the following lines, we provide such initialization by recalling the 
% solution that we obtained for the simplified Goddard problem in the Pset 5.
% For the height, we just select a stright-line in time connecting y0 to
% 1.5e5 (which is more or less the final height that we found in Pset 5).
% For the velocity, we select the average v(t) = vMax/2 in [0,tf].
% For the mass, we select a straight-line in time between 0 and tSw, the
% switching time computed in Pset 5 (see below).
% Finally, for the control, we select the maximal value u(t) = uMax in [0,tf].

% Finding what index NSw the time tSw corresponds to
global tSw; tSw = (m0 - mf)/(b*uMax);
h = (1.0*T/(1.0*N));
NSw = 0; indexFound = 0; iterator = 0;
while indexFound == 0
    % If iterator*h <= tSw < iteartor*h + h, then we have found the index
    if iterator*h <= tSw && tSw < (iterator + 1)*h
        NSw = iterator + 1;
        indexFound = 1;
    end
    iterator = iterator + 1;
end
uInit = zeros(N+1,1);
yInit = zeros(N+1,1);
vInit = 0.5*vMax*ones(N+1,1);
mInit = mf*ones(N+1,1);
% Initialization exxplained above
for i=1:N+1
    yInit(i) = y0*(1. - (i-1)*1.0/N) + 1.5e5*(i-1)*1.0/N;
    if (i-1) <= NSw
        mInit(i) = m0*(1. - (i-1)*1.0/NSw) + mf*(i-1)*1.0/NSw;
    end
    if i<= N
        if (i-1)*1.0*T/N < tSw
            uInit(i) = uMax;
        end
    end
end
% Initialization for fmincon
varInit = [yInit; vInit; mInit; uInit];

% Lower and upper bounds.
lb = zeros(4*N+4,1); ub = uMax*ones(4*N+4,1); % For the control: 0 \le u \le uMax
ub(1:N+1) = yMax; % For the state y : 0 \le y \le yMax
ub(N+2:2*N+2) = vMax; % For the state v : 0 \le v \le vMax
lb(2*N+3:3*N+3) = mf; ub(2*N+3:3*N+3) = m0; % For the state m : mf \le v \le m0

% Solving the problme via fmincon
options=optimoptions('fmincon','Display','iter','Algorithm','sqp','MaxFunctionEvaluations',100000,'MaxIterations',10000);
[var,Fval,convergence] = fmincon(@cost,varInit,[],[],[],[],lb,ub,@constraint,options); % Solving the problem
convergence % = 1, good

% Collecting the solution. Note that var = [y;v;m;u]
y = var(1:N+1); v = var(N+2:2*N+2); m = var(2*N+3:3*N+3); u = var(3*N+4:4*N+4); % Collecting the solution
tState = zeros(N+1,1);
for i = 1:N
    tState(i+1) = tState(i) + (1.0*T/(1.0*N));
end
t = zeros(N+1,1);
for i = 1:N
    t(i+1) = t(i) + (1.0*T/(1.0*N));
end

% Plotting
subplot(221); plot(tState,y,'linewidth',3);
title('\textbf{a) Height}','interpreter','latex','FontSize',22,'FontWeight','bold');
xlabel('\boldmath{$t$} \ \textbf{(s)}','interpreter','latex','FontSize',20,'FontWeight','bold');
ylabel('\boldmath{$h$} \ \textbf{(m)}','interpreter','latex','FontSize',20,'FontWeight','bold');
xlim([-inf inf]);
ylim([-inf inf]);
grid on;
subplot(222); plot(tState,v,'linewidth',3) ;
title('\textbf{b) Velocity}','interpreter','latex','FontSize',22,'FontWeight','bold');
xlabel('\boldmath{$t$} \ \textbf{(s)}','interpreter','latex','FontSize',20,'FontWeight','bold');
ylabel('\boldmath{$v$} \ \textbf{(m/s)}','interpreter','latex','FontSize',20,'FontWeight','bold');
xlim([-inf inf]);
ylim([-inf inf]);
grid on;
subplot(223); plot(tState,m,'linewidth',3) ;
title('\textbf{c) Mass}','interpreter','latex','FontSize',22,'FontWeight','bold');
xlabel('\boldmath{$t$} \ \textbf{(s)}','interpreter','latex','FontSize',20,'FontWeight','bold');
ylabel('\boldmath{$m$} \ \textbf{(kg)}','interpreter','latex','FontSize',20,'FontWeight','bold');
xlim([-inf inf]);
ylim([-inf inf]);
grid on;
subplot(224); plot(t,u,'linewidth',3);
title('\textbf{d) Optimal Control}','interpreter','latex','FontSize',22,'FontWeight','bold');
xlabel('\boldmath{$t$} \ \textbf{(s)}','interpreter','latex','FontSize',20,'FontWeight','bold');
ylabel('\boldmath{$u$}','interpreter','latex','FontSize',20,'FontWeight','bold');
xlim([-inf inf]);
ylim([-inf inf]);
grid on;