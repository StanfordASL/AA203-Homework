clear all; clf; clc; format long;

global v; v = 1.; % Scenario
global l; l = 5.;
global M; M = 10.;
global T; T = 10.;

P0Guess = [.1;.1]; % Initial guess for [px(0);py(0)].
                   % Note that a good choice is fundamental to make the
                   % indirect method converge!

options=optimset('Display','iter','LargeScale','on');
[P0,FVAL,EXITFLAG]=fsolve(@shootingFunc,P0Guess,options); % Solving F(z)=0
EXITFLAG % 1 if convergence is achieved

options = odeset('AbsTol',1e-9,'RelTol',1e-9); % Plotting
[t,z] = ode113(@Rdyn,[0;T],[0.0;0.0;P0],options) ;
subplot(121); plot(z(:,1),z(:,2),'linewidth',3) ;
title('\textbf{a) Optimal Trajectory}','interpreter','latex','FontSize',22,'FontWeight','bold');
xlabel('\boldmath{$x$} \ \textbf{(m)}','interpreter','latex','FontSize',20,'FontWeight','bold');
ylabel('\boldmath{$y$} \ \textbf{(m)}','interpreter','latex','FontSize',20,'FontWeight','bold');
xlim([-1,11]);
ylim([-1,6]);
set(gca,'Xtick',-1:1:11);
set(gca,'Ytick',-1:1:6);
grid on;
subplot(122); plot(t,v*z(:,4)/(1.0 + v*z(:,3)),'linewidth',3);
title('\textbf{b) Optimal Control}','interpreter','latex','FontSize',22,'FontWeight','bold');
xlabel('\boldmath{$t$} \ \textbf{(s)}','interpreter','latex','FontSize',20,'FontWeight','bold');
ylabel('\boldmath{$u$}','interpreter','latex','FontSize',20,'FontWeight','bold');
xlim([0,10]);
ylim([-0.25,1]);
set(gca,'Xtick',-1:1:10);
set(gca,'Ytick',-0.25:0.1:1);
grid on;