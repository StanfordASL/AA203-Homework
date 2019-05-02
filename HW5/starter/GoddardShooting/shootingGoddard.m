% Solving Goddard's problem via shooting method.
% From the homework, we know that Goddard's problem is solved
% with switching time tSw = (m0 - mf)/(b*uMax). We want to
% discover what is the optimal final time tf that maximizes
% the final height h(tf). For this, we implement a shooting
% method exactly as seen in the last lecture.

clear all; clf; clc; format long;

global g; g = 9.81;
global b;
global uMax;
global h0; h0 = 0.;
global v0; v0 = 0.;
global m0;
global mf;

% Scenario: recall that we must satisfy uMax > m0*g
m0 = 12000; mf = 1000;
b = 1e-3; uMax = 1.2e5;

% Initial guess for [tf,ph(0),pv(0),pm(0)], see also the homework.
% A good inital guess is crucial to make the shooting method converge.
% From the homework we know that phy(0) = -1 and pv(t) = t - tf. Then
% pv(0) = -tf. For a guess for pm(0), the computations done in the
% homework give: pm(0) = -(tf*(uMax/m0 - g) + v0)/(b*uMax).
% The only guess will be on tf
tf = 253.302; % Time guessed from the script on dichotomic search
tSw = (m0 - mf)/(b*uMax);
if tSw > tf
    tSw = tf;
end
py0 = -1.;
pv0 = -tf;
pm0 = -(tf*(uMax/m0 - g) + v0)/(b*uMax);
X0Guess = [tf;py0;pv0;pm0];

options=optimset('Display','iter','LargeScale','on','TolX',1e-6,'MaxIter',100,'MaxFunEvals',100);
[X0,FVAL,EXITFLAG]=fsolve(@shootingFunc,X0Guess,options); % Solving F(tf,py(0),pv(0),pm(0))=0
EXITFLAG % 1 or 2 if convergence is achieved

% Plotting
tf = X0(1);
fprintf('Switching time tSw = %f\n',tSw);
fprintf('Final time tf = %f\n',tf);
options = odeset('AbsTol',1e-9,'RelTol',1e-9);
[t,x] = ode113(@(t,x) Zdyn(t,x), [0 tf], [h0;v0;m0;X0(2);X0(3);X0(4)], options);
subplot(221); plot(t,x(:,1),'linewidth',3) ;
title('\textbf{a) Height}','interpreter','latex','FontSize',22,'FontWeight','bold');
xlabel('\boldmath{$t$} \ \textbf{(s)}','interpreter','latex','FontSize',20,'FontWeight','bold');
ylabel('\boldmath{$h$} \ \textbf{(m)}','interpreter','latex','FontSize',20,'FontWeight','bold');
xlim([-inf inf]);
ylim([-inf inf]);
grid on;
subplot(222); plot(t,x(:,2),'linewidth',3) ;
title('\textbf{b) Velocity}','interpreter','latex','FontSize',22,'FontWeight','bold');
xlabel('\boldmath{$t$} \ \textbf{(s)}','interpreter','latex','FontSize',20,'FontWeight','bold');
ylabel('\boldmath{$v$} \ \textbf{(m/s)}','interpreter','latex','FontSize',20,'FontWeight','bold');
xlim([-inf inf]);
ylim([-inf inf]);
grid on;
subplot(223); plot(t,x(:,3),'linewidth',3) ;
title('\textbf{c) Mass}','interpreter','latex','FontSize',22,'FontWeight','bold');
xlabel('\boldmath{$t$} \ \textbf{(s)}','interpreter','latex','FontSize',20,'FontWeight','bold');
ylabel('\boldmath{$m$} \ \textbf{(kg)}','interpreter','latex','FontSize',20,'FontWeight','bold');
xlim([-inf inf]);
ylim([-inf inf]);
grid on;
control = zeros(size(t));
for i = 1:size(t)
    phi = x(i,5)/x(i,3) - x(i,6)*b; % Phi function
    if phi < 0 % Optimal control from our optimal policy
        control(i) = uMax;
    end
end
subplot(224); plot(t,control,'linewidth',3);
title('\textbf{d) Optimal Control}','interpreter','latex','FontSize',22,'FontWeight','bold');
xlabel('\boldmath{$t$} \ \textbf{(s)}','interpreter','latex','FontSize',20,'FontWeight','bold');
ylabel('\boldmath{$u$}','interpreter','latex','FontSize',20,'FontWeight','bold');
xlim([-inf inf]);
ylim([-inf inf]);
grid on;