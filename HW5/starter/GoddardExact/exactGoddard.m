% Exaclty solving Goddard's problem.
% From the homework, we know that Goddard's problem is solved
% with switching time tSw = (m0 - mf)/(b*uMax). We want to
% discover what is the optimal final time tf that maximizes
% the final height h(tf). For this, we implement a dichotomic
% search on tf with the following idea: the final time tf
% that maximizes h(tf) is the time for which the time derivative
% of h(t) at tf is zero, i.e., 0 = h'(tf) = v(tf). Then, we
% seek tf as the zero for v(tf), where the velocity v arises
% from integrating the rocket dynamics with the optimal control
% given in the homework, i.e., u(t) = uMax if t <= tSw and
% u(t) = 0 otherwise.

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

% Parameters for the dichotomic search.
% For given initial times tA, tB such that v(tA) > 0, v(tB) < 0,
% we iteratively evaluate v at tMed = (tA + tB)/2 until we find
% v(tMed) = 0. Therefore: tf = tMed.
tA = 1.;
tB = 500.;
dichotomyFuncTA = dichotomyFunc(tA);
dichotomyFuncTB = dichotomyFunc(tB);
tMed = (tA + tB)/2.;
dichotomyFuncTMed = dichotomyFunc(tMed);
iterDichotomy = 1;
iterDichotomyMax = 1000;
epsDichotomy = 1e-1;

if dichotomyFuncTA < 0 || dichotomyFuncTB > 0
    fprintf('Wrong guess times tA and tB! Choose them such that: v(tA) > 0 and v(tB) < 0...\n',iterDichotomy);
else
    % Classical dichotomic/binary/bisection search
    while ( abs(dichotomyFuncTMed) > epsDichotomy && iterDichotomy < iterDichotomyMax )
        % TODO: Implement dichotomic search. See initialization in 
        % lines 32-37 for reference. 
    end
    tf = tMed;

    % Optimal switching time.
    tSw = (m0 - mf)/(b*uMax);
    if tSw > tf % Verifying that: 0 < tSw <= tf
        tSw = tf;
    end

    % Plotting
    fprintf('Switching time tSw = %f\n',tSw);
    fprintf('Final time tf = %f\n',tf);
    options = odeset('AbsTol',1e-9,'RelTol',1e-9);
    [t,x] = ode113(@(t,x) Xdyn(t,x,tf), [0 tf], [h0;v0;m0], options);
    subplot(221); plot(t,x(:,1),'linewidth',3);
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
        if t(i) <= tSw % Optimal control from our optimal policy
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
end
