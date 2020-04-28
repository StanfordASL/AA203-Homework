% Dubins Car Model - AA203


%% Grid: generate a box-type grid of lower corner 'grid_min' and upper corner 'grid_max'

grid_min = [-2; -2.5; -pi]; % Lower corner of computation domain
grid_max = [5; 4; pi]; % Upper corner of computation domain
N = [45; 45; 35]; % Number of grid points per dimension
pdDims = 3; % 3rd dimension is periodic
g = createGrid(grid_min, grid_max, N, pdDims); % Generate the grid


%% Target set

toler = [0.2; 0.2; pi/10];
goal = [2; 2; 3*pi/8];
lower = goal - toler;
upper = goal + toler;
data0 = shapeRectangleByCorners(g, lower, upper);


%% Time vector

t0 = 0;
tMax = 10;
dt = 0.05;
tau = t0:dt:tMax;


%% Problem parameters

speed = 0.4; % Speed v of the model
uMax = 0.5; % Control bound
dMax = [0.05; 0.; 0.]; % Disturbance bound: the Dubins model of this library
                       % takes three disturbance bounds, one for each
                       % dimension. However, we have disturbance only on
                       % variable 'x'
uMode = 'min'; % Minimize on controls
dMode = 'max'; % Maximize on disturbances


%% Pack problem parameters

% Define dynamic system
x0 = [0;0;0]; % Starting point
dCar = DubinsCar(x0, uMax, speed, dMax);

% Put grid and dynamic systems into schemeData
schemeData.grid = g;
schemeData.dynSys = dCar;
schemeData.accuracy = 'veryHigh'; %set accuracy
schemeData.uMode = uMode;
schemeData.dMode = dMode;


%% Compute value function

HJIextraArgs.visualize = true; %show plot
HJIextraArgs.fig_num = 1; %set figure number
HJIextraArgs.deleteLastPlot = true; %delete previous plot as you update
data = HJIPDE_solve(data0, tau, schemeData, 'minVWithTarget', HJIextraArgs);
save('dubinsAA203.mat', 'tau', 'g', 'data')


%% Visualize slices

load('dubinsAA203.mat')

% Section for J(t=-2.5)
figure;
ind = find(tau==2.5);
visSetIm(g, data(:,:,:,ind));

% Section for J(t=-5)
figure;
ind = find(tau==5);
visSetIm(g, data(:,:,:,ind));

% Section for J(t=-0)
figure;
ind = find(tau==10);
visSetIm(g, data(:,:,:,ind));