% Normalized exponential model for the air density

function rho = normRhoFunc(y)

global h0;

rho = exp(-y/h0);