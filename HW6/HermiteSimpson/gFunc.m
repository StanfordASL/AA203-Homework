% Earth's gravity force

function g = gFunc(y)

global mu;
global rE;

g = mu/((rE + y)*(rE + y));