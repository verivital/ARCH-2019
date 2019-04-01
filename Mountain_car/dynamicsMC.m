function [dx] = dynamicsMC(t,x,u)
% Mountain car dynamics
% x(1) = position
% x(2) = velocity

% Originally the discrete dynamics were taken from here
% https://perma.cc/6Z2N-PFWC, where the time step is 0.5 seconds
% So, we need to divide our dynamics by 0.5 to have them in continuous time

dx(1,1) = 2*x(2);
dx(2,1) = 0.003*u-0.005*cos(3*x(1));
end

