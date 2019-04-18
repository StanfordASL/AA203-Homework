function dx = dynamics(obj, ~, x, u, d)
% Dynamics of the Dubins Car
%    \dot{h} = v
%    \dot{v} = u/m - g + d
%    \dot{m} = -b * u
%   Control: u;
%
% Riccardo Bonalli, 2019-04-11

if iscell(x)
  dx = cell(length(obj.dims), 1);
  
  for i = 1:length(obj.dims)
    dx{i} = dynamics_cell_helper(obj, x, u, d, obj.dims, obj.dims(i));
  end
else
    dx = zeros(obj.nx, 1);

    dx(1) = x(2);
    dx(2) = u/x(3) - obj.g + d;
    dx(3) = -obj.b * u;
end
end

function dx = dynamics_cell_helper(obj, x, u, d, dims, dim)

switch dim
  case 1
    dx = x{dims==2};
  case 2
    dx = u./x{dims==3} - obj.g + d;
  case 3
    dx = -obj.b * u;
  otherwise
    error('Only dimension 1-3 are defined for dynamics of Goddard!')
end
end