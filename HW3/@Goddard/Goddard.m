classdef Goddard < DynSys
  properties
    % Constant gravity
    g
    
    % Constant thrust ratio
    b
      
    % Control bounds
    uMax
    
    % Disturbance
    dMax
    
    % Dimensions that are active
    dims
  end
  
  methods
    function obj = Goddard(x, g, b, uMax, dMax)
      % obj = Goddard(x, g, b, uMax, dMax)
      % Goddard Launcher class
      %
      % Dynamics:
      %    \dot{h} = v
      %    \dot{v} = u/m - g + d
      %    \dot{m} = -b * u
      %         u \in [0, uMax]
      %         d \in [-dMax, dMax]
      %
      % Inputs:
      %   x      - state: [h; v; m]
      %   g      - gravity
      %   b      - constant thrust ratio
      %   uMax   - control bounds
      %   dMax   - disturbance bounds
      %
      % Output:
      %   obj    - a GoddardLauncher object
      
      if numel(x) ~= obj.nx
        error('Initial state does not have right dimension!');
      end
      
      if ~iscolumn(x)
        x = x';
      end
      
      % Basic vehicle properties
      dims = 1:3;
      obj.nx = 3;
      obj.nu = 1;
      obj.nd = 1;
      
      obj.x = x;
      obj.xhist = obj.x;
      
      obj.dims = dims;
      obj.g = g;
      obj.b = b;
      obj.uMax = uMax;
      obj.dMax = dMax;
    end
    
  end % end methods
end % end classdef
