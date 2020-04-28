function dOpt = optDstb(obj, ~, ~, deriv, dMode)
% dOpt = optCtrl(obj, t, y, deriv, dMode)
%     Dynamics
%         \dot{h} = v
%         \dot{v} = u/m - g + d
%         \dot{m} = -b * u

%% Input processing
if nargin < 5
  dMode = 'max';
end

if ~iscell(deriv)
  deriv = num2cell(deriv);
end

dOpt = cell(obj.nd, 1);

%% Optimal disturbance
if strcmp(dMode, 'max')
  dOpt = (deriv{obj.dims==2}>=0)*obj.dMax + (deriv{obj.dims==2}<0)*(-obj.dMax);
elseif strcmp(dMode, 'min')
  dOpt = (deriv{obj.dims==2}>=0)*(-obj.dMax) + (deriv{obj.dims==2}<0)*obj.dMax;
else
  error('Unknown uMode!')
end

end