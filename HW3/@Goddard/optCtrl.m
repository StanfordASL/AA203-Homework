function uOpt = optCtrl(obj, ~, ~, deriv, uMode)
% uOpt = optCtrl(obj, t, y, deriv, uMode)

%% Input processing
if nargin < 5
  uMode = 'min';
end

if ~iscell(deriv)
  deriv = num2cell(deriv);
end

%% Optimal control
if strcmp(uMode, 'min')
  uOpt = ((deriv{obj.dims==2} - obj.b*deriv{obj.dims==3})<0)*obj.uMax;
elseif strcmp(uMode, 'max')
  uOpt = ((deriv{obj.dims==2} - obj.b*deriv{obj.dims==3})>=0)*obj.uMax;
else
  error('Unknown uMode!')
end

end