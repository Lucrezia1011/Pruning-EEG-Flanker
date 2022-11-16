function [tfce] = matlab_tfce_transform(img,H,E,C,dh, varargin)
%MATLAB_TFCE_TRANSFORM performs threshold free cluster enhancement
%   [tfced] = matlab_tfce_transform(img,H,E,C,dh) performs threshold
%   free cluster enhancement on 'img' as per Smith & Nichols (2009).
%   -- img the 3D image to be transformed
%   -- H height exponent,   H = 2
%   -- E extent exponent,   E = 0.5
%   -- C connectivity,      C = 26
%   -- dh size of steps for cluster formation   dh = 0.1
% https://github.com/markallenthornton/MatlabTFCE/blob/master/matlab_tfce_transform.m
if nargin > 5 && size(img,3) == 1
   gridall = varargin{1};
   dims = varargin{2};
   img_temp = zeros(size(gridall));
   img_temp(gridall == 1) = img;
   img_temp = reshape(img_temp, dims);
   img = img_temp;
   
end

tfce  = zeros(size(img));
for ii =1:2
if ii ==2
    img = -img;
end
% set cluster thresholds
threshs = 0:dh:max(img(:));
threshs = threshs(2:end);
ndh = length(threshs);

% find positive voxels (greater than first threshold)
nvox = length(img(:));

% find connected components
vals = zeros(nvox,1);
cc = arrayfun(@(x) bwconncomp(bsxfun(@ge,img,x),C), threshs);
for h = 1:ndh
    clustsize = zeros(nvox,1);
    ccc = cc(h);
    voxpercc = cellfun(@numel,ccc.PixelIdxList);
    for c = 1:ccc.NumObjects
        clustsize(ccc.PixelIdxList{c}) = voxpercc(c);
    end
    % calculate transform
    curvals = (clustsize.^E).*(threshs(h)^H);
    vals = vals + curvals;
end
tfced = NaN(size(img));
tfced(:) = vals.*dh;
if ii==1
    tfce = tfced;
else
    tfce = tfce - tfced;
end
end

end