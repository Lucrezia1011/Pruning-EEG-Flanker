clearvars % clear matlab workspace
clc % clear matlab command window


datapath = '/data/liuzzil2/UMD_Flanker/datatogit/';

% load PCA components
load([datapath,'PCA_coeffs.mat'])

tstart = -0.4;% trial start time for the autoregressor
tend = 0.6; % trial end time for the autoregressor


normalizeopt = 2; % normalized overall variance of residuals
ncomp = 2; % number of ICA components to keep

subject = 5;% subject to load
tbse = [tstart, tend]; % 0 mean in the same window of analysis


downsampf = 30; % downsampling frequency (2*lowpass frequency)


twind = 0.1; % test different window lenghts? (Shorter or longer attractors?)
tstep = round( downsampf * 0.04) ;
xstep = 1; %t+xstep sample to predict

%%

load(sprintf('%s/data%d_lowpass%dHz_sampling%dHz.mat',datapath,subject,downsampf/2,downsampf))
[~,t1] =  min( abs(erp.time - tstart));
[~,te] =  min( abs(erp.time - (tend - twind)));
%                     t1 = t1 -1 ;
%                     te = te + 1;
if te >= length(erp.time)
    te =  length(erp.time) - 1;
end
twindsamp = ceil(twind*downsampf);

tb = (t1 : tstep : te  );
time = erp.time(tb + round(twindsamp/2) );


erppca = zeros(ncomp,length(erp.sampleinfo),length(erp.time));


for cc = 1:ncomp
    erppca(cc,:,:) = squeeze( mean(erp.trial .* coeff(:,cc)' ,2) );

end

%
if normalizeopt == 1 % normalize each component, already 0-meaned in window of interest
    erptemp = erppca(:,:,erp.time>= tstart & erp.time <= tend);
    erppca = ( erppca - mean(erptemp,3)) / std(erptemp(:));   % norm1

elseif normalizeopt == 2
    erptemp = erppca(:,:,erp.time>= tstart & erp.time <= tend);
    erppca =( erppca - mean(erptemp,3));
    erptemp = erppca(:,:,erp.time>= tstart & erp.time <= tend);
    erptemp = reshape(permute(erptemp,[1,3,2]), [ncomp,size(erptemp,2)*size(erptemp,3)]);
    erppca =  erppca  ./ std(erptemp,0,2);
end
erptemp = erppca(:,:,erp.time>= tstart & erp.time <= tend);
[N,edges] =histcounts(erptemp(:),'Normalization','pdf');
edges = (edges(1:end-1) + edges(2:end))/2;


%%


condname = 'All'; % Example withouth separating task conditions
datapc =  mean(erppca,2);
x = erppca - datapc;

A = zeros(ncomp,ncomp,length(tb));
pvalue = zeros(ncomp,length(tb));
F = zeros(ncomp,length(tb));


n = size(x,2)*twind*downsampf;%numel(Pt); % number of observations
p = ncomp;%numel(a); % number of regression parameters
DFE = n - p;% degrees of freedom for error
DFM = p -1; % corrected degrees of freedom form model

eigv = zeros(ncomp, length(tb));
rmse = zeros(ncomp, length(tb));

for t = 1:length(tb)
    Pt = x(:,:,tb(t) + (0:(twindsamp-1)));
    % make xstep a vector?? use instead of components?
    Pt1 = x(:,:,tb(t) + xstep + (0:twindsamp-1)) ;



    Pt = reshape(Pt,[ncomp,size(x,2)*twindsamp ])';
    Pt1 = reshape(Pt1,[ncomp,size(x,2)*twindsamp])';


    a = Pt\Pt1;  % inv(Pt) * Pt1
    A(:,:,t) = a;
    eigv(:,t) = eig(a);

    %  [V,D] = eig(a); % A*V = V*D.
    yhat = Pt*a;


    SSM = sum( (yhat - mean(Pt1,1)).^2); % corrected sum of squares
    SSE = sum( (yhat - Pt1).^2 ); % sum of squares of residuals
    rmse(:,t) = sqrt(  mean( (yhat - Pt1).^2 ));
    MSM = SSM ./ DFM; % mean of squares for model
    MSE = SSE ./ DFE;% mean of squares or error
    F(:,t) = MSM ./ MSE;  %(explained variance) / (unexplained variance)

end

[~,tt,~] = intersect(erp.time, time);
Aall.A = A;
Aall.eig = eigv;
Aall.rmse = rmse;
Aall.time = time;
Aall.erpm = squeeze(datapc(:,:,tt));
Aall.residualm = squeeze(mean(x(:,:,tt),2)); % mean over trials
Aall.residuals = squeeze(std(x(:,:,tt),[],2)); % std over trials
Aall.erphist = [edges;N];
%                         Aall.ff = ff;
%                         Aall.Presidual = mean(pp,3);
%                         Aall.pvalue = pvalue;
Aall.ntrials = size(x,2);
Aall.F = F;
Aall.DFM = DFM;
Aall.DFE = DFE;

figure
plot(Aall.time, abs(Aall.eig)); xlabel('time(s)'); ylabel('eigenvalues')


