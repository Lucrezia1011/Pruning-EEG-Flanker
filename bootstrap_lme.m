function npeaks = bootstrap_lme(Atb,timevar,timevarname,N,lmefunc,fixeff,varargin)

if nargin == 6
    H =2; E =0.5; dh =0.1; C = 4;
else
    H = varargin{1};
    E = varargin{2};
    dh = varargin{3};
    C = varargin{4};
end

npeaks = cell(N,1);
nvars = size(Atb,1);
Atb.(timevarname) = zeros(nvars,1);

if strcmp(timevarname,fixeff)  % time varying fix effect
    parfor n = 1:N
        Atb_shuff = Atb;
        indshuff = randi(nvars,nvars,1);
        timeshuff = timevar(indshuff,:);
        teig_shuff = zeros(1,size(timevar,2));
        for t = 1:size(timevar,2)
            Atb_shuff.(fixeff) = timeshuff(:,t);
            lme_shuff = fitlme(Atb_shuff, lmefunc );
            teig_shuff(t) = lme_shuff.Coefficients.tStat(strcmp(lme_shuff.CoefficientNames,fixeff));
        end
        [tfce_shuff] = matlab_tfce_transform(teig_shuff,H,E,C,dh);
        npeaks{n}= [min(tfce_shuff), max(tfce_shuff)];
        
    end
    
else
    parfor n = 1:N
        Atb_shuff = Atb;
        Atb_shuff.(fixeff) = Atb.(fixeff)(randi(nvars,nvars,1));        
        teig_shuff = zeros(1,size(timevar,2));
        for t = 1:size(timevar,2)
            Atb_shuff.(timevarname) = timevar(:,t);
            lme_shuff = fitlme(Atb_shuff, lmefunc );
            teig_shuff(t) = lme_shuff.Coefficients.tStat(strcmp(lme_shuff.CoefficientNames,fixeff));
        end
        [tfce_shuff] = matlab_tfce_transform(teig_shuff,H,E,C,dh);
        npeaks{n} = [min(tfce_shuff), max(tfce_shuff)];
        
    end
    
end
npeaks = cell2mat(npeaks);

end


