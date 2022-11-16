function npeaks = bootstrap_lme_multitest(Atb,timevar,timevarname,N,fixeff,varargin)

if nargin == 5
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


parfor n = 1:N
    Atb_shuff = Atb;
    indshuff = randi(nvars,nvars,1);
    tfce_shuff = zeros(length(fixeff),size(timevar,2));
    for l = 1:length(fixeff)
        Atb_shuff.(fixeff{l}) = Atb.(fixeff{l})(indshuff);
        teig_shuff = zeros(1,size(timevar,2));

        lmefunc =  sprintf('eig ~ %s + (1|sub)',fixeff{l});
        for t = 1:size(timevar,2)
            Atb_shuff.(timevarname) = timevar(:,t);

            lme_shuff = fitlme(Atb_shuff, lmefunc );
            teig_shuff(t) = lme_shuff.Coefficients.tStat(strcmp(lme_shuff.CoefficientNames,fixeff{l}));
        end
        tfce_shuff(l,:) = matlab_tfce_transform(teig_shuff,H,E,C,dh);
    end
    npeaks{n} = [min(tfce_shuff(:)), max(tfce_shuff(:))];
    
end

npeaks = cell2mat(npeaks);

end


