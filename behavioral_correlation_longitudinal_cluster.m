% Plot final temporal cluster analysis

clear all
datapath = '/data/liuzzil2/UMD_Flanker/derivatives/';
cd(datapath)
subsdir = dir;
subsdir(1:2) = [];
subslist = cell(length(subsdir ) ,1);
close all
% only include particIpants with at least 70% accuracy
behaveTable = readtable('/data/liuzzil2/UMD_Flanker/results/behavioral_table_accuracy70.csv');

samplings = 30 ; % sampling frequency 
stimnames =  {'cue','flan','resp'};

nf = length(stimnames);


normalizeopt  = 2; % Final version = 2
nbonf = 18; % Bonferroni corrections

conds = {'IncongCorr'}; % all trial types
% e.g.: conds = {'Correct'; 'Incongruent';'Congruent'; 'IncongCorr';'CongCorr'};

for iic = 1:3
   
  
    cond = conds{iic};

    ncomp = 4;% number of PCA components
    cc = 1; % Component to plot


    downsampf = samplings;
    twind = 0.1; % time window of dyanmics analysis = 0.1s
   
    tstep = round( downsampf * 0.04) ;
    xstep =  1; % time step: 1 = next time point t+1
  
    for ss  = 1:length(stimnames)
       
       
        stimname =  stimnames{ss};
        if strcmp(stimname, 'cue')
            N = 10000;
        else
            N = 20000;
        end
        
        if strcmp(stimname,'flan')
            xl = [-0.4 0.6];
            load('/data/liuzzil2/UMD_Flanker/results/flan_grandavg_erp.mat')
%             load('/data/liuzzil2/UMD_Flanker/results/pcaSubConcat_correct_congruent_n335.mat')
            t1 = -0.5; t2 = 0;
        elseif  strcmp(stimname,'cue')   
            xl = [-0.6 0.6];
            load('/data/liuzzil2/UMD_Flanker/results/cue_grandavg_erp.mat')
%             load('/data/liuzzil2/UMD_Flanker/results/pcaSubConcat_correct_congruent_n335.mat')
            
        else
            xl = [-0.5 0.5];
            load('/data/liuzzil2/UMD_Flanker/results/resp_grandavg_erp.mat')
%             load('/data/liuzzil2/UMD_Flanker/results/pcaSubConcat_resp_correct_commiss_n317.mat')
            t1 = -0.6; t2 = -0.1;
        end
              
        %%
 
        Atb = table;
        
        eig =  [];
%         rmse = [];
%         EEGresidual = [];
%         Presidual = [];
        
%         xx = 0:0.01:1000;
        for jj = 1:length(subsdir)
            sub = subsdir(jj).name(5:end);
           
            for age = [12,15,18]
                
                agegroup = num2str(age);
                
                subdir = ['sub-',sub];
                
                filebids = [subdir,'_task-flanker_eeg'];
                
                outputfolder = sprintf('/data/liuzzil2/UMD_Flanker/derivatives/sub-%s/age-%s/',sub,agegroup);
                
                filename = sprintf('%sAtest3_%s_%delec_f%dHzlowp_norm%d_w%dms_step%d_xstep%d.mat',...
                        outputfolder,stimname,ncomp,downsampf,normalizeopt,twind*1000,tstep,xstep);
                    
       
                bv = behaveTable( behaveTable.sdan == str2double(sub) & behaveTable.age == age,:);
                if exist( filename,'file') && ~isempty(bv)
                    
                  
                    load(filename)
                    
                    %                 yy = fpdf(xx,Aall.(cond).DFM,Aall.(cond).DFE);
                    %                 pvalue = zeros(size(Aall.(cond).F));
                    %                 for k = 1:size(Aall.(cond).F,2)
                    %                     [~,indf] = min(abs(Aall.(cond).F(:,k) - xx),[],2);
                    %                     for ik = 1:size(Aall.(cond).F,1)
                    %                         pvalue(ik,k) = trapz(xx(indf(ik):end), yy(indf(ik):end));
                    %                     end
                    %                 end
%                     if size(Aall.(cond).EEGresidual,2) == 1
%                         Aall.(cond).EEGresidual = Aall.(cond).EEGresidual';
%                     end
                    
                    
                    eig = cat(1,eig, ...
                        sum( abs(Aall.(cond).eig(cc,:)) ,1 ));
%                     rmse = cat(1,rmse,...
%                         mean( Aall.(cond).rmse(cc,:) ,1 ));
                    %                 EEGresidual = cat(1,EEGresidual,...
                    %                     mean(mean(abs(Aall.(cond).EEGresidual(cc,:,:)),1),2));
%                     EEGresidual = cat(1,EEGresidual,...
%                         mean(Aall.(cond).EEGresidual(cc,:),1));
                    %                 pvalue = cat(1,pvalue, ...
                    %                     Aall.(cond).pvalue);
                    
                    A = table;
                    A.sub = str2double(sub);
                    A.age = age;
                    A.ntrials = Aall.(cond).ntrials;
                    A.accuracyI = (bv.accuracy_IL + bv.accuracy_IR)/2;
                    A.accuracyC = (bv.accuracy_CL + bv.accuracy_CR)/2;
                    
                    if strcmp(cond, 'IncongCorr') || strcmp(cond, 'Incongruent')
                        A.RT = bv.RT_incongruent;
                        A.RTs = bv.RT_incongruent_std;
                    elseif strcmp(cond, 'CongCorr') || strcmp(cond, 'Congruent')
                        A.RT = bv.RT_congruent;
                        A.RTs = bv.RT_congruent_std;
                    elseif strcmp(cond, 'Commission')
                        A.RT = bv.RT_commission;
                        A.RTs = bv.RT_commission_std;
                    elseif strcmp(cond, 'Correct')
                        A.RT = bv.RT_correct;
                        A.RTs = bv.RT_correct_std;
                    else
                        A.RT = (bv.RT_congruent + bv.RT_incongruent)/2;
                        A.RTs = (bv.RT_congruent_std + bv.RT_incongruent_std)/2;
                    end
                    
%                     if isfield(Aall.(cond),'Presidual')
%                         Presidual = mean(Aall.(cond).Presidual(cc,:),1);
%                     end

                    Atb = cat(1,Atb,A);
                    
                end
            end
        end
        
        if strcmp(cond, 'IncongCorr') || strcmp(cond, 'Incongruent')
            Atb.accuracy =  Atb.accuracyI;
        elseif strcmp(cond, 'CongCorr') || strcmp(cond, 'Congruent')
            Atb.accuracy =  Atb.accuracyC;
        else
            Atb.accuracy =  (Atb.accuracyC + Atb.accuracyI) / 2;
        end
        time = Aall.All.time;
%         if strcmp(stimname,'flan')
%             timefull = linspace(-0.5,1,size(EEGresidual,2));
%         elseif  strcmp(stimname,'resp')
%             timefull = linspace(-1,1,size(EEGresidual,2));
%         else
%             timefull = linspace(-0.6,0.4,size(EEGresidual,2));
%         end
        % figure
        % plot(Aall.CongCorr.ff, mean(A.Presidual,1))
        
        
        tots_ages = readtable('/data/liuzzil2/UMD_Flanker/results/TOTS_12yrs_15yrs_18yrs_ages_days_months_years.xlsx');
        new_ages = readtable('/data/liuzzil2/UMD_Flanker/results/new_ages_flanker_12_15yrs.xlsx');
        
        Atb.aged = zeros(size(Atb.age));
        for n = 1:length(Atb.age)
            if  Atb.age(n) == 12
                ag = tots_ages.x12yEEGAgeYears(tots_ages.ID ==  Atb.sub(n));
                if isempty(ag)
                    ag = new_ages.age_at_12yr_visit_years( new_ages.subject ==  Atb.sub(n) );
                elseif isnan(ag)
                    ag = new_ages.age_at_12yr_visit_years( new_ages.subject ==  Atb.sub(n) );
                end
                
            elseif Atb.age(n) == 15
                ag =  tots_ages.x15yEEGAgeYears(tots_ages.ID ==  Atb.sub(n));
                if isempty(ag)
                    ag = new_ages.age_at_15yr_visit_years( new_ages.subject == Atb.sub(n) );
                elseif isnan(ag)
                    ag = new_ages.age_at_15yr_visit_years( new_ages.subject ==  Atb.sub(n) );
                end
                
            else
                ag = tots_ages.x18yEEGAgeMonths_1(tots_ages.ID ==  Atb.sub(n));
                
                
            end
            
            if isnan(ag)
                ag = Atb.age(n);
            end
            Atb.aged(n) = ag;
            
        end
        
        
        
        %% RT and RTs correlation
        % figure; set(gcf,'color','w');
        % scatter(Atb.RT, Atb.RTs)
        % xlabel('mean(RT) (ms)'); ylabel('stdev(RT) (ms)')
        %
        % [P,S] = polyfit(Atb.RT,Atb.RTs,1);
        % yfit =  P(1) * Atb.RT + P(2);
        % hold on
        % plot(Atb.RT, yfit)
        % yresid = Atb.RTs - yfit;
        % SSresid = sum(yresid.^2);
        % SStotal = (length(Atb.RTs)-1) * var(Atb.RTs);
        % rsq = 1 - SSresid/SStotal;
        % title(sprintf('Correlation of mean(RT) and stdev(RT): R^2=%.2f',rsq))
        % saveas(gcf,'/data/liuzzil2/UMD_Flanker/results/RT_RTs_Corr.jpg');
        
   
        %% Uncomment from here
%         A = table;
%         A.sub = Atb.sub;
%         A.age = Atb.age;
%         A.accuracy = Atb.accuracy;
%         A.RT = Atb.RT;
%         A.RTs = Atb.RTs;
%         
%         teig_long = zeros(size(time));
%         peig_long = zeros(size(time));
%         pacc_long = zeros(size(time));
%         accorr_long =zeros(size(time));
%         prt_long = zeros(size(time));
%         rtcorr_long= zeros(size(time));
%         prts_long = zeros(size(time));
%         rtscorr_long= zeros(size(time));
%         for t = 1:length(time)
%             A.eig = eig(:,t);
%             lme_agelong = fitlme(A, 'eig ~ age + (1|sub)' );
%             lme_acclong = fitlme(A, 'accuracy ~ eig + (1|sub)' );
%             lme_rtlong = fitlme(A, 'RT ~ eig + (1|sub)' );
%             lme_rtslong = fitlme(A, 'RTs ~ eig + (1|sub)' );
%             teig_long(t) = lme_agelong.Coefficients.tStat(strcmp(lme_agelong.CoefficientNames,'age'));
%             peig_long(t) = lme_agelong.Coefficients.pValue(strcmp(lme_agelong.CoefficientNames,'age'));
%             pacc_long(t) = lme_acclong.Coefficients.pValue(strcmp(lme_acclong.CoefficientNames,'eig'));
%             accorr_long(t) = sqrt(lme_acclong.Coefficients.tStat(strcmp(lme_acclong.CoefficientNames,'eig'))^2 / ...
%                 (lme_acclong.Coefficients.tStat(strcmp(lme_acclong.CoefficientNames,'eig'))^2 + ...
%                 lme_acclong.Coefficients.DF(strcmp(lme_acclong.CoefficientNames,'eig')))) * ...
%                 sign(lme_acclong.Coefficients.tStat(strcmp(lme_acclong.CoefficientNames,'eig')));
%             prt_long(t) = lme_rtlong.Coefficients.pValue(strcmp(lme_rtlong.CoefficientNames,'eig'));
%             rtcorr_long(t) = sqrt(lme_rtlong.Coefficients.tStat(strcmp(lme_rtlong.CoefficientNames,'eig'))^2 / ...
%                 (lme_rtlong.Coefficients.tStat(strcmp(lme_rtlong.CoefficientNames,'eig'))^2 + ...
%                 lme_rtlong.Coefficients.DF(strcmp(lme_rtlong.CoefficientNames,'eig')))) * ...
%                 sign(lme_rtlong.Coefficients.tStat(strcmp(lme_rtlong.CoefficientNames,'eig')));
%             prts_long(t) = lme_rtslong.Coefficients.pValue(strcmp(lme_rtslong.CoefficientNames,'eig'));
%             rtscorr_long(t) = sqrt(lme_rtslong.Coefficients.tStat(strcmp(lme_rtslong.CoefficientNames,'eig'))^2 / ...
%                 (lme_rtslong.Coefficients.tStat(strcmp(lme_rtslong.CoefficientNames,'eig'))^2 + ...
%                 lme_rtslong.Coefficients.DF(strcmp(lme_rtslong.CoefficientNames,'eig')))) * ...
%                 sign(lme_rtslong.Coefficients.tStat(strcmp(lme_rtslong.CoefficientNames,'eig')));
%         end
%         
        %%
        A = table;
        A.sub = Atb.sub;
        A.age = zscore(Atb.aged);
        A.accuracy = zscore(Atb.accuracy);
        A.RT = zscore(Atb.RT);
        A.RTs = zscore(Atb.RTs);
        teig_long = zeros(size(time));
        eigeff_long = zeros(2,size(time,2));
        accorr_long = zeros(size(time));
        accseff_long = zeros(2,size(time,2));
        rtscorr_long = zeros(size(time));
        rtseff_long = zeros(2,size(time,2));
        rtcorr_long = zeros(size(time));
        rteff_long = zeros(2,size(time,2));
        for t = 1:length(time)
            
            A.eig = eig(:,t);
            A.eig = zscore(A.eig);
            
            lme_agelong = fitlme(A, 'eig ~ age + (1|sub)' );
            teig_long(t) = lme_agelong.Coefficients.tStat(strcmp(lme_agelong.CoefficientNames,'age'));
            eigeff_long(:,t) = lme_agelong.Coefficients.Estimate;
            lme_acclong = fitlme(A, 'accuracy ~ eig  + (1|sub)' );
            
            accorr_long(t) = lme_acclong.Coefficients.tStat(strcmp(lme_acclong.CoefficientNames,'eig'));
            accseff_long(:,t) = lme_acclong.Coefficients.Estimate(...
                strcmp(lme_acclong.CoefficientNames,'(Intercept)')  | strcmp(lme_acclong.CoefficientNames,'eig'));
            lme_rtslong = fitlme(A, 'RTs ~ eig  + (1|sub)' );
            
            rtscorr_long(t) = lme_rtslong.Coefficients.tStat(strcmp(lme_rtslong.CoefficientNames,'eig'));
            rtseff_long(:,t) = lme_rtslong.Coefficients.Estimate(...
                strcmp(lme_rtslong.CoefficientNames,'(Intercept)') | strcmp(lme_rtslong.CoefficientNames,'eig'));
            
            lme_rtlong = fitlme(A, 'RT ~ eig  + (1|sub)' );
            rtcorr_long(t) = lme_rtlong.Coefficients.tStat(strcmp(lme_rtlong.CoefficientNames,'eig'));
            rteff_long(:,t) = lme_rtlong.Coefficients.Estimate(...
                strcmp(lme_rtlong.CoefficientNames,'(Intercept)') | strcmp(lme_rtlong.CoefficientNames,'eig'));
        end
        
%         t0 = 0.0147;
%         [~,t]= min(abs(time - t0));
%         A.eig = eig(:,t);
%         A.eig = zscore(A.eig);
%         lme = fitlme(A, 'RTs ~ eig  + (1|sub)' );    
        %% Bar plots
%         figure(1); set(gcf,'color','w')
%         subplot(1,nf,ss)
%         cla; hold all
%         co = get(gca,'colororder');
%         co(3,:) = co(5,:); co(5,:) = co(2,:);
%         set(gca,'colororder',co)
%         t0 = 0.0147;
%         [~,t]= min(abs(time - t0));
%         ages = [12, 15, 18];
%         for a = 1:3
%             bar(a, mean(eig(Atb.age == ages(a)  ,t) ))
%             errorbar(a, mean(eig(Atb.age == ages(a)  ,t) ), ...
%                 std(eig(Atb.age == ages(a)  ,t) )/sqrt(nnz(Atb.age == ages(a))),'k.')
%             
%         end 
%         set(gca,'Xtick',1:3,'Xticklabels',ages); 
%         xlabel('age group'); ylabel('eigenvalue'); 
%         title(sprintf('Eigenvalue at %d ms',round(time(t)*1e3)))
%         ylim([0.3 0.6])
%         
% %         subplot(2,nf,ss+nf)
% %         cla; hold all
% %         set(gca,'colororder',co)
% %         if ss == 2
% %             t0 = 0.4147;
% %         else
% %             t0 = 0.01;
% %         end
% %         [~,t]= min(abs(time - t0));
% %         ages = [12, 15, 18];
% %         for a = 1:3
% %             bar(a, mean(eig(Atb.age == ages(a)  ,t) ))
% %             errorbar(a, mean(eig(Atb.age == ages(a)  ,t) ), ...
% %                 std(eig(Atb.age == ages(a)  ,t) )/sqrt(nnz(Atb.age == ages(a))),'k.')
% %         end 
% %         set(gca,'Xtick',1:3,'Xticklabels',ages); 
% %         xlabel('age group'); ylabel('eigenvalue'); 
% %         title(sprintf('Eigenvalue at %d ms',round(time(t)*1e3)))
      
%% Scatter plot of RT vs age
%     figure(2); clf; set(gcf,'color', 'w')
%     subplot(121)
%     scatter(Atb.aged, Atb.RT)
%     lme = fitlme(Atb, 'RT ~ aged' );
% %     [r,p] = corr(Atb.aged, Atb.RT);
%     r = sign(lme.Coefficients.tStat(2))*sqrt(lme.Coefficients.tStat(2)^2 / (lme.Coefficients.tStat(2)^2 + lme.Coefficients.DF(2)));
%     hold on
%     plot([12,20],lme.Coefficients.Estimate(1) + lme.Coefficients.Estimate(2)*[12,20],'k')
%     xlabel('age (years)'); ylabel('mean(RT) (ms)')
%     title(sprintf('mean(RT) ~ age, R = %.2f, p = %.1s',r,lme.Coefficients.pValue(2) ))
%     subplot(122)
%     scatter(Atb.aged, Atb.RTs)
%     lmes = fitlme(Atb, 'RTs ~ aged' );
% %     [r,p] = corr(Atb.aged, Atb.RTs);
%     r = sign(lmes.Coefficients.tStat(2))*sqrt(lmes.Coefficients.tStat(2)^2 / (lmes.Coefficients.tStat(2)^2 + lmes.Coefficients.DF(2)));
%     hold on
%     plot([12,20],lmes.Coefficients.Estimate(1) + lmes.Coefficients.Estimate(2)*[12,20],'k')
%     xlabel('age (years)'); ylabel('stdev(RT) (ms)')
%     title(sprintf('stdev(RT) ~ age, R = %.2f, p = %.1s',r,lmes.Coefficients.pValue(2) ))
    
%      saveas(gcf,'/data/liuzzil2/UMD_Flanker/results/scatter_RT_age.jpg');
%      saveas(gcf,'/data/liuzzil2/UMD_Flanker/results/scatter_RT_age.fig');
        %%
%         addpath /data/liuzzil2/UMD_Flanker/matlab
%         H =2; E =0.5; dh =0.1; C = 4;
%         [tfce_age] = matlab_tfce_transform(teig_long,H,E,C,dh);
%         [tfce_acc] = matlab_tfce_transform(accorr_long,H,E,C,dh);
%         [tfce_rts] = matlab_tfce_transform(rtscorr_long,H,E,C,dh);
%         [tfce_rt] = matlab_tfce_transform(rtcorr_long,H,E,C,dh);
%        
%         
%         agenull_file = sprintf('/data/liuzzil2/UMD_Flanker/results/eig_fix-age_rand-sub_f%dHz_cond-%s_N%d_%s.mat',downsampf,cond,N,stimname);
% %         agenull_file2 = sprintf('/data/liuzzil2/UMD_Flanker/results/eig_fix-age_rand-sub_f%dHz_cond-%s_N%d_%s2.mat',downsampf,cond,N,stimname);
%         if ~exist(agenull_file,'file')
%             A = table;
%             A.sub = Atb.sub;
%             A.age = Atb.aged;
%             
%             lmefunc = 'eig ~ age + (1|sub)';
%             fixeff = 'age';
%             fprintf('Running N=%d bootstrap permutations:\n%s\n\n',N,lmefunc)
%             npeaks = bootstrap_lme(A,eig,'eig',N,lmefunc,fixeff);
%             save(agenull_file,'npeaks')
%         end
%         
%         rtsnull_file = sprintf('/data/liuzzil2/UMD_Flanker/results/RTs_fix-eig_rand-sub_f%dHz_cond-%s_N%d_%s.mat',downsampf,cond,N,stimname);
% %         rtsnull_file2 = sprintf('/data/liuzzil2/UMD_Flanker/results/RTs_fix-eig_rand-sub_f%dHz_cond-%s_N%d_%s2.mat',downsampf,cond,N,stimname);
%         if ~exist(rtsnull_file,'file')
%             A = table;
%             A.sub = Atb.sub;
%             A.RTs = Atb.RTs;
%             lmefunc = 'RTs ~ eig + (1|sub)';
%             fixeff = 'eig';
%             clc; fprintf('Running N=%d bootstrap permutations:\n%s\n\n',N,lmefunc)
%             npeaks = bootstrap_lme(A,eig,'eig',N,lmefunc,fixeff);
%             save(rtsnull_file,'npeaks')
%         end
%         
%         rtnull_file = sprintf('/data/liuzzil2/UMD_Flanker/results/RT_fix-eig_rand-sub_f%dHz_cond-%s_N%d_%s.mat',downsampf,cond,N,stimname);
% %         rtnull_file2 = sprintf('/data/liuzzil2/UMD_Flanker/results/RT_fix-eig_rand-sub_f%dHz_cond-%s_N%d_%s2.mat',downsampf,cond,N,stimname);
%         if ~exist(rtnull_file,'file')
%             A = table;
%             A.sub = Atb.sub;
%             A.RT = Atb.RT;
%             lmefunc = 'RT ~ eig + (1|sub)';
%             fixeff = 'eig';
%             clc; fprintf('Running N=%d bootstrap permutations:\n%s\n\n',N,lmefunc)
%             npeaks = bootstrap_lme(A,eig,'eig',N,lmefunc,fixeff);
%             save(rtnull_file,'npeaks')
%         end
%         
%         accnull_file = sprintf('/data/liuzzil2/UMD_Flanker/results/Accuracy_fix-eig_rand-sub_f%dHz_cond-%s_N%d_%s.mat',downsampf,cond,N,stimname);
% %         accnull_file2 = sprintf('/data/liuzzil2/UMD_Flanker/results/Accuracy_fix-eig_rand-sub_f%dHz_cond-%s_N%d_%s2.mat',downsampf,cond,N,stimname);
%         if ~exist(accnull_file,'file')
%             A = table;
%             A.sub = Atb.sub;
%             A.accuracy = Atb.accuracy;
%             
%             lmefunc = 'accuracy ~ eig + (1|sub)';
%             fixeff = 'eig';
%             fprintf('Running N=%d bootstrap permutations:\n%s\n\n',N,lmefunc)
%             npeaks = bootstrap_lme(A,eig,'eig',N,lmefunc,fixeff);
%             save(accnull_file,'npeaks')
%         end
%         
%       
        
        %%
%        
%         a = 0.05/nbonf;
%         
%         
%         load(rtnull_file)
%         npeaksrt = npeaks;
%         if size(npeaksrt,1) == 1
%             npeaksrt = [npeaksrt(1:2:end)',npeaksrt(2:2:end)'];
%         end
% %         load(rtnull_file2)
% %         npeaksrt = [npeaksrt; npeaks];
%         
%         load(rtsnull_file)
%         npeaksrts = npeaks;
%         if size(npeaksrts,1) == 1
%             npeaksrts = [npeaksrts(1:2:end)',npeaksrts(2:2:end)'];
%         end
% %         load(rtsnull_file2)
% %         npeaksrts = [npeaksrts; npeaks];
%         
%         load(accnull_file)
%         npeaksacc = npeaks;
%         if size(npeaksacc,1) == 1
%             npeaksacc = [npeaksacc(1:2:end)',npeaksacc(2:2:end)'];
%         end
% %         load(accnull_file2)
% %         npeaksacc = [npeaksacc; npeaks];
% %         
%         
%         load(agenull_file)
%         npeaksage = npeaks;
%         if size(npeaksage,1) == 1
%             npeaksage = [npeaksage(1:2:end)',npeaksage(2:2:end)'];
%         end
% %         load(agenull_file2)
% %         npeaksage = [npeaksage; npeaks];
%         
% %         npeaks = [npeaksrt;npeaksrts;npeaksacc;npeaksage];
% %         
% %         npeaksrt = npeaks;
% %         npeaksrts = npeaks;
% %         npeaksacc = npeaks;
% %         npeaksage = npeaks;
%         
% %     figure; set(gcf,'color','w','name',sprintf('Distributions %s f%dHz',stimname,downsampf))
% %         subplot(414)
% %         histogram(npeaksrts(:,2),'Normalization','pdf','DisplayStyle','stairs')
% %         hold on
% %         histogram(tfce_rts,'Normalization','probability','DisplayStyle','stairs')
%         rtsnull = sort(npeaksrts(:,2),'descend');
% %         plot([1,1]*rtsnull(ceil(a*N)) , [0,1],'k--')
% %         title('RTs ~ eig + (1|sub)')
% %         
% %         
% %         
% %         subplot(413)
% %         histogram(npeaksrt(:,2),'Normalization','pdf','DisplayStyle','stairs')
% %         hold on
% %         histogram(tfce_rt,'Normalization','probability','DisplayStyle','stairs')
%         rtnull = sort(npeaksrt(:,2),'descend');
% %         plot([1,1]*rtnull(ceil(a*N)) , [0,1],'k--')
% %         title('RT ~ eig + (1|sub)')
% %         
% %         
% %         
% %         
% %         subplot(412)
% %         histogram(npeaksacc(:,1),'Normalization','pdf','DisplayStyle','stairs')
% %         hold on
% %         histogram(tfce_acc,'Normalization','probability','DisplayStyle','stairs')
%         accnull = sort(npeaksacc(:,1),'ascend');
% %         plot([1,1]*accnull(ceil(a*N)) , [0,1],'k--')
% %         title('accuracy ~ eig + (1|sub)')
% %         legend('null','data',sprintf('a=%.4f',a))
% %         
% %         
% %         
% %         subplot(411)
% %         histogram(npeaksage(:,1),'Normalization','pdf','DisplayStyle','stairs')
% %         hold on
% %         histogram(tfce_age,'Normalization','probability','DisplayStyle','stairs')
%         agenull = sort(npeaksage(:,1),'ascend');
% %         plot([1,1]*agenull(ceil(a*N)) , [0,1],'k--')
% %         title('eig ~ age + (1|sub)')
%         
% %         
% %         saveas(gcf,sprintf('/data/liuzzil2/UMD_Flanker/results/NullDistributions_%s_%s_f%dHz_%dcomps_downsamp_norm_w%dms_PC%d.jpg',...
% %             stimname,cond,downsampf,ncomp,twind*1000,cc));
% %         
%         
        
        %%
%         ss = ff;
        
        figure(5);set(gcf, 'color' , 'w','position',[205          64        1420         855]) %[10    30   834   1134]
        subplot(4,nf,ss)
        ax = gca;
        co  = get(ax,'colororder');
        co(3,:) = co(2,:); co(2,:) = co(5,:);
        colororder(co);
      
        plot(time, cat(1,mean(eig(Atb.age==12,:),1),...
            mean(eig(Atb.age==15,:),1),mean(eig(Atb.age==18,:),1 )),'linewidth',2)
%         colororder({'k','k'})
        hold on;
        fill([time fliplr(time)],...
            [mean(eig(Atb.age==12,:),1)+std(eig(Atb.age==12,:))/sqrt(size(eig(Atb.age==12,:),1)),...
            fliplr(mean(eig(Atb.age==12,:),1)-std(eig(Atb.age==12,:))/sqrt(size(eig(Atb.age==12,:),1)))],...
            co(1,:),'edgecolor','none','facealpha',0.3)
        fill([time fliplr(time)],...
            [mean(eig(Atb.age==15,:),1)+std(eig(Atb.age==15,:))/sqrt(size(eig(Atb.age==15,:),1)),...
            fliplr(mean(eig(Atb.age==15,:),1)-std(eig(Atb.age==15,:))/sqrt(size(eig(Atb.age==15,:),1)))],...
            co(2,:),'edgecolor','none','facealpha',0.3)
        fill([time fliplr(time)],...
            [mean(eig(Atb.age==18,:),1)+std(eig(Atb.age==18,:))/sqrt(size(eig(Atb.age==18,:),1)),...
            fliplr(mean(eig(Atb.age==18,:),1)-std(eig(Atb.age==18,:))/sqrt(size(eig(Atb.age==18,:),1)))],...
            co(3,:),'edgecolor','none','facealpha',0.3)
        ylabel('eigenvalue');
        
        yyaxis right; ax.YAxis(2).Color = 'k'; hold all; 
        plot(time,teig_long,'k--','linewidth',1)
%         indsig = tfce_age < agenull(ceil(a*N));
%         plot(time(indsig),teig_long(indsig),'k*','linewidth',2)
        
        xlabel('time (s)'); ylabel('tstat'); grid on
        title('eig ~ age + (1|sub)')
        if ss == 1
        legend('12yo','15yo','18yo','location','northwest')
        end
        xlim(xl);
        
        subplot(4,nf,ss + nf)
        ax = gca;
        co(2,:) = co(3,:); colororder(co);

%         erp1 = sum(grandavg.all.all.avg .* coeff(:,1),1) ;
%         erp2 = sum(grandavg.all.all.avg .* coeff(:,2),1) ;
        
       
%         timerp = grandavg.age12.all.time;
      
%         plot(timerp, [erp1; erp2],'linewidth',2)
        ylabel('sum of potentials (\muV)')
       
        yyaxis right; hold all;ax.YAxis(2).Color = 'k';
        plot(time, accorr_long,'k--','linewidth',1)
%         indsig = tfce_acc < accnull(ceil(a*N));
%         plot(time(indsig),accorr_long(indsig),'k*','linewidth',2)
        xlim(xl);
        xlabel('time (s)'); ylabel('tstat'); grid on
        title('accuracy ~ eig + (1|sub)')
        
        
        
        subplot(4,nf,ss + (nf)*3)
        ax = gca;
       
%         plot(timerp, [erp1; erp2],'linewidth',2)
        ylabel('sum of potentials (\muV)')
        yyaxis right; hold all; ax.YAxis(2).Color = 'k';
        plot(time, rtscorr_long,'k--','linewidth',1)
%         indsig = tfce_rts > rtsnull(ceil(a*N));
%         plot(time(indsig),rtscorr_long(indsig),'k*','linewidth',2)
        xlim(xl);
        xlabel('time (s)'); ylabel('tstat'); grid on
        title('stdev(RT) ~ eig + (1|sub)')
        
        subplot(4,nf,ss + (nf)*2)
        ax = gca;
        
%         plot(timerp, [erp1; erp2],'linewidth',2); 
        ylabel('sum of potentials (\muV)')
        yyaxis right; hold all; ax.YAxis(2).Color = 'k';
        plot(time, rtcorr_long,'k--','linewidth',1)
%         indsig = tfce_rt > rtnull(ceil(a*N));
%         plot(time(indsig),rtcorr_long(indsig),'k*','linewidth',2)
        xlim(xl);
        xlabel('time (s)'); ylabel('tstat'); grid on
        title('mean(RT) ~ eig + (1|sub)')
        
        %%
        
        figure(3);  
        set(gcf, 'color' , 'w','position',[104    64   1668    355])
        subplot(1,nf,ss )
%         [~,t] = min(teig_long);
        [~,t]= min(abs(time - 0));
        A = table;
        A.age = Atb.aged;
        A.sub = Atb.sub;
        A.eig = eig(:,t);
        scatter(A.age, A.eig); hold all; x = xlim;
        
        lme_agelong = fitlme(A, 'eig ~ age + (1|sub)' );
        eigeff_long_t = lme_agelong.Coefficients.Estimate;
        
        plot(x,eigeff_long_t(1) + eigeff_long_t(2)*x,'linewidth',2)
        for jj = 1:length(subsdir)
            sub = subsdir(jj).name(5:end);
            inds = A.sub == str2double(sub);
            plot(A.age(inds), A.eig(inds),'k')
        end
        xlabel('age (years)'); ylabel('eigenvalue')
        title(sprintf('%s at time %dms',stimname,  round(time(t)*1000)))
      
               
        if ss > 1
        figure(4)  
        set(gcf, 'color' , 'w','position',[203    60   908   855]) 
        subplot(3,nf-1,ss-1 )
        [~,t] = min(accorr_long);
        A.eig = eig(:,t);
        A.accuracy = Atb.accuracy;
        scatter(A.eig, A.accuracy); hold all; x = xlim;
        lme_acclong = fitlme(A, 'accuracy ~ eig  + (1|sub)' );
            
            accseff_long_t = lme_acclong.Coefficients.Estimate(...
                strcmp(lme_acclong.CoefficientNames,'(Intercept)')  | strcmp(lme_acclong.CoefficientNames,'eig'));
            
        plot(x,accseff_long_t(1) + accseff_long_t(2)*x,'linewidth',2)
        for jj = 1:length(subsdir)
            sub = subsdir(jj).name(5:end);
            inds = A.sub == str2double(sub);
            [eigst,sind] = sort(A.eig(inds),'ascend');
            a = A.accuracy(inds);
            plot(eigst, a(sind),'k')
        end
        xlabel('eigenvalue'); ylabel('accuracy');
        title(sprintf('Accuracy %s at time %dms',stimname,round(time(t)*1000)))
        
        
        
        
        subplot(3,nf-1,ss-1 + (nf-1))
        
        [~,t] = max(rtcorr_long);
        
        A.eig = eig(:,t);
        A.RT = Atb.RT;
        scatter(A.eig, A.RT); hold all; x = xlim;
        
        lme_rtlong = fitlme(A, 'RT ~ eig  + (1|sub)' );
            
            rteff_long_t = lme_rtlong.Coefficients.Estimate(...
                strcmp(lme_rtlong.CoefficientNames,'(Intercept)') | strcmp(lme_rtlong.CoefficientNames,'eig'));
           
        
        
        plot(x,rteff_long_t(1) + rteff_long_t(2)*x,'linewidth',2)
        for jj = 1:length(subsdir)
            sub = subsdir(jj).name(5:end);
            inds = A.sub == str2double(sub);
            [eigst,sind] = sort(A.eig(inds),'ascend');
            a = A.RT(inds);
            plot(eigst, a(sind),'k')
        end
        xlabel('eigenvalue'); ylabel('mean(RT)')
        title(sprintf('mean(RT) %s at time %dms',stimname,round(time(t)*1000)))
        
        
        
        
        
        subplot(3,nf-1,ss-1 + 2*(nf-1))
        
        [~,t] = max(rtscorr_long);
        
        A.eig = eig(:,t);
        A.RTs = Atb.RTs;
        scatter(A.eig, A.RTs); hold all; x = xlim;
        
        lme_rtslong = fitlme(A, 'RTs ~ eig  + (1|sub)' );
            
            rtseff_long_t = lme_rtslong.Coefficients.Estimate(...
                strcmp(lme_rtslong.CoefficientNames,'(Intercept)') | strcmp(lme_rtslong.CoefficientNames,'eig'));
           
        
        
        plot(x,rtseff_long_t(1) + rtseff_long_t(2)*x,'linewidth',2)
        for jj = 1:length(subsdir)
            sub = subsdir(jj).name(5:end);
            inds = A.sub == str2double(sub);
            [eigst,sind] = sort(A.eig(inds),'ascend');
            a = A.RTs(inds);
            plot(eigst, a(sind),'k')
        end
        xlabel('eigenvalue'); ylabel('stdev(RT)')
        title(sprintf('stdev(RT) %s at time %dms',stimname,round(time(t)*1000)))
        end
%           saveas(gcf,sprintf('/data/liuzzil2/UMD_Flanker/results/LongitudinalAgeEffct_%s_%s_f%dHz_%dcomps_downsamp_norm_w%dms_PC%d.jpg',...
%             stimname,cond,downsampf,ncomp,twind*1000,cc));
%         
        
    end
%     figure(5)
%     saveas(gcf,sprintf('/data/liuzzil2/UMD_Flanker/results/ClusterCorr_nbonf%d_%dHz_%s_%dcomps_downsamp_norm%d_w%dms_PC%d.jpg',...
%         nbonf,downsampf,cond,ncomp,normalizeopt,twind*1000,cc));
%     
%     saveas(gcf,sprintf('/data/liuzzil2/UMD_Flanker/results/ClusterCorr_nbonf%d_%dHz_%s_%dcomps_downsamp_norm%d_w%dms_PC%d.fig',...
%         nbonf,downsampf,cond,ncomp,normalizeopt,twind*1000,cc));
% 
%     saveas(gcf,sprintf('/data/liuzzil2/UMD_Flanker/results/ClusterCorr_%dHz_%s_%dcomps_downsamp_norm%d_w%dms_PC%d.jpg',...
%         downsampf,'cue',ncomp,normalizeopt,twind*1000,cc));
%     figure(3)
%     saveas(gcf,sprintf('/data/liuzzil2/UMD_Flanker/results/AgeLongitudinalEffects_%s_f%dHz_%dcomps_downsamp_norm_w%dms_PC%d.jpg',...
%         cond,downsampf,ncomp,twind*1000,cc));
%     
%     saveas(gcf,sprintf('/data/liuzzil2/UMD_Flanker/results/AgeLongitudinalEffects_%s_f%dHz_%dcomps_downsamp_norm_w%dms_PC%d.fig',...
%         cond,downsampf,ncomp,twind*1000,cc));
%     
%     figure(4)
%     saveas(gcf,sprintf('/data/liuzzil2/UMD_Flanker/results/LongitudinalEffects_%s_f%dHz_%dcomps_downsamp_norm_w%dms_PC%d.jpg',...
%         cond,downsampf,ncomp,twind*1000,cc));
%     
%     saveas(gcf,sprintf('/data/liuzzil2/UMD_Flanker/results/LongitudinalEffects_%s_f%dHz_%dcomps_downsamp_norm_w%dms_PC%d.fig',...
%         cond,downsampf,ncomp,twind*1000,cc));
% 
% 
%     figure(1)
%     saveas(gcf,sprintf('/data/liuzzil2/UMD_Flanker/results/Eigbar_%s_f%dHz_%dcomps_downsamp_norm_w%dms_PC%d.fig',...
%         cond,downsampf,ncomp,twind*1000,cc));
%     saveas(gcf,sprintf('/data/liuzzil2/UMD_Flanker/results/Eigbar_%s_f%dHz_%dcomps_downsamp_norm_w%dms_PC%d.jpg',...
%         cond,downsampf,ncomp,twind*1000,cc));
 
end


