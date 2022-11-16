clear % clear matlab workspace
clc % clear matlab command window
%addpath(genpath('C:\Users\Berger\Documents\eeglab13_4_4b'));% enter the path of the EEGLAB folder in this line
%addpath(genpath('C:\Users\Berger\Documents\eeglab13_4_4b'))
nih_matmod load eeglab
eeglab % call eeglab to set up the plugin
addpath(genpath('/data/liuzzil2/UMD_Flanker/matlab/'));

addpath /home/liuzzil2/fieldtrip-20190812/
ft_defaults

datapath = '/data/liuzzil2/UMD_Flanker/derivatives/';
cd(datapath)
subsdir = dir;
subsdir(1:2) = [];
subslist = cell(length(subsdir ) ,1);

load('/data/liuzzil2/UMD_Flanker/results/correctCI_commission_allstims.mat')

for stn = 1:2
    if stn == 1
        stimname =  'resp';
    elseif stn == 2
        stimname =  'flan';
    else 
        stimname =  'cue';
    end
 
     % sub = 6257, age 18, only 25 trials, jj = 98
    normalizeopt = 2; % normalized overall variance of residuals
    ncomp = 2; % try 3 components

    if strcmp(stimname,'flan')
%         load('/data/liuzzil2/UMD_Flanker/results/pcaSubConcat_correct_congruent_n335.mat')
        tend = 0.6;
        tstart = -0.4;
    elseif strcmp(stimname,'resp')
%         load('/data/liuzzil2/UMD_Flanker/results/pcaSubConcat_resp_correct_commiss_n317.mat')
        tend = 0.5;
        tstart = -0.5;
    else
%         load('/data/liuzzil2/UMD_Flanker/results/pcaSubConcat_correct_congruent_n335.mat')
       
        tend = 0.6;
        tstart = -0.6;   
    end
     tbse = [tstart, tend]; % 0 mean in the same window of analysis
    % coeff = coeff(:,2);
    
for downsampf = [40] % leave empty if no downsampling
    
    twind = 0.1; % test different window lenghts? (Shorter or longer attractors?)
    % tstep = round( dowsampf * twind /2) ;
    tstep = round( downsampf * 0.04) ;
    xstep = 1; %round( downsampf * 0.025) ;
   

    % EEG = [];
    % sub = '6015'; agegroup = '12';
    % filename = ['sub-',sub,'_task-flanker_eeg_processed_data'];
    % outputfolder = sprintf('/data/liuzzil2/UMD_Flanker/derivatives/sub-%s/age-%s/',sub,agegroup);
    % EEG = pop_loadset([filename,'.set'] , outputfolder);
    % data = eeglab2fieldtrip(EEG,'raw');
    %
    % changroup{1} = [21,22,14:19,9:12,4,5];% FZ
    % changroup{2} = [23:28, 20,32:34,38];% FL
    % changroup{3} = [1:3, 8, 121:124, 116:118];% FR
    % changroup{4} = [6:7, 13,30:31,37,54,55,105,106,112,79,80,87];% CZ
    % changroup{5} = [44:47, 39:42,35,36,29];% CL
    % changroup{6} = [108:111, 114,115,102:104,93,98];% CR
    % changroup{7} = [61,62,67,70:72,74:78,82:83];% OZ
    % changroup{8} = [50:53, 57:60, 64:66, 69];% OL
    % changroup{9} = [84:86, 89:92,95:97,100:101];% OR
    %
    % coeff = zeros(length(data.label),ncomp);
    %
    % for n = 1:length(data.label)
    %     for cc = 1:ncomp
    %         coeff(n,cc) = any(str2double(data.label{n}(2:end)) == changroup{cc});
    %     end
    % end
    
    %% sub > 6350, jj = 121:end
    for jj = 1:length(subsdir)
        sub = subsdir(jj).name(5:end);
        
        for age = [12,15,18] % redo age 15, 18
            agegroup = num2str(age);
            
            subdir = ['sub-',sub];
            
            filebids = [subdir,'_task-flanker_eeg'];
            
            outputfolder = sprintf('/data/liuzzil2/UMD_Flanker/derivatives/sub-%s/age-%s/',sub,agegroup);
            bidsfolder = sprintf('/data/liuzzil2/UMD_Flanker/bids/sub-%s/age-%s/eeg/',sub,agegroup);
            if strcmp(stimname, 'flan')
                filename = [filebids '_processed_data'];
                rejtrialsf = [filebids '_rejected_trials.mat'];
            else
                filename = [filebids '_processed_data_',stimname];
                rejtrialsf = [filebids '_rejected_trials_',stimname,'.mat'];
            end
            filenamebids = [bidsfolder,filebids ];
            
            fileout = sprintf('%sAtest3_%s_%dcomps_f%dHzlowp_norm%d_w%dms_step%d_xstep%d.mat',...
                        outputfolder,stimname,ncomp,downsampf,normalizeopt,twind*1000,tstep,xstep);
                    
%             unix(sprintf('rm %sAtest3_*_f100Hz*',outputfolder));  
%             unix(sprintf('rm %sAtest3_*_f80Hz*',outputfolder));  
%             unix(sprintf('rm %sAtest3_*_f25Hz*',outputfolder));     
%             unix(sprintf('rm %sAtest3_*_f35Hz*',outputfolder));     
%             unix(sprintf('rm %sAtest3_*_f45Hz*',outputfolder));     
%             unix(sprintf('rm %sAtest3_*_f50Hzlowp_norm1_w500ms*',outputfolder));     
%             unix(sprintf('rm %sAtest3_*_f50Hz_norm1*',outputfolder));

            
            if exist( [outputfolder,filename,'.set'],'file') && ~exist(fileout,'file')
                
                
                EEG = [];
                EEG = pop_loadset([filename,'.set'] , outputfolder);
                load([outputfolder,rejtrialsf]) %,'rejtrials'
                data = eeglab2fieldtrip(EEG,'raw');
                
                if length(data.trial) > 50 % only use datasets with at least 50 trials
                    
                    %                 tempTable = array2table({sub,age,length(data.trial)}, 'VariableNames',["sub","age","ntrials"]);
                    %                 if k == 0
                    %                     subTable = tempTable;
                    %                 else
                    %                     subTable = cat(1,subTable,tempTable);
                    %                 end
                    %                 k = k +1;
                    %% STEP 1: Import data file and events information
                    headerfile = [filenamebids, '.set'];
                    datafile = [filenamebids, '.fdt'];
                    
                    hdr = pop_loadset([filebids,'.set']   ,  bidsfolder);
                    events = hdr.event;
                    
                    ntrials = 400;
                    ntrsp = 0;
                    tablenames =  ["cel#","obs#","rsp#","eval","rtim","trl#","CoNo","SITI",...
                        "BkNb","TotN","TRLN","TRTP","Feed","RPTP"];
             
                    eventnames = {'resp','FLAN','Cue+'};
                    t = 1000;
                    for ii = 1:length(events)
                        
                        if strcmp(events(ii).type,eventnames{stn}) && length(events(ii).codes) == 4
                            t = events(ii).init_time;
                        elseif strcmp(events(ii).type,'TRSP') && length(events(ii).codes) >= 14 && ...
                                events(ii).init_time > t && events(ii).init_time < (t+3)
                            if ntrsp == 0
                                % Initialize table
                                eventTable = array2table(events(ii).codes(1:14,2)', 'VariableNames',tablenames);
                                ntrsp = 1;
                            else
                                tempTable = array2table(events(ii).codes(1:14,2)', 'VariableNames',tablenames);
                                eventTable = cat(1,eventTable,tempTable);
                            end
                        end
                        
                    end
                    
%                     rejtrials = rejtrials(1+length(rejtrials)-size(eventTable,1):end);

                    eventclean = eventTable(~rejtrials,:);
                    
                  
                    
                    %%
                    %                 close all
                    
                    if ~isempty(downsampf )
                        cfg = [];
                        cfg.demean          = 'yes';
                        cfg.baselinewindow  = tbse;
                        cfg.lpfilter = 'yes';
                        cfg.lpfreq = downsampf/2;
                        cfg.trials = cell2mat(eventclean.BkNb) > 1;% eliminate practice trials
                        datares = ft_preprocessing(cfg,data);
                        
                        cfg = [];
                        cfg.resamplefs = downsampf;
                        datares = ft_resampledata(cfg,datares);
                    else
                        cfg = [];
                        cfg.trials = cell2mat(eventclean.BkNb) > 1;% eliminate practice trials
                        datares = ft_preprocessing(cfg,data);
                    end
                    eventclean = eventclean(cell2mat(eventclean.BkNb) > 1,:);  
                      %% 4 conditions: correclty pressed left, correctly pressed right
                    

                    Correct =  strcmp(eventclean.RPTP , 'Correct');
                    Commission =  strcmp(eventclean.RPTP , 'Commission');
                    Icorrect =  strcmp(eventclean.RPTP , 'Correct') & strcmp(eventclean.CoNo , 'Incongruent');
                    Ccorrect =  strcmp(eventclean.RPTP , 'Correct') & strcmp(eventclean.CoNo , 'Congruent');
                    Icommis = strcmp(eventclean.RPTP , 'Commission') & strcmp(eventclean.CoNo , 'Incongruent');
                    Congr = ~strcmp(eventclean.RPTP , 'Omission') & strcmp(eventclean.CoNo , 'Congruent');
                    Incong = ~strcmp(eventclean.RPTP , 'Omission') & strcmp(eventclean.CoNo , 'Incongruent');
                    Respall = ~strcmp(eventclean.RPTP , 'Omission') ;
                    %                 % Topoplots
                    %                 cfg = [];
                    %                 cfg.trials = Icorrect;
                    %                 cfg.keeptrials         = 'no';
                    %                 erpIcorrect = ft_timelockanalysis(cfg, data);
                    %
                    %                 cfg = [];
                    %                 cfg.trials = Icorrect;
                    %                 cfg.keeptrials         = 'no';
                    %                 erpCcorrect = ft_timelockanalysis(cfg, data);
                    %
                    %                 cfg = [];
                    %                 cfg.layout = 'GSN-HydroCel-128.sfp';
                    %                 cfg.interactive = 'no';
                    %                 cfg.showoutline = 'yes';
                    %                 figure
                    %                 ft_multiplotER(cfg, erpCcorrect, erpIcorrect)
                    
                    %%
                    
                    Aall = [];
                    for cond = 1:7 %1:7
                        if cond == 1
                            condmat = Correct;
                            condname = 'Correct';
                        elseif cond == 2
                            condmat = Commission;
                            condname = 'Commission';
                        elseif cond == 3
                            condmat = Ccorrect;
                            condname = 'CongCorr';
                        elseif cond == 4
                            condmat = Icorrect;
                            condname = 'IncongCorr';
                        elseif cond == 5
                            condmat = Respall;
                            condname = 'All';
                        elseif cond == 6
                            condmat = Congr;
                            condname = 'Congruent';
                        elseif cond == 7
                            condmat = Incong;
                            condname = 'Incongruent';
                        end
                        

                        cfg = [];
                        cfg.trials = condmat;
                        cfg.keeptrials         = 'yes';
                        erp = ft_timelockanalysis(cfg, datares);
                        
                        [~,t1] =  min( abs(erp.time - tstart));
                        [~,te] =  min( abs(erp.time - (tend - twind)));
                        %                     t1 = t1 -1 ;
                        %                     te = te + 1;
                        if te >= length(erp.time)
                            te =  length(erp.time) - 1;
                        end
                        twindsamp = ceil(twind*datares.fsample);
                        
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
                            erppca =( erppca - mean(erptemp,3));  % norm2
                            erptemp = erppca(:,:,erp.time>= tstart & erp.time <= tend);
                            erptemp = reshape(permute(erptemp,[1,3,2]), [ncomp,size(erptemp,2)*size(erptemp,3)]);  % norm2
                            erppca =  erppca  ./ std(erptemp,0,2);   % norm2
                        end
                        erptemp = erppca(:,:,erp.time>= tstart & erp.time <= tend);
                        [N,edges] =histcounts(erptemp(:),'Normalization','pdf');
                        edges = (edges(1:end-1) + edges(2:end))/2;
                        
                        erptemp = reshape(permute(erptemp,[1,3,2]), [ncomp,size(erptemp,2)*size(erptemp,3)]);  % norm2
%                         figure; set(gcf,'color','w'); plot(erptemp'); 
%                         xlabel('data point'); ylabel('z-score'); 
%                         legend('PC 1','PC 2','location','best')
%                         figure; set(gcf,'color','w')
%                         histogram(erptemp(1,:,:),'Normalization','pdf','DisplayStyle','bar')
%                         hold on
%                         histogram(erptemp(2,:,:),'Normalization','pdf','DisplayStyle','bar')
%                         xlim([-8 8]);legend('PC 1','PC 2','location','best')
%                         xlabel('z-score'); ylabel('probability distribution'); 
%                         v = var(erppca,0,2);
%                         figure; plot(erp.time, squeeze(v))
                        %                     figure;
                        %                     for cc = 1:ncomp
                        %                         subplot(3,3,cc)
                        %                         co = get(gca,'colororder');
                        %                         x = squeeze(erppca(cc,:,:));
                        %                         plot(erp.time, mean(x,1))
                        %                         hold on
                        %                         fill([erp.time, fliplr(erp.time)], ...
                        %                             [mean(x,1)+std(x,0,1) , fliplr(mean(x,1)-std(x,0,1) )],...
                        %                             co(1,:),'edgecolor','none','facealpha',0.2)
                        %                     end
                        
                        %                     rtIcorrect = cell2mat(eventclean.rtim(condmat));
                        %                     srt = sort(rtIcorrect);
                        %                 figure; plot(srt)
                        %                 ylabel('reaction time'); title('Correct Incongruent')
                        %                     tend = srt(round(0.8 * length(srt)))/1000;
                        
                        
                        %                 figure; set(gcf,'color','w','position',[  1129         325         360         888])
                        
                        
                        datapc =  mean(erppca,2);
                        x = erppca - datapc;
                        
%                         figure; set(gcf,'color','w')
%                         plot(erp.time, squeeze(mean(x,2)), 'linewidth',2)
%                         xlabel('time(s)'); 
%                         title('Mean residual $\bar{x}$','interpreter','latex','Fontsize',11)
%                         xlim([tstart, tend]); ylim([-0.001, 0.001])
                        
                        % Example trial, jj 6, age 18
%                         figure; set(gcf,'color','w'); subplot(311)
%                         plot(erp.time,squeeze(datapc)','linewidth',2)
%                         xlabel('time (s)'); ylabel('\muV'); grid on
%                         xlim([tstart, tend]); ylim([-2, 2])
%                         legend('PC 1','PC 2','location','best'); 
%                         title('Average ERP $\bar{s}$','interpreter','latex','Fontsize',11,'fontweight','bold')
%                         subplot(312)
%                         plot(erp.time,squeeze(erppca(1,1,:))','linewidth',2)
%                         hold on
%                         plot(erp.time,squeeze(erppca(2,1,:))','linewidth',2)
%                         title('Single trial ERP $s_i$','interpreter','latex','Fontsize',11,'fontweight','bold')
%                        
%                         xlabel('time (s)'); ylabel('\muV'); grid on
%                         xlim([tstart, tend]); ylim([-2, 2])
%                         subplot(313)
%                         plot(erp.time,squeeze(x(1,1,:))','linewidth',2)
%                         hold on
%                         plot(erp.time,squeeze(x(2,1,:))','linewidth',2)
%                         title('Single trial residual $x_i = s_i - \bar{s}$','interpreter','latex','Fontsize',11,'fontweight','bold')
%                         xlabel('time (s)'); ylabel('\muV'); grid on
%                         xlim([tstart, tend]); ylim([-2, 2])
%                         
%                         figure; set(gcf,'color','w')
%                     for ii = 1:20
%                         plot(erp.time,squeeze(x(:,ii,:))' + (ii-1)*4,'k')
%                         hold on; xlim([tstart, tend]); 
%                     end
%                     ylim([-3 80]); set(gca,'yTick',[])
%                         xlabel('time (s)');

                        % Fourier spectrum of residuals
%                         nffts = 2.^(7:11) ;
%                         nfft = nffts( nffts < size(x,3)/2);
%                         if isempty(nfft)
%                             nfft = 128;
%                         end
%                         pp = zeros(ncomp,nfft(end)+1,size(x,2));
%                         for cc = 1:ncomp
%                             [pp(cc,:,:),ff] = pwelch(squeeze(x(cc,:,:))',[],[],[],downsampf);
%                         end
                        
                        
                        A = zeros(ncomp,ncomp,length(tb));
                        pvalue = zeros(ncomp,length(tb));
                        F = zeros(ncomp,length(tb));
                        %                     Pc = mean(datapc(:,t1:te),1);
                        
                        n = size(x,2)*twind*datares.fsample;%numel(Pt); % number of observations
                        p = ncomp;%numel(a); % number of regression parameters
                        DFE = n - p;% degrees of freedom for error
                        DFM = p -1; % corrected degrees of freedom form model
                        %                     xx = 0:0.01:1000;
                        %                     yy = fpdf(xx,DFM,DFE);
                        
                        
                        eigv = zeros(ncomp, length(tb));
                        rmse = zeros(ncomp, length(tb));
                        
                        for t = 1:length(tb)
                            Pt = x(:,:,tb(t) + (0:(twindsamp-1)));
                            % make xstep a vector?? use instead of components?
                            Pt1 = x(:,:,tb(t) + xstep + (0:twindsamp-1)) ;
                            
                             % Example trial, jj 6, age 18
%                             figure; set(gcf,'color','w');
%                             scatter(squeeze(Pt(1,100,:)),squeeze(Pt1(1,100,:)),'filled')
%                             hold on
%                             scatter(squeeze(Pt(2,100,:)),squeeze(Pt1(2,100,:)),'filled')
%                             xlabel('x(t)');  ylabel('x(t+1)')
%                             axis equal

                            
                            Pt = reshape(Pt,[ncomp,size(x,2)*twindsamp ])';
                            Pt1 = reshape(Pt1,[ncomp,size(x,2)*twindsamp])';
                            
% %                             % Example trial, jj 6, age 18
%                              figure; set(gcf,'color','w');
%                             scatter((Pt(:,1)),(Pt1(:,1)),5,'filled')
%                             hold on
%                             scatter((Pt(:,2)),(Pt1(:,2)),5,'filled')
%                             xlabel('x(t)');  ylabel('x(t+1)')
%                             axis equal; grid on
%                             xlim([-4 4]);ylim([-4 4])
                            
                            a = Pt\Pt1;  % inv(Pt) * Pt1
                            A(:,:,t) = a;
                            eigv(:,t) = eig(a);
                            
                            %                         [V,D] = eig(a); % A*V = V*D.
                            yhat = Pt*a;
                            
                             % Example trial, jj 6, age 18
%                              hold on; set(gcf,'color','w');
%                             scatter((Pt(:,1)),(yhat(:,1)),'filled')
%                             hold on
%                             scatter((Pt(:,2)),(yhat(:,2)),'filled')
%                             xlabel('x(t)');  ylabel('x(t+1)')
%                             axis equal
    
                            SSM = sum( (yhat - mean(Pt1,1)).^2); % corrected sum of squares
                            SSE = sum( (yhat - Pt1).^2 ); % sum of squares of residuals
                            rmse(:,t) = sqrt(  mean( (yhat - Pt1).^2 ));
                            MSM = SSM ./ DFM; % mean of squares for model
                            MSE = SSE ./ DFE;% mean of squares or error
                            F(:,t) = MSM ./ MSE;  %(explained variance) / (unexplained variance)
                            
                            %                         [~,indf] = min(abs(F(:,k) - xx),[],2);
                            %                         for ik = 1:ncomp
                            %                             pvalue(ik,k) = trapz (xx(indf(ik):end), yy(indf(ik):end));
                            %                         end
                            
                        end
                        %                     figure; plot(time, abs(eigv))
                        %                     title(['Eigenvalues, xstep = ',num2str(1000*xstep/downsampf),'ms'])
                        
                        
                        %                     for tt = 1:length(erp.sampleinfo)
                        %                         figure(1); clf
                        %
                        %                         plot(erp.time(1:end-1), squeeze(x(:,tt,2:end)),'linewidth',2); hold on
                        %                         co = get(gca, 'colororder');
                        %                         plot(erp.time(t1:te), squeeze(Pta(1,tt,:)),'--','color',co(1,:), 'linewidth',2)
                        %                         plot(erp.time(t1:te), squeeze(Pta(2,tt,:)),'--','color',co(2,:), 'linewidth',2)
                        %                         plot(erp.time(t1:te), squeeze(Pta(3,tt,:)),'--','color',co(3,:), 'linewidth',2)
                        %                         xlabel('time'); xlim([tstart, tend])
                        %                         % root mean square error of the fit, average over all time points
                        %                         title(sprintf('%s: trial %d. RMSE over trials = %.2f',condname,tt,mean(rmse)))
                        %                         pause(1)
                        %                     end
                        
                        
                        %                     figure(1); clf
                        %                     subplot(411)
                        %                     plot(erp.time, squeeze(mean(erppca,2)),'linewidth',2);
                        %                     xlim([tstart, tend]); hold on
                        %                     title('PCA components')
                        %                     subplot(412)
                        %                     plot(erp.time, squeeze( mean(abs(x),2) ) ,'linewidth',2);
                        %                     xlim([tstart, tend]);
                        %                     title('Average amplitude of residual EEG signal')
                        %                     subplot(413)
                        %                     plot(erp.time(t1:te), abs(eigv),'linewidth',2)
                        %                     xlabel('time'); xlim([tstart, tend])
                        %                     title('Absolute value of A eigenvalues'); %ylim([0 1])
                        %                     subplot(414)
                        %                     plot(erp.time(t1:te), rmse,'linewidth',2)
                        %                     xlabel('time');  xlim([tstart, tend])
                        %                     title('RMSE of x(t+1) = A*x(t)')
                        %
                        [~,tt,~] = intersect(erp.time, time);
                        Aall.(condname).A = A;
                        Aall.(condname).eig = eigv;
                        Aall.(condname).rmse = rmse;
                        Aall.(condname).time = time;
                        Aall.(condname).erpm = squeeze(datapc(:,:,tt));
                        Aall.(condname).residualm = squeeze(mean(x(:,:,tt),2)); % mean over trials 
                        Aall.(condname).residuals = squeeze(std(x(:,:,tt),[],2)); % std over trials
                        Aall.(condname).erphist = [edges;N]; 
%                         Aall.(condname).ff = ff;
%                         Aall.(condname).Presidual = mean(pp,3);
%                         Aall.(condname).pvalue = pvalue;
                        Aall.(condname).ntrials = size(x,2);
                        Aall.(condname).F = F;
                        Aall.(condname).DFM = DFM;
                        Aall.(condname).DFE = DFE;
                    end
                    %                 figure; plot(erp.time,A)
                    %                 xlim([tstart, tend])
                    %                 legend('PC 1','PC 2','PC 3','location','best')
                    %                 xlabel('time'); title([condname,' regression coeff']); ylabel('A')
                    save(fileout,'Aall')
                    
                    
                    
                end
                %%
                
            end
        end
        clc
        fprintf('Done subjects %d/%d\n', jj, length(subsdir))
    end
    
end
end
%%
% close all
% cond = 'IncongCorr';
% cc = 2;
% 
% A = [];
% A.age12 = [];
% A.age15 = [];
% A.age18 = [];
% xx = 0:0.01:4000;
% for jj = 1:length(subsdir)
%     sub = subsdir(jj).name(5:end);
%     ages = [12,15,18];
%     for ii = 1:3
%         age = ages(ii);
%         agegroup = num2str(age);
%         
%         subdir = ['sub-',sub];
%         
%         filebids = [subdir,'_task-flanker_eeg'];
%         
%         outputfolder = sprintf('/data/liuzzil2/UMD_Flanker/derivatives/sub-%s/age-%s/',sub,agegroup);
%         if strcmp(stimname, 'resp') || strcmp(stimname, 'flan')
%             filename = sprintf('%sAtest3_%s_%dcomps_f%dHz_norm_w%dms_step%d_xstep%d.mat',...
%                 outputfolder,stimname,ncomp,downsampf,twind*1000,tstep,xstep);
%         else
%             filename = sprintf('%sA_test3_pcaSubConcat_%dcomps_f%dHz_norm_w%dms_step%d_xstep%d.mat',...
%                 outputfolder,ncomp,downsampf,twind*1000,tstep,xstep);
%         end
%         
%         if exist( filename,'file')
%             load(filename)
%             
%             %                 yy = fpdf(xx,Aall.(cond).DFM,Aall.(cond).DFE);
%             %                 pvalue = zeros(size(Aall.(cond).F));
%             %                 for k = 1:size(Aall.(cond).F,2)
%             %
%             %                     for ik = 1:size(Aall.(cond).F,1)
%             %                         [~,indf] = min(abs(Aall.(cond).F(ik,k) - xx),[],2);
%             %                         pvalue(ik,k) = trapz(xx(indf:end), yy(indf:end));
%             %                     end
%             %                 end
%             if size(Aall.(cond).EEGresidual,2) == 1
%                 Aall.(cond).EEGresidual = Aall.(cond).EEGresidual';
%             end
%             if isempty(A.(['age',agegroup]))
%                 A.(['age',agegroup]).eig =  sum( abs(Aall.(cond).eig(cc,:)) ,1 );
%                 A.(['age',agegroup]).rmse = mean( Aall.(cond).rmse(cc,:) ,1 ) ;
%                 %                 A.(['age',agegroup]).EEGresidual = mean(mean(abs(Aall.(cond).EEGresidual(cc,:,:)),1),2);
%                 A.(['age',agegroup]).EEGresidual = mean(Aall.(cond).EEGresidual(cc,:),1);
%                 %                 A.(['age',agegroup]).pvalue = Aall.(cond).pvalue;
%                 A.(['age',agegroup]).ntrials = Aall.(cond).ntrials;
%                 
%             else
%                 A.(['age',agegroup]).eig = cat(1,A.(['age',agegroup]).eig, ...
%                     sum( abs(Aall.(cond).eig(cc,:)) ,1 ));
%                 A.(['age',agegroup]).rmse = cat(1,A.(['age',agegroup]).rmse,...
%                     mean( Aall.(cond).rmse(cc,:) ,1 ));
%                 %                 A.(['age',agegroup]).EEGresidual = cat(1,A.(['age',agegroup]).EEGresidual,...
%                 %                     mean(mean(abs(Aall.(cond).EEGresidual(cc,:,:)),1),2));
%                 A.(['age',agegroup]).EEGresidual = cat(1,A.(['age',agegroup]).EEGresidual,...
%                     mean(Aall.(cond).EEGresidual(cc,:),1));
%                 %                 A.(['age',agegroup]).pvalue = cat(1,A.(['age',agegroup]).pvalue, ...
%                 %                     Aall.(cond).pvalue);
%                 A.(['age',agegroup]).ntrials = cat(1,A.(['age',agegroup]).ntrials, ...
%                     Aall.(cond).ntrials);
%                 
%             end
%             
%             if isfield(Aall.Correct,'Presidual')
%                 if ~isfield(A.(['age',agegroup]),'Presidual')
%                     A.(['age',agegroup]).Presidual = mean(Aall.(cond).Presidual(cc,:),1);
%                 else
%                     A.(['age',agegroup]).Presidual = cat(1,A.(['age',agegroup]).Presidual,...
%                         mean(Aall.(cond).Presidual(cc,:),1));
%                 end
%             end
%             
%             
%         end
%     end
% end
% 
% %     figure
% %     plot(Aall.CongCorr.ff, mean(A.(['age',agegroup]).Presidual,1))
% 
% 
% residual = [A.age12.EEGresidual;A.age15.EEGresidual;A.age18.EEGresidual];
% %     [iis,iit] = find(A.age12.EEGresidual >  (mean(residual(:)) + 20*std(residual(:)) ));
% n12 = 1:size(A.age12.eig,1); %n12(iis) = [];
% %     [iis,iit] = find(A.age15.EEGresidual >  (mean(residual(:)) + 20*std(residual(:)) ));
% n15 = 1:size(A.age15.eig,1); %n15(iis) = [];
% %     [iis,iit] = find(A.age18.EEGresidual >  (mean(residual(:)) + 20*std(residual(:)) ));
% n18 = 1:size(A.age18.eig,1);  %n18(iis) = [];
% k = size(A.age12.eig,1);
% 
% A.age12.eig = A.age12.eig(n12,:);
% A.age12.rmse = A.age12.rmse(n12,:);
% A.age12.EEGresidual = squeeze(A.age12.EEGresidual(n12,:,:));
% A.age15.eig = A.age15.eig(n15,:);
% A.age15.rmse = A.age15.rmse(n15,:);
% A.age15.EEGresidual = squeeze(A.age15.EEGresidual(n15,:,:));
% A.age18.eig = A.age18.eig(n18,:);
% A.age18.rmse = A.age18.rmse(n18,:);
% A.age18.EEGresidual = squeeze(A.age18.EEGresidual(n18,:,:));
% 
% 
% % figure(2); clf; set(gcf,'color','w')
% % bar(1:3,[mean(A.age12.ntrials), mean(A.age15.ntrials),  mean(A.age18.ntrials)])
% % hold on
% % errorbar(1:3,[mean(A.age12.ntrials), mean(A.age15.ntrials),  mean(A.age18.ntrials)],...
% %     [std(A.age12.ntrials), std(A.age15.ntrials),  std(A.age18.ntrials)],'k.')
% % set(gca,'XtickLabels',{'age 12','age 15','age 18'})
% % title([cond, ' number of trials'])
% % ylabel('no.trials')
% % saveas(gcf,['/data/liuzzil2/UMD_Flanker/results/A_test3_pcaSubConcat6_',cond,'_trials.jpg'])
% 
% 
% time = Aall.CongCorr.time;
% if strcmp(stimname,'flan')
%     timefull = linspace(-0.5,1,downsampf*1.5);
% elseif  strcmp(stimname,'resp')
%     timefull = linspace(-1,1,downsampf*2);
% end
% figure; clf; set(gcf,'color','w')
% 
% subplot(311)
% plot(timefull, cat(1, mean(A.age12.EEGresidual,1),...
%     mean(A.age15.EEGresidual,1),mean(A.age18.EEGresidual,1)) ,'linewidth',2);
% co = get(gca,'colororder'); hold on;
% fill([timefull fliplr(timefull)],...
%     [mean(A.age12.EEGresidual,1)+std(A.age12.EEGresidual)/sqrt(size(A.age12.eig,1)),...
%     fliplr(mean(A.age12.EEGresidual,1)-std(A.age12.EEGresidual)/sqrt(size(A.age12.eig,1)))],...
%     co(1,:),'edgecolor','none','facealpha',0.3)
% fill([timefull fliplr(timefull)],...
%     [mean(A.age15.EEGresidual,1)+std(A.age15.EEGresidual)/sqrt(size(A.age15.eig,1)),...
%     fliplr(mean(A.age15.EEGresidual,1)-std(A.age15.EEGresidual)/sqrt(size(A.age15.eig,1)))],...
%     co(2,:),'edgecolor','none','facealpha',0.3)
% fill([timefull fliplr(timefull)],...
%     [mean(A.age18.EEGresidual,1)+std(A.age18.EEGresidual)/sqrt(size(A.age18.eig,1)),...
%     fliplr(mean(A.age18.EEGresidual,1)-std(A.age18.EEGresidual)/sqrt(size(A.age18.eig,1)))],...
%     co(3,:),'edgecolor','none','facealpha',0.3)
% %     xlim([tstart, tend+0.1]);
% title('Average amplitude of residual EEG signal')
% 
% subplot(312)
% plot(time, cat(1,mean(A.age12.eig,1),...
%     mean(A.age15.eig,1),mean(A.age18.eig,1 )),'linewidth',2)
% hold on;
% fill([time fliplr(time)],...
%     [mean(A.age12.eig,1)+std(A.age12.eig)/sqrt(size(A.age12.eig,1)),...
%     fliplr(mean(A.age12.eig,1)-std(A.age12.eig)/sqrt(size(A.age12.eig,1)))],...
%     co(1,:),'edgecolor','none','facealpha',0.3)
% fill([time fliplr(time)],...
%     [mean(A.age15.eig,1)+std(A.age15.eig)/sqrt(size(A.age15.eig,1)),...
%     fliplr(mean(A.age15.eig,1)-std(A.age15.eig)/sqrt(size(A.age15.eig,1)))],...
%     co(2,:),'edgecolor','none','facealpha',0.3)
% fill([time fliplr(time)],...
%     [mean(A.age18.eig,1)+std(A.age18.eig)/sqrt(size(A.age18.eig,1)),...
%     fliplr(mean(A.age18.eig,1)-std(A.age18.eig)/sqrt(size(A.age18.eig,1)))],...
%     co(3,:),'edgecolor','none','facealpha',0.3)
% xlabel('time');% xlim([tstart, tend])
% title('Absolute value of A eigenvalues'); %ylim([0 1])
% 
% subplot(313)
% plot(time, cat(1,mean(A.age12.rmse,1),...
%     mean(A.age15.rmse,1),mean(A.age18.rmse,1) ),'linewidth',2)
% hold on;
% fill([time fliplr(time)],...
%     [mean(A.age12.rmse,1)+std(A.age12.rmse)/sqrt(size(A.age12.eig,1)),...
%     fliplr(mean(A.age12.rmse,1)-std(A.age12.rmse)/sqrt(size(A.age12.eig,1)))],...
%     co(1,:),'edgecolor','none','facealpha',0.3)
% fill([time fliplr(time)],...
%     [mean(A.age15.rmse,1)+std(A.age15.rmse)/sqrt(size(A.age15.eig,1)),...
%     fliplr(mean(A.age15.rmse,1)-std(A.age15.rmse)/sqrt(size(A.age15.eig,1)))],...
%     co(2,:),'edgecolor','none','facealpha',0.3)
% fill([time fliplr(time)],...
%     [mean(A.age18.rmse,1)+std(A.age18.rmse)/sqrt(size(A.age18.eig,1)),...
%     fliplr(mean(A.age18.rmse,1)-std(A.age18.rmse)/sqrt(size(A.age18.eig,1)))],...
%     co(3,:),'edgecolor','none','facealpha',0.3)
% xlabel('time'); % xlim([tstart, tend])
% title('RMSE of x(t+1) = A*x(t)')
% legend('12yo','15yo','18yo','location','best')
% xl = get(gca,'xlim');
% subplot(311); xlim(xl)
% 
% % saveas(gcf,['/data/liuzzil2/UMD_Flanker/results/A_test3_pcaSubConcat6_',cond,'_norm_PC',num2str(cc),'.jpg'])
% if length(cc) == 1
%     saveas(gcf,sprintf('/data/liuzzil2/UMD_Flanker/results/%s_%s_%dcomps_f%dHz_norm_w%dms_step%d_xstep%d_PCAav_PC%d.jpg',...
%         stimname,cond,ncomp,downsampf,twind*1000,tstep,xstep,cc));
% else
%     saveas(gcf,sprintf('/data/liuzzil2/UMD_Flanker/results/%s_%s_%dcomps_f%dHz_norm_w%dms_step%d_xstep%d.jpg',...
%         stimname,cond,ncomp,downsampf,twind*1000,tstep,xstep));
% end
% 

