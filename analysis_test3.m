clearvars % clear matlab workspace
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

for stn = 1:3
    if stn == 1
        stimname =  'resp';
    elseif stn == 2
        stimname =  'flan';
    else
        stimname =  'cue';
    end

    normalizeopt = 2; % normalized overall variance of residuals
    ncomp = 2; % number of ICA components to keep

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

    downsampf = 30; % leave empty if no downsampling

    twind = 0.1; % test different window lenghts? (Shorter or longer attractors?)
    % tstep = round( dowsampf * twind /2) ;
    tstep = round( downsampf * 0.04) ;
    xstep = 1; %t+xstep sample to predict


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

    %% 
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


            if exist( [outputfolder,filename,'.set'],'file') %&& ~exist(fileout,'file')


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

                    % rejtrials = rejtrials(1+length(rejtrials)-size(eventTable,1):end);

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

                    Icorrect(strcmp(eventclean.RPTP , 'Omission')) = [];
                    Ccorrect(strcmp(eventclean.RPTP , 'Omission')) = [];
                    Commission(strcmp(eventclean.RPTP , 'Omission')) = [];

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

                    condmat = Respall;
                    condname = 'All';

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
                    %% Example plots, begin comment
%                     erptemp = reshape(permute(erptemp,[1,3,2]), [ncomp,size(erptemp,2)*size(erptemp,3)]);  % norm2
% 
%                     figure; set(gcf,'color','w'); plot(erptemp');
%                     xlabel('data point'); ylabel('z-score');
%                     legend('PC 1','PC 2','location','best')
%                     saveas(gcf,'/data/liuzzil2/UMD_Flanker/results/analysis_description_1.fig')
%                     figure; set(gcf,'color','w')
%                     histogram(erptemp(1,:,:),'Normalization','pdf','DisplayStyle','bar')
%                     hold on
%                     histogram(erptemp(2,:,:),'Normalization','pdf','DisplayStyle','bar')
%                     xlim([-8 8]);legend('PC 1','PC 2','location','best')
%                     xlabel('z-score'); ylabel('probability distribution');
%                     saveas(gcf,'/data/liuzzil2/UMD_Flanker/results/analysis_description_2.fig')
%                     v = var(erppca,0,2);
%                     figure; plot(erp.time, squeeze(v))
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
% 
%                     rtIcorrect = cell2mat(eventclean.rtim(condmat));
%                     srt = sort(rtIcorrect);
%                     figure; plot(srt)
%                     ylabel('reaction time'); title('Correct Incongruent')
%                     tend = srt(round(0.8 * length(srt)))/1000;
% 
% 
%                     figure; set(gcf,'color','w','position',[  1129         325         360         888])
                    %% end comment
                    datapc =  mean(erppca(:,Ccorrect,:),2); % congrunet correct
                    xcc = erppca(:,Ccorrect,:) - datapc;

                    datapc =  mean(erppca(:,Icorrect,:),2); % congrunet correct
                    xic = erppca(:,Icorrect,:) - datapc;

                    datapc =  mean(erppca(:,Commission,:),2); % congrunet correct
                    xco = erppca(:,Commission,:) - datapc;

                    for cond =1:4

                        if cond == 1

                            condname = 'Commission';
                            datapc =  mean(erppca(:,Commission,:),2);
                            x = xco;
                        elseif cond == 2

                            condname = 'CongCorr';
                            datapc =  mean(erppca(:,Ccorrect,:),2);
                            x = xcc;
                        elseif cond == 3

                            condname = 'IncongCorr';
                            datapc =  mean(erppca(:,Icorrect,:),2);
                            x= xic;
                        elseif cond == 4

                            condname = 'All';
                            datapc =  mean(erppca,2);
                            x = cat(2, xcc, xic, xco);
                        end
                        %% Example plots, begin comment
%                         figure; set(gcf,'color','w')
%                         plot(erp.time, squeeze(mean(x,2)), 'linewidth',2)
%                         xlabel('time(s)');
%                         title('Mean residual $\bar{x}$','interpreter','latex','Fontsize',11)
%                         xlim([tstart, tend]); ylim([-0.001, 0.001])
%                         saveas(gcf,'/data/liuzzil2/UMD_Flanker/results/analysis_description_3.fig')
%                         %                         Example trial, jj 6, age 18
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
%                         xlabel('time (s)'); ylabel('\muV'); grid on
%                         xlim([tstart, tend]); ylim([-2, 2])
%                         subplot(313)
%                         plot(erp.time,squeeze(x(1,1,:))','linewidth',2)
%                         hold on
%                         plot(erp.time,squeeze(x(2,1,:))','linewidth',2)
%                         title('Single trial residual $x_i = s_i - \bar{s}$','interpreter','latex','Fontsize',11,'fontweight','bold')
%                         xlabel('time (s)'); ylabel('\muV'); grid on
%                         xlim([tstart, tend]); ylim([-2, 2])
%                         saveas(gcf,'/data/liuzzil2/UMD_Flanker/results/analysis_description_4.fig')
% 
%                         figure; set(gcf,'color','w')
%                         for ii = 1:20
%                             plot(erp.time,squeeze(x(:,ii,:))' + (ii-1)*4,'k')
%                             hold on; xlim([tstart, tend]);
%                         end
%                         ylim([-3 80]); set(gca,'yTick',[])
%                         xlabel('time (s)');
% 
%                         %                         Fourier spectrum of residuals
%                         nffts = 2.^(7:11) ;
%                         nfft = nffts( nffts < size(x,3)/2);
%                         if isempty(nfft)
%                             nfft = 128;
%                         end
%                         pp = zeros(ncomp,nfft(end)+1,size(x,2));
%                         for cc = 1:ncomp
%                             [pp(cc,:,:),ff] = pwelch(squeeze(x(cc,:,:))',[],[],[],downsampf);
%                         end
%                         saveas(gcf,'/data/liuzzil2/UMD_Flanker/results/analysis_description_5.fig')
                        %% end comment

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


