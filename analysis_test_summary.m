% clear matlab workspace
clc % clear matlab command window
%addpath(genpath('C:\Users\Berger\Documents\eeglab13_4_4b'));% enter the path of the EEGLAB folder in this line
%addpath(genpath('C:\Users\Berger\Documents\eeglab13_4_4b'))
% nih_matmod load eeglab
% eeglab % call eeglab to set up the plugin
% addpath(genpath('/data/liuzzil2/UMD_Flanker/matlab/'));
%
% addpath /home/liuzzil2/fieldtrip-20190812/
% ft_defaults

addpath /home/liuzzil2/fieldtrip-20190812/
ft_defaults

cd /data/liuzzil2/UMD_Flanker/results

n = 3;
% coeff(:,n) = zeros(size(coeff,1),1);
% coeff(68,n) = 1;
close all
figure(1); set(gcf,'color','w','position',[239 42 1313 762])
figure(2); set(gcf,'color','w','position',[239 42 1313 762])
co  = get(gca,'colororder');

load('correctCI_commission_allstims.mat')
for ii = 1:2
    if ii ==1
        stimname =  'flan';
        
        load('flan_grandavg_erp.mat')
        %          load('pcaSubConcat_correct_congruent_n335.mat')
        %         load('pcaSubConcat_flan_correct_commission_n335.mat')
        if n ==1
            yl = [-60 80];
        else
            yl = [-20 50];
        end
        xl = [-0.5 1];
        t1 = -0.5; t2 = 0;
    else
        stimname =  'resp';
        
        load('resp_grandavg_erp.mat')
        %         load('pcaSubConcat_resp_correct_commiss_n317.mat')
        %         load('pcaSubConcat_resp_correct_commiss_n317.mat')
        %         yl = [-15 35]; xl = [-1 1];
        if n ==1
            yl = [-40 80];
        else
            yl = [-20 50];
        end
        xl = [-1 1];
        t1 = -0.6; t2 = -0.1;
    end
    
    time = grandavg.all.correct.time;
    
    fieldnames = {'all','age12','age15','age18'};
    figure(1); set(gcf,'name','Incong.Corr. - Congruent.Corr.')
    for jj = 1:length(fieldnames)
        subplot(2,4,(ii-1)*4 + jj)
        hold all
        erp1 = sum(grandavg.(fieldnames{jj}).congruentC.avg .* coeff(:,n),1) ;
        %     erp1 = erp1 - mean(erp1(timebl));
        erp1v = sqrt(sum(((grandavg.(fieldnames{jj}).congruentC.var ) .* abs(coeff(:,n))).^2,1))...
            ./ sqrt(grandavg.(fieldnames{jj}).congruentC.dof(1,:));
        plot(time, erp1, 'color',co(1,:),'linewidth',2)
        
        erp2 = sum(grandavg.(fieldnames{jj}).incongruentC.avg .* coeff(:,n),1) ;
        %     erp2 = erp2 - mean(erp2(timebl));
        erp2v = sqrt(sum(((grandavg.(fieldnames{jj}).incongruentC.var ) .* abs(coeff(:,n))).^2,1))...
            ./ sqrt(grandavg.(fieldnames{jj}).incongruentC.dof(1,:));
        plot(time, erp2, 'color',co(2,:),'linewidth',2)
        
        plot(time, erp2 - erp1, '--k')
        
        fill([time, fliplr(time)],[erp1 + erp1v, fliplr(erp1 - erp1v)], co(1,:),...
            'edgecolor','none','facealpha',0.2  )
        
        fill([time, fliplr(time)],[erp2 + erp2v, fliplr(erp2 - erp2v)], co(2,:),...
            'edgecolor','none','facealpha',0.2  )
        
        grid on;  xlabel('time (s)'); ylim(yl); xlim(xl)
        title(sprintf('%s %s, N=%d',fieldnames{jj},stimname,...
            grandavg.(fieldnames{jj}).congruentC.dof(1)))
        if jj ==1
            legend('Cong.Corr.','Incong.Corr.','diff','location','northwest')
        end
        
    end
    
    figure(2); set(gcf,'name','Incong.Comm. - Incong.Corr.')
    for jj = 1:length(fieldnames)
        
        subplot(2,4,(ii-1)*4 + jj)
        hold all
        erp1 = sum(grandavg.(fieldnames{jj}).incongruentC.avg .* coeff(:,n),1) ;
        %     erp1 = erp1 - mean(erp1(timebl));
        erp1v = sqrt(sum(((grandavg.(fieldnames{jj}).incongruentC.var ) .* abs(coeff(:,n))).^2,1)) ...
            ./ sqrt(grandavg.(fieldnames{jj}).incongruentC.dof(1,:));
        plot(time, erp1, 'color',co(1,:),'linewidth',2)
        
        erp2 = sum(grandavg.(fieldnames{jj}).commissionI.avg .* coeff(:,n),1) ;
        %     erp2 = erp2 - mean(erp2(timebl));
        erp2v = sqrt(sum(((grandavg.(fieldnames{jj}).commissionI.var ) .* abs(coeff(:,n))).^2,1)) ...
            ./ sqrt(grandavg.(fieldnames{jj}).commissionI.dof(1,:));
        plot(time, erp2, 'color',co(2,:),'linewidth',2)
        
        plot(time, erp2 - erp1, '--k')
        
        fill([time, fliplr(time)],[erp1 + erp1v, fliplr(erp1 - erp1v)], co(1,:),...
            'edgecolor','none','facealpha',0.2  )
        
        fill([time, fliplr(time)],[erp2 + erp2v, fliplr(erp2 - erp2v)], co(2,:),...
            'edgecolor','none','facealpha',0.2  )
        grid on; xlabel('time (s)'); ylim(yl);   xlim(xl)
        title(sprintf('%s %s, N=%d',fieldnames{jj},stimname,...
            grandavg.(fieldnames{jj}).incongruentC.dof(1) ))
        if jj ==1
            legend('Incong.Corr.','Incong.Comm.','diff','location','northwest')
        end
        
    end
end
figure(1);
saveas(gcf,sprintf('/data/liuzzil2/UMD_Flanker/results/congruency_summaryfig_ages_pc%d.jpg',n))


figure(2);
saveas(gcf,sprintf('/data/liuzzil2/UMD_Flanker/results/correct-commission_summaryfig_ages_pc%d.jpg',n))
%%
figure(3); set(gcf,'color','w')
pcacomp = grandavg.all.all;
pcacomp.avg = repmat(coeff(:,n),[1,length(pcacomp.time)]);

cfg = [];
cfg.layout = 'GSN-HydroCel-128.sfp';
cfg.parameter = 'avg';
cfg.interpolatenan = 'no';
cfg.zlim =[-0.2 0.2];
cfg.comment    = 'no';
ft_topoplotER(cfg, pcacomp)
colorbar
saveas(gcf,sprintf('/data/liuzzil2/UMD_Flanker/results/correctCI-commission_pc%d.jpg',n))
%%

figure(4); set(gcf,'color','w','position',[239 42 1313 380])
co  = get(gca,'colororder');
stimname =  'cue';

load('cue_grandavg_erp.mat')
N = (grandavg.age12.all.dof(1,1) + grandavg.age15.all.dof(1,1) + grandavg.age18.all.dof(1,1));
grandavg.all.all.avg = (grandavg.age12.all.avg*grandavg.age12.all.dof(1,1) + ...
    grandavg.age15.all.avg*grandavg.age15.all.dof(1,1) + ...
    grandavg.age18.all.avg*grandavg.age18.all.dof(1,1)) / N;

grandavg.all.all.var =  (grandavg.age12.all.var*(grandavg.age12.all.dof(1,1) -1 ) +...
    grandavg.age15.all.var*(grandavg.age15.all.dof(1,1) - 1) + ...
    grandavg.age18.all.var*(grandavg.age18.all.dof(1,1) - 1) )/ (N-1);

grandavg.all.all.dof = ones(size(grandavg.all.all.dof))*N;

%         load('pcaSubConcat_correct_congruent_n335.mat')
% load('pcaSubConcat_flan_correct_commission_n335.mat')
%         yl = [-20 40]; xl = [-0.5 1];

for n = 1:2
    if n == 1
        yl = [-50 4];
    elseif n ==2
        yl = [-6 6];
    end
    xl = [-0.6 0.6];
    t1 = -0.5; t2 = 0;
    
    time = grandavg.all.all.time;
    
    fieldnames = {'all','age12','age15','age18'};
    for jj = 1:length(fieldnames)
        subplot(2,4,(n-1)*4 + jj)
        hold all
        erp1 = sum(grandavg.(fieldnames{jj}).all.avg .* coeff(:,n),1) ;
        %     erp1 = erp1 - mean(erp1(timebl));
        erp1v = sqrt(sum(((grandavg.(fieldnames{jj}).all.var ) .* abs(coeff(:,n))).^2,1))...
            ./ sqrt(grandavg.(fieldnames{jj}).all.dof(1,:));
        plot(time, erp1, 'color',co(1,:),'linewidth',2)
        
        fill([time, fliplr(time)],[erp1 + erp1v, fliplr(erp1 - erp1v)], co(1,:),...
            'edgecolor','none','facealpha',0.2  )
        
        
        grid on;  xlabel('time (s)'); ylim(yl); xlim(xl)
        title(sprintf('%s %s, N=%d',fieldnames{jj},stimname,...
            grandavg.(fieldnames{jj}).all.dof(1)))
        
    end
end
saveas(gcf,sprintf('/data/liuzzil2/UMD_Flanker/results/cue_summaryfig_ages_pc12.jpg'))

%%


datapath = '/data/liuzzil2/UMD_Flanker/derivatives/';
cd(datapath)
subsdir = dir;
subsdir(1:2) = [];
subslist = cell(length(subsdir ) ,1);
close all
% figure; set(gcf,'color','w','position', [  1          50        1920         869]);

behaveTable = readtable('/data/liuzzil2/UMD_Flanker/results/behavioral_table_accuracy70.csv');

% sub = 6257, age 18, only 25 trials, jj = 98
samplings = [30 40];
stimnames =  {'flan','resp','cue'};

nf = 2; %length(samplings);
load('/data/liuzzil2/UMD_Flanker/results/correctCI_commission_allstims.mat')

% for ff = 1%:2

ff =1;

downsampf = samplings(ff);
twind = 0.1;
% tstep = round( dowsampf * twind /2) ;
tstep = round( downsampf * 0.04) ;
xstep =  1;% round( downsampf * 0.025) ;
ncomp = 2;
%%
ss= 2;
%     for ss  = 1:length(stimnames)

stimname =  stimnames{ss};


if strcmp(stimname,'flan')
    xl = [-0.4 0.6];
    load('/data/liuzzil2/UMD_Flanker/results/flan_grandavg_erp.mat')
    %             load('/data/liuzzil2/UMD_Flanker/results/pcaSubConcat_correct_congruent_n335.mat')
    taskepoch = [-0.5 1];
elseif  strcmp(stimname,'cue')
    xl = [-0.6 0.6];
    load('/data/liuzzil2/UMD_Flanker/results/cue_grandavg_erp.mat')
    %             load('/data/liuzzil2/UMD_Flanker/results/pcaSubConcat_correct_congruent_n335.mat')
    taskepoch = [-0.75 0.75];
    
else
    xl = [-0.5 0.5];
    load('/data/liuzzil2/UMD_Flanker/results/resp_grandavg_erp.mat')
    %             load('/data/liuzzil2/UMD_Flanker/results/pcaSubConcat_resp_correct_commiss_n317.mat')
    
    taskepoch = [-1 1];
end


cond = 'All';

ages = [12,15,18];
figure(1);  clf; set(gcf,'position',[  220         181        1306         720],'color','w')
figure(2); clf; set(gcf,'position',[ 640         580        1066         339],'color','w')
figure(3); clf; set(gcf,'position',[ 640         580        1066         339],'color','w')
for aa = 1:3
    age = ages(aa);
    k = 0;
    agegroup = num2str(age);
    
    for jj = 1:length(subsdir)
        sub = subsdir(jj).name(5:end);
        
        
        subdir = ['sub-',sub];
        
        filebids = [subdir,'_task-flanker_eeg'];
        
        outputfolder = sprintf('/data/liuzzil2/UMD_Flanker/derivatives/sub-%s/age-%s/',sub,agegroup);
        
        filename = sprintf('%sAtest3_%s_%dcomps_f%dHzlowp_norm2_w%dms_step%d_xstep%d.mat',...
            outputfolder,stimname,ncomp,downsampf,twind*1000,tstep,xstep);
        bv = behaveTable( behaveTable.sdan == str2double(sub) & behaveTable.age == age,:);
        if exist( filename,'file') && ~isempty(bv)
            
            load(filename)
            if k == 0
                eigval = zeros(2,length(Aall.(cond).time),200);
                eigvec1 = zeros(2,length(Aall.(cond).time),200);
                eigvec2 = zeros(2,length(Aall.(cond).time),200);
                ressd = zeros([size(Aall.(cond).residuals),200]);
                resm = zeros([size(Aall.(cond).residualm),200]);
                erp = zeros([size(Aall.(cond).erpm),200]);
                x =  -22.05 : 0.1 : 22.05 ;
                erphist = zeros([length(x),200]);
                A = zeros(4,length(Aall.(cond).time),200);
            end
            
            %                     figure; plot(Aall.(cond).ff, Aall.(cond).Presidual )
            k = k +1;
            
            A(:,:,k) = reshape(Aall.(cond).A, [4,length(Aall.(cond).time)]);
            
            eigval(:,:,k) = Aall.(cond).eig;
            for tt = 1:length(Aall.(cond).time)
                [V,D]= eig(Aall.(cond).A(:,:,tt));  %A*V = V*D.
                eigvec1(:,tt,k) = V(:,1);
                eigvec2(:,tt,k) = V(:,2);
            end
            ressd(:,:,k) = Aall.(cond).residuals;
            resm(:,:,k) = Aall.(cond).residualm;
            erp(:,:,k) = Aall.(cond).erpm;
            [~,iia,~]= intersect(round(x*100),round(Aall.(cond).erphist(1,:)*100));
            erphist(iia,k) = Aall.(cond).erphist(2,:);
            %                     figure; plot(Aall.(cond).time, eigv)
            %                     legend('A_{1to1}','A_{2to1}','A_{1to2}','A_{2to2}')
            
            %                     figure; plot(timee, Aall.(cond).EEGresidual)
        end
        
        
    end
    A(:,:,k+1:end) = [];
    ressd(:,:,k+1:end) = [];
    resm(:,:,k+1:end) = [];
    erp(:,:,k+1:end) = [];
    eigval(:,:,k+1:end) = [];
    eigvec1(:,:,k+1:end) = [];
    eigvec2(:,:,k+1:end) = [];
    erphist(:,k+1:end) = [];
    
    eigval1corr = zeros(4,k);
    eigval2corr = zeros(4,k);
    for iik = 1:k
        eigval1corr(:,iik) = corr(A(:,:,iik)', abs(eigval(1,:,iik))');
        eigval2corr(:,iik) = corr(A(:,:,iik)', abs(eigval(2,:,iik))');
    end
    figure(4); bar( [mean(eigval1corr,2), mean(eigval2corr,2)]')
    legend('A_{11}','A_{12}','A_{21}','A_{22}','location','best')
    
    %             timee = [sort(Aall.(cond).time(1)-(1/downsampf) : -(1/downsampf) : taskepoch(1),'ascend') , ...
    %                 Aall.(cond).time, ...
    %                 Aall.(cond).time(end)+(1/downsampf) : (1/downsampf) :taskepoch(2)];
    %
    time = Aall.(cond).time;
    

    figure(1);
    co = get(gca,'colororder');
    subplot(2,3,aa); cla
    plot(time, mean(abs(eigvec1),3),'linewidth',2); hold on
    plot(time, mean(abs(eigval(1,:,:)),3),'k','linewidth',2)
    fill([time, fliplr(time)],[mean(abs(eigvec1(1,:,:)),3) + std(abs(eigvec1(1,:,:)),[],3)/sqrt(size(eigvec1,3)), ...
        fliplr(mean(abs(eigvec1(1,:,:)),3) - std(abs(eigvec1(1,:,:)),[],3)/sqrt(size(eigvec1,3)))],co(1,:),...
        'facealpha',0.2,'edgecolor','none')
    
    fill([time, fliplr(time)],[mean(abs(eigvec1(2,:,:)),3) + std(abs(eigvec1(2,:,:)),[],3)/sqrt(size(eigvec1,3)), ...
        fliplr(mean(abs(eigvec1(2,:,:)),3) - std(abs(eigvec1(2,:,:)),[],3)/sqrt(size(eigvec1,3)))],co(2,:),...
        'facealpha',0.2,'edgecolor','none')
    
    fill([time, fliplr(time)],[mean(abs(eigval(1,:,:)),3) + std(abs(eigval(1,:,:)),[],3)/sqrt(size(eigval,3)), ...
        fliplr(mean(abs(eigval(1,:,:)),3) - std(abs(eigval(1,:,:)),[],3)/sqrt(size(eigval,3)))],[0.5 0.5 0.5],...
        'facealpha',0.2,'edgecolor','none')
    
    title('First eigenvector'); grid on; ylim([0.3 0.9])
    if aa == 3
        legend('|V_{11}|','|V_{12}|','D_1',...
            'location','best')
    end
    subplot(2,3,aa+3); cla
    plot(time, mean(abs(eigvec2),3),'linewidth',2)
    hold on
    plot(time, mean(abs(eigval(2,:,:)),3),'k','linewidth',2)
    
    fill([time, fliplr(time)],[mean(abs(eigvec2(1,:,:)),3) + std(abs(eigvec2(1,:,:)),[],3)/sqrt(size(eigvec2,3)), ...
        fliplr(mean(abs(eigvec2(1,:,:)),3) - std(abs(eigvec2(1,:,:)),[],3)/sqrt(size(eigvec2,3)))],co(1,:),...
        'facealpha',0.2,'edgecolor','none')
    
    fill([time, fliplr(time)],[mean(abs(eigvec2(2,:,:)),3) + std(abs(eigvec2(2,:,:)),[],3)/sqrt(size(eigvec2,3)), ...
        fliplr(mean(abs(eigvec2(2,:,:)),3) - std(abs(eigvec2(2,:,:)),[],3)/sqrt(size(eigvec2,3)))],co(2,:),...
        'facealpha',0.2,'edgecolor','none')
    
    fill([time, fliplr(time)],[mean(abs(eigval(2,:,:)),3) + std(abs(eigval(2,:,:)),[],3)/sqrt(size(eigval,3)), ...
        fliplr(mean(abs(eigval(2,:,:)),3) - std(abs(eigval(2,:,:)),[],3)/sqrt(size(eigval,3)))],[0.5 0.5 0.5],...
        'facealpha',0.2,'edgecolor','none')
    
    title('Second eigenvector'); grid on; ylim([0.3 0.9])
    if aa == 3
        legend('|V_{21}|','|V_{22}|','D_2',...
            'location','best')
    end
    figure(2);
    subplot(2,4,2)
    plot(time, mean(erp(1,:,:),3),'linewidth',2); hold on
    %             fill([time, fliplr(time)],[mean((erp(1,:,:)),3) + std((erp(1,:,:)),[],3)/sqrt(size(erp,3)), ...
    %                 fliplr(mean((erp(1,:,:)),3) - std((erp(1,:,:)),[],3)/sqrt(size(erp,3)))],co(aa,:),...
    %                 'facealpha',0.2,'edgecolor','none')
    
    title('PC 1')
    xlim(xl); grid on
    
    subplot(2,4,6)
    plot(time, mean(erp(2,:,:),3),'linewidth',2); hold on
    %               fill([time, fliplr(time)],[mean((erp(2,:,:)),3) + std((erp(2,:,:)),[],3)/sqrt(size(erp,3)), ...
    %                 fliplr(mean((erp(2,:,:)),3) - std((erp(2,:,:)),[],3)/sqrt(size(erp,3)))],co(aa,:),...
    %                 'facealpha',0.2,'edgecolor','none')
    title('PC 2')
    xlim(xl); grid on
    
    
    subplot(2,4,3)
    plot(time, mean(resm(1,:,:),3),'linewidth',2)
    hold on
    %             fill([time, fliplr(time)],[mean((resm(1,:,:)),3) + std((resm(1,:,:)),[],3)/sqrt(size(resm,3)), ...
    %                 fliplr(mean((resm(1,:,:)),3) - std((resm(1,:,:)),[],3)/sqrt(size(resm,3)))],co(aa,:),...
    %                 'facealpha',0.2,'edgecolor','none')
    title('Mean of residuals')
    xlim(xl); ylim([-0.1 0.1]); grid on
    
    subplot(2,4,7)
    plot(time, mean(resm(2,:,:),3),'linewidth',2)
    hold on
    %              fill([time, fliplr(time)],[mean((resm(2,:,:)),3) + std((resm(2,:,:)),[],3)/sqrt(size(resm,3)), ...
    %                 fliplr(mean((resm(2,:,:)),3) - std((resm(2,:,:)),[],3)/sqrt(size(resm,3)))],co(aa,:),...
    %                 'facealpha',0.2,'edgecolor','none')
    title('Mean of residuals')
    xlim(xl); ylim([-0.1 0.1]);grid on
    
    subplot(2,4,4)
    plot(time, mean(ressd(1,:,:),3),'linewidth',2)
    hold on
    %              fill([time, fliplr(time)],[mean((ressd(1,:,:)),3) + std((ressd(1,:,:)),[],3)/sqrt(size(ressd,3)), ...
    %                 fliplr(mean((ressd(1,:,:)),3) - std((ressd(1,:,:)),[],3)/sqrt(size(ressd,3)))],co(aa,:),...
    %                 'facealpha',0.2,'edgecolor','none')
    title('Std of residuals')
    xlim(xl); grid on
    
    subplot(2,4,8)
    plot(time, mean(ressd(2,:,:),3),'linewidth',2)
    hold on
    %             fill([time, fliplr(time)],[mean((ressd(2,:,:)),3) + std((ressd(2,:,:)),[],3)/sqrt(size(ressd,3)), ...
    %                 fliplr(mean((ressd(2,:,:)),3) - std((ressd(2,:,:)),[],3)/sqrt(size(ressd,3)))],co(aa,:),...
    %                 'facealpha',0.2,'edgecolor','none')
    title('Std of residuals')
    xlim(xl); grid on
    
    
    subplot(2,4,[1,5])
    plot(x, mean(erphist,2),'linewidth',2)
    hold on
    title('Datapoint distribution')
    grid on; xlim([-10 10])
    
    
    if aa == 3
        legend('12 yo','15 yo','18 yo','location','best')
    end
    figure(3);
    subplot(1,3,aa)
    plot(time, mean(abs(A),3),'linewidth',2)
    title(sprintf('%d yo',age));
    if aa == 3
        legend('A_{11}','A_{12}','A_{21}','A_{22}','location','best')
    end
    xlim(xl); grid on; ylim([0 1])
    
end
%         figure(1);
%         saveas(gcf,sprintf('/data/liuzzil2/UMD_Flanker/results/%s_%dHz_%s_eig_summary.jpg',stimname,downsampf,cond))
%
%         figure(2);
%         saveas(gcf,sprintf('/data/liuzzil2/UMD_Flanker/results/%s_%dHz_%s_residuals_summary.jpg',stimname,downsampf,cond))
%
%         figure(3);
%         saveas(gcf,sprintf('/data/liuzzil2/UMD_Flanker/results/%s_%dHz_%s_A_summary.jpg',stimname,downsampf,cond))


%     end
% end