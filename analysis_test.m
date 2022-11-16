clear% clear matlab workspace
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

erpcorrect = cell(1,1);
erpcommission = cell(1,1);
erpcongruentC = cell(1,1);
erpincongruentC = cell(1,1);
erpLeft = cell(1,1);
erpRight = cell(1,1);
erpcommissionI = cell(1,1);
erpcongruent = cell(1,1);
erpincongruent = cell(1,1);
erpall = cell(1,1);
erpomission = cell(1,1);


% sub = 6257, age 18, only 25 trials, jj = 98
stimname = 'cue';

if strcmp(stimname, 'flan')
    tbasel = [-0.5 -0.1];
elseif  strcmp(stimname, 'resp')
%     tbasel = [-0.3 -0.1];
    tbasel = [-1 -0.5];
else  

    tbasel = [-0.75 0];

end

behaveTable = readtable('/data/liuzzil2/UMD_Flanker/results/behavioral_table_accuracy70.csv');
 
%%
k = 0;
ntrials = [];
ntrials.sub = zeros(length(subsdir),1);
ntrials.age12 = zeros(length(subsdir),1);
ntrials.age15 = zeros(length(subsdir),1);
ntrials.age18 = zeros(length(subsdir),1);


for jj = 1:length(subsdir)
    sub = subsdir(jj).name(5:end);
    ntrials.sub(jj) = str2double(sub);
    for age = [12,15,18]
        agegroup = num2str(age);
        
        subdir = ['sub-',sub];
        
        filebids = [subdir,'_task-flanker_eeg'];
        
        outputfolder = sprintf('/data/liuzzil2/UMD_Flanker/derivatives/sub-%s/age-%s/',sub,agegroup);
        bidsfolder = sprintf('/data/liuzzil2/UMD_Flanker/bids/sub-%s/age-%s/eeg/',sub,agegroup);
        if strcmp(stimname, 'flan')
            filename = [filebids '_processed_data.set'];
            rejtrialsf = [filebids '_rejected_trials.mat']; 
        elseif strcmp(stimname, 'resp')
            filename = [filebids '_processed_data_resp.set'];
            rejtrialsf = [filebids '_rejected_trials_resp.mat'];
        else
            filename = [filebids '_processed_data_cue.set'];
            rejtrialsf = [filebids '_rejected_trials_cue.mat'];
        end
        filenamebids = [bidsfolder,filebids ];
        
        bv = behaveTable( behaveTable.sdan == str2double(sub) & behaveTable.age == age,:);
        
        if exist( [outputfolder,filename],'file') && ~isempty(bv)
            
            EEG = [];
            EEG = pop_loadset(filename , outputfolder);
            load([outputfolder,rejtrialsf]) %,'rejtrials'
            data = eeglab2fieldtrip(EEG,'raw');
            
            cfg = [];
            cfg.demean        = 'yes';
            cfg.baselinewindow = tbasel ;
            data = ft_preprocessing(cfg,data);
            
            if length(data.trial) > 50 % only use datasets with at least 50 trials
                
                tempTable = array2table({sub,age,length(data.trial)}, 'VariableNames',["sub","age","ntrials"]);
                if k == 0
                    subTable = tempTable;
                else
                    subTable = cat(1,subTable,tempTable);
                end
                k = k +1;
                %% STEP 1: Import data file and events information
                headerfile = [filenamebids, '.set'];
                datafile = [filenamebids, '.fdt'];
                
                hdr = pop_loadset([filebids,'.set']   ,  bidsfolder);
                events = hdr.event;
                
                ntrsp = 0;
                tablenames =  ["cel#","obs#","rsp#","eval","rtim","trl#","CoNo","SITI",...
                    "BkNb","TotN","TRLN","TRTP","Feed","RPTP"];
                
                if strcmp(stimname,'resp')
                    eventname = 'resp';
                elseif strcmp(stimname,'flan') 
                    eventname = 'FLAN';
                else
                    eventname = 'Cue+';
                end
                    
                t = 1000;
                for ii = 1:length(events)
                    
                    if strcmp(events(ii).type,eventname) && length(events(ii).codes) == 4
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
                
                eventclean = eventTable(~rejtrials,:);
                
                %% 4 conditions: correclty pressed left, correctly pressed right
                correct = strcmp(eventclean.RPTP , 'Correct') ;
                commission = strcmp(eventclean.RPTP , 'Commission') ;
                omission =  strcmp(eventclean.RPTP , 'Omission') ;
                if ~strcmp(stimname, 'cue')
                    cfg = [];
                    cfg.trials = correct;
                    erpcorrect{k} = ft_timelockanalysis(cfg, data);

                    cfg = [];
                    cfg.trials = commission;
                    erpcommission{k} = ft_timelockanalysis(cfg, data);

                    correctI = strcmp(eventclean.CoNo , 'Incongruent') & correct;
                    correctC = strcmp(eventclean.CoNo , 'Congruent') & correct;
                    cfg = [];
                    cfg.trials = correctC;
                    erpcongruentC{k} = ft_timelockanalysis(cfg, data);

                    cfg = [];
                    cfg.trials = correctI;
                    erpincongruentC{k} = ft_timelockanalysis(cfg, data);


                    commissionI = strcmp(eventclean.CoNo , 'Incongruent') & commission;
                    cfg = [];
                    cfg.trials = commissionI;
                    erpcommissionI{k} = ft_timelockanalysis(cfg, data);


                    incongruent = strcmp(eventclean.CoNo , 'Incongruent') & ~omission;
                    congruent = strcmp(eventclean.CoNo , 'Congruent') & ~omission;
                    cfg = [];
                    cfg.trials = congruent;
                    erpcongruent{k} = ft_timelockanalysis(cfg, data);

                    cfg = [];
                    cfg.trials = incongruent;
                    erpincongruent{k} = ft_timelockanalysis(cfg, data);
                end
                
                cfg = [];
                cfg.trials = ~omission;
                erpall{k} = ft_timelockanalysis(cfg, data);
                
                if nnz(omission) > 0
                    cfg = [];
                    cfg.trials = omission;
                    erpomission{k} = ft_timelockanalysis(cfg, data);
                end
                
                %             cfg = [];
                %             cfg.layout = 'GSN-HydroCel-128.sfp';
                %             cfg.interactive = 'no';
                %             cfg.showoutline = 'yes';
                %             figure
                %             ft_multiplotER(cfg, erpcorrect, erpcommission)
                
                
                
                %% use ft_timelockanalysis to compute the ERPs
                
                
                
                right = cell2mat(eventclean.("rsp#")) == 4;
                left = cell2mat(eventclean.("rsp#")) == 1;
                
             
                
                
                cfg = [];
                cfg.trials = right;
                erpRight{k} = ft_timelockanalysis(cfg, data);
                
                
                cfg = [];
                cfg.trials = left;
                erpLeft{k} = ft_timelockanalysis(cfg, data);
                
                
                if age == 12
                    ntrials.age12(jj) = nnz(right | left);
                elseif age == 15
                    ntrials.age15(jj) = nnz(right | left);
                else
                    ntrials.age18(jj) = nnz(right | left);
                end
          
                
                
                
            end
        end
    end
end

clc
fprintf('Number of included recordings: n = %d\n',k)
%%
grandavg = [];

agefields= {'age12','age15','age18','all'};
for ii = 1:4
age = agefields{ii};
if strcmp(age,'all')
    agegroup = true(size(erpall));
else
    agegroup = cell2mat(subTable.age) == str2double(age(4:5));
    na = find(agegroup);
end
grandavg.(agefields{ii}) = [];
% agegroup(na(83:end)) = 0;
cfg = [];
cfg.parameter = 'avg';
grandavg.(agefields{ii}).correct = ft_timelockgrandaverage(cfg, erpcorrect{agegroup} );
grandavg.(agefields{ii}).commission = ft_timelockgrandaverage(cfg, erpcommission{agegroup} );
grandavg.(agefields{ii}).commissionI = ft_timelockgrandaverage(cfg, erpcommissionI{agegroup} );
grandavg.(agefields{ii}).congruentC = ft_timelockgrandaverage(cfg, erpcongruentC{agegroup} );
grandavg.(agefields{ii}).incongruentC = ft_timelockgrandaverage(cfg, erpincongruentC{agegroup} );
grandavg.(agefields{ii}).congruent = ft_timelockgrandaverage(cfg, erpcongruent{agegroup} );
grandavg.(agefields{ii}).incongruent = ft_timelockgrandaverage(cfg, erpincongruent{agegroup} );
grandavg.(agefields{ii}).respLeft = ft_timelockgrandaverage(cfg, erpLeft{agegroup} );
grandavg.(agefields{ii}).respRight = ft_timelockgrandaverage(cfg, erpRight{agegroup} );
grandavg.(agefields{ii}).all = ft_timelockgrandaverage(cfg, erpall{agegroup} );
% grandavg.(agefields{ii}).omission = ft_timelockgrandaverage(cfg, erpomission{agegroup} );

% 
end
save(sprintf('/data/liuzzil2/UMD_Flanker/results/%s_grandavg_erp.mat',stimname),'grandavg')

%% Plot correct vs commission, congruent vs incongruent and Right vs LEft resp

cfg = [];
cfg.layout = 'GSN-HydroCel-128.sfp';
cfg.interactive = 'no';
cfg.showoutline = 'yes';
figure
ft_multiplotER(cfg, grandavg.all.correct, grandavg.all.commission)
set(gcf,'color','w','position',[ 2666   423  1055   849])
saveas(gcf,['/data/liuzzil2/UMD_Flanker/results/topoplot_correct_commission_n' num2str(k) '_' stimname '.jpg'])

cfg = [];
cfg.layout = 'GSN-HydroCel-128.sfp';
cfg.interactive = 'no';
cfg.showoutline = 'yes';
figure
ft_multiplotER(cfg, grandavg.all.congruentC, grandavg.all.incongruentC)
set(gcf,'color','w','position',[ 2666   423  1055   849])
saveas(gcf,['/data/liuzzil2/UMD_Flanker/results/topoplot_congruentC_incongruentC_n' num2str(k) '_' stimname '.jpg'])


cfg = [];
cfg.layout = 'GSN-HydroCel-128.sfp';
cfg.interactive = 'no';
cfg.showoutline = 'yes';
figure
ft_multiplotER(cfg, grandavg.all.respRight, grandavg.all.respLeft)
set(gcf,'color','w','position',[ 2666   423  1055   849])
saveas(gcf,['/data/liuzzil2/UMD_Flanker/results/topoplot_CRL_n' num2str(k) '_' stimname '.jpg'])


%%
s = 'commission';
cfg = [];
cfg.parameter = 'avg';
cfg.keepindividual = 'yes' ;
grandavg_correct = ft_timelockgrandaverage(cfg, erpcorrect{:} );
grandavg_commission = ft_timelockgrandaverage(cfg, erpcommission{:} );
grandavg_alltrials = ft_timelockgrandaverage(cfg, erpall{:} );
grandav = cat(1, grandavg_correct.individual, grandavg_commission.individual );
grandav = reshape(permute(grandav,[2,3,1]),[104,length(grandavg_alltrials.time)*k*2])';


% grandav = grandavg_alltrials.individual ;
% grandav = reshape(permute(grandav,[2,3,1]),[104,length(grandavg_alltrials.time)*k])';
% 
% grandav = cat(2,grandavg.age12.all.avg,grandavg.age15.all.avg,grandavg.age18.all.avg)';


% if strcmp(s,'commission')
%     grandav = (grandavg_correct.avg + grandavg_commission.avg)'/2;
% else
%     grandav = (grandavg_erpL.avg + grandavg_erpR.avg)'/2;   
% end

% grandav = reshape(permute(grandavg_correct.individual,[2,3,1]),[104,375*335])';
% grandav = reshape(permute(grandavg_correct.individual,[2,3,1]),[104,375*335*2])';

% grandav = cat(2, grandavg.all.congruent.avg, grandavg.all.incongruent.avg,...
%     grandavg.age12.congruent.avg, grandavg.age12.incongruent.avg, ...
%     grandavg.age15.congruent.avg, grandavg.age15.incongruent.avg, ...
%     grandavg.age18.congruent.avg, grandavg.age18.incongruent.avg)';

gradav = cat(2, grandavg.age12.congruentC.avg, grandavg.age12.incongruentC.avg, grandavg.age12.commission.avg,...
    grandavg.age15.congruentC.avg, grandavg.age15.incongruentC.avg, grandavg.age15.commission.avg,...
    grandavg.age18.congruentC.avg, grandavg.age18.incongruentC.avg, grandavg.age18.commission.avg)';


[coeff, score] = pca(grandav);
figure
plot(cumsum(var(score))/ sum(var(score)))
xlabel('principal components'); ylabel('explained variance')
xlim([0 20])

saveas(gcf,'/data/liuzzil2/UMD_Flanker/results/PCA_variance.jpg')


%%
load('/data/liuzzil2/UMD_Flanker/results/cue_grandavg_erp.mat')
grandav = cat(2,grandavg.age12.all.avg,grandavg.age15.all.avg,grandavg.age18.all.avg);

load('/data/liuzzil2/UMD_Flanker/results/flan_grandavg_erp.mat')
grandav = cat(2,grandav, grandavg.age12.congruentC.avg, grandavg.age12.incongruentC.avg, grandavg.age12.commission.avg,...
    grandavg.age15.congruentC.avg, grandavg.age15.incongruentC.avg, grandavg.age15.commission.avg,...
    grandavg.age18.congruentC.avg, grandavg.age18.incongruentC.avg, grandavg.age18.commission.avg);

load('/data/liuzzil2/UMD_Flanker/results/resp_grandavg_erp.mat')
grandav = cat(2,grandav, grandavg.age12.congruentC.avg, grandavg.age12.incongruentC.avg, grandavg.age12.commission.avg,...
    grandavg.age15.congruentC.avg, grandavg.age15.incongruentC.avg, grandavg.age15.commission.avg,...
    grandavg.age18.congruentC.avg, grandavg.age18.incongruentC.avg, grandavg.age18.commission.avg);


[coeff, score] = pca(grandav');
figure; set(gcf,'color','w'); cla
plot(cumsum(var(score))/ sum(var(score)),'linewidth',1)
hold on
plot(cumsum(var(score))/ sum(var(score)),'k.','MarkerSize',25)
xlabel('principal components'); ylabel('explained variance')
xlim([0 10]); grid on; title('Cumulative variance explained')

save('/data/liuzzil2/UMD_Flanker/results/correctCI_commission_allstims','coeff')

%% Can we see difference in correct-commision in PCA?

figure; %set(gcf,'color','w','position',[2588   490  727  731]);
set(gcf,'color','w','position',[171     4   826   660]);
clim = [-0.2 0.2];
iin =0;
for cc = 1:3
    pcacomp = grandavg.age12.correct;
    pcacomp.avg = repmat(coeff(:,cc+iin),[1,length(pcacomp.time)]);
    
    cfg = [];
    cfg.layout = 'GSN-HydroCel-128.sfp';
    cfg.parameter = 'avg';
    cfg.interpolatenan = 'no';
    cfg.zlim =clim;
    cfg.comment    = 'no';
    
    if cc == 1
%         y = [-40 80];
        y = [-40 60];
    elseif cc == 2
%         y = [-20 40];
        y = [-20 40];
    else
        y= [-20 20];
%         y= [-15 15];
    end
        
    
    subplot(3,4,(cc-1)*4 + 1)
    ft_topoplotER(cfg, pcacomp)
    title(sprintf('PC %d',cc+iin))
    
    subplot(3,4,(cc-1)*4 + 2)
    hold on
    if strcmp(s,'commission')
        plot(data.time{1}, sum(grandavg.age12.correct.avg .* coeff(:,cc+iin),1))
        plot(data.time{1}, sum(grandavg.age12.commission.avg .* coeff(:,cc+iin),1))
    else
        plot(data.time{1}, sum(grandavg.age12.congruent.avg .* coeff(:,cc+iin),1))
        plot(data.time{1}, sum(grandavg.age12.incongruent.avg .* coeff(:,cc+iin),1))
    end
    ylim(y)
    grid on
    xlabel('time'); title('Age 12')
    subplot(3,4,(cc-1)*4 + 3)
    hold on
    if strcmp(s,'commission')
        plot(data.time{1}, sum(grandavg.age15.correct.avg .* coeff(:,cc+iin),1))
        plot(data.time{1}, sum(grandavg.age15.commission.avg .* coeff(:,cc+iin),1))
    else
        plot(data.time{1}, sum(grandavg.age15.congruent.avg .* coeff(:,cc+iin),1))
        plot(data.time{1}, sum(grandavg.age15.incongruent.avg .* coeff(:,cc+iin),1))
    end
   
    ylim(y)
    grid on
    xlabel('time'); title('Age 15')
    subplot(3,4,(cc-1)*4 + 4)
    hold on
    if strcmp(s,'commission')
        plot(data.time{1}, sum(grandavg.age18.correct.avg .* coeff(:,cc+iin),1))
        plot(data.time{1}, sum(grandavg.age18.commission.avg .* coeff(:,cc+iin),1))
    else
        plot(data.time{1}, sum(grandavg.age18.congruent.avg .* coeff(:,cc+iin),1))
        plot(data.time{1}, sum(grandavg.age18.incongruent.avg .* coeff(:,cc+iin),1))
    end
   
    ylim(y)
    grid on
    xlabel('time'); title('Age 18')
end
subplot(3,4,10)
if strcmp(s,'commission')
    legend('correct', 'commission','location','best')
    saveas(gcf,['/data/liuzzil2/UMD_Flanker/results/pcaSubConcat',num2str(1+iin),...
        '_',num2str(3+iin),'_',stimname,'_correct_commission_n',num2str(k),'_ages.jpg'])
    save(['/data/liuzzil2/UMD_Flanker/results/pcaSubConcat_' stimname '_correct_commission_n',num2str(k)],'coeff')
else
    legend('congruent', 'incongruent','location','best')
    saveas(gcf,['/data/liuzzil2/UMD_Flanker/results/pcaSubConcat',num2str(1+iin),...
        '_',num2str(3+iin),'_',stimname,'_correct_congruent_n',num2str(k),'_ages.jpg'])
    save(['/data/liuzzil2/UMD_Flanker/results/pcaSubConcat_' stimname '_correct_congruent_n',num2str(k)],'coeff')
end

