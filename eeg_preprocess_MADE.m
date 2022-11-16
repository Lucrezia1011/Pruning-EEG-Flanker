% ************************************************************************
% The Maryland Analysis of Developmental EEG (UMADE) Pipeline
% Version 1.0
% Developed at the Child Development Lab, University of Maryland, College Park

% Contributors to MADE pipeline:
% Ranjan Debnath (rdebnath@umd.edu)
% George A. Buzzell (gbuzzell@umd.edu)
% Santiago Morales Pamplona (moraless@umd.edu)
% Stephanie Leach (sleach12@umd.edu)
% Maureen Elizabeth Bowers (mbowers1@umd.edu)
% Nathan A. Fox (fox@umd.edu)

% MADE uses EEGLAB toolbox and some of its plugins. Before running the pipeline, you have to install the following:
% EEGLab:  https://sccn.ucsd.edu/eeglab/downloadtoolbox.php/download.php

% You also need to download the following plugins/extensions from here: https://sccn.ucsd.edu/wiki/EEGLAB_Extensions

% Specifically, download:
% MFFMatlabIO: https://github.com/arnodelorme/mffmatlabio/blob/master/README.txt
% FASTER: https://sourceforge.net/projects/faster/
% ADJUST: https://www.nitrc.org/projects/adjust/
% Adjusted ADJUST (included in this pipeline):  https://github.com/ChildDevLab/MADE-EEG-preprocessing-pipeline

% After downloading these plugins (as zip files), you need to place it in the eeglab/plugins folder.
% For instance, for FASTER, you uncompress the downloaded extension file (e.g., 'FASTER.zip') and place it in the main EEGLAB "plugins" sub-directory/sub-folder.
% After placing all the required plugins, add the EEGLAB folder to your path by using the following code:

% addpath(genpath(('...')) % Enter the path of the EEGLAB folder in this line

% Please cite the following references for in any manuscripts produced utilizing MADE pipeline:

% EEGLAB: A Delorme & S Makeig (2004) EEGLAB: an open source toolbox for
% analysis of single-trial EEG dynamics. Journal of Neuroscience Methods, 134, 9?21.

% firfilt (filter plugin): developed by Andreas Widmann (https://home.uni-leipzig.de/biocog/content/de/mitarbeiter/widmann/eeglab-plugins/)

% FASTER: Nolan, H., Whelan, R., Reilly, R.B., 2010. FASTER: Fully Automated Statistical
% Thresholding for EEG artifact Rejection. Journal of Neuroscience Methods, 192, 152?162.

% ADJUST: Mognon, A., Jovicich, J., Bruzzone, L., Buiatti, M., 2011. ADJUST: An automatic EEG
% artifact detector based on the joint use of spatial and temporal features. Psychophysiology, 48, 229?240.
% Our group has modified ADJUST plugin to improve selection of ICA components containing artifacts

% This pipeline is released under the GNU General Public License version 3.

% ************************************************************************

%% User input: user provide relevant information to be used for data processing
% Preprocessing of EEG data involves using some common parameters for
% every subject. This part of the script initializes the common parameters.

clear % clear matlab workspace
clc % clear matlab command window
%addpath(genpath('C:\Users\Berger\Documents\eeglab13_4_4b'));% enter the path of the EEGLAB folder in this line
%addpath(genpath('C:\Users\Berger\Documents\eeglab13_4_4b'))
nih_matmod load eeglab
eeglab % call eeglab to set up the plugin
addpath(genpath('/data/liuzzil2/UMD_Flanker/matlab/'));

%%
% 1. Enter the path of the folder that has the raw data to be analyzed
rawdata_location = '/data/liuzzil2/UMD_Flanker/bids/';

% 2. Enter the path of the folder where you want to save the processed data
output_location = '/data/liuzzil2/UMD_Flanker/derivatives/';

% 3. Enter the path of the channel location file
% channel_locations = ['path' filesep 'channel location file name.extension'];

% 4. Do your data need correction for anti-aliasing filter and/or task related time offset?
adjust_time_offset = 1; % 0 = NO (no correction), 1 = YES (correct time offset)

% If your data need correction for time offset, initialize the offset time (in milliseconds)
% YOU NEED TO CHANGE THE "xx" TO A REAL NUMBER WITHOUT QUOTATION MARKS!
filter_timeoffset =  0;     % anti-aliasing time offset (in milliseconds). 0 = No time offset

stimulus_markers = {'Cue+', 'FLAN'};      % enter the stimulus makers that need to be adjusted for time offset
response_markers = {'resp'};       % enter the response makers that need to be adjusted for time offset

% 5. Do you want to down sample the data?
down_sample = 0; % 0 = NO (no down sampling), 1 = YES (down sampling)
% sampling_rate = xxx; % set sampling rate (in Hz), if you want to down sample

% 6. Do you want to delete the outer layer of the channels? (Rationale has been described in MADE manuscript)
%    This fnction can also be used to down sample electrodes. For example, if EEG was recorded with 128 channels but you would
%    like to analyse only 64 channels, you can assign the list of channnels to be excluded in the 'outerlayer_channel' variable.
delete_outerlayer = 1; % 0 = NO (do not delete outer layer), 1 = YES (delete outerlayer);
% If you want to delete outer layer, make a list of channels to be deleted
outerlayer_channel =  {'E17' 'E38' 'E43' 'E44' 'E48' 'E49' 'E113' 'E114' ...
    'E119' 'E120' 'E121' 'E125' 'E126' 'E127' 'E128' 'E56' 'E63' 'E68' 'E73' 'E81' 'E88' 'E94' 'E99' 'E107'}; % list of channels
% recommended list for EGI 128 chanenl net: {'E17' 'E38' 'E43' 'E44' 'E48' 'E49' 'E113' 'E114' 'E119' 'E120' 'E121' 'E125' 'E126' 'E127' 'E128' 'E56' 'E63' 'E68' 'E73' 'E81' 'E88' 'E94' 'E99' 'E107'}

% 7. Initialize the filters
highpass = 0.2; % High-pass frequency
lowpass  = 50; % Low-pass frequency. We recommend low-pass filter at/below line noise frequency (see manuscript for detail)

% 8. Are you processing task-related or resting-state EEG data?
task_eeg = 1; % 0 = resting, 1 = task
task_event_markers = {'Cue+'};%{'resp'};%{'FLAN'}; % enter all the event/condition markers

% 9. Do you want to epoch/segment your data?
epoch_data = 1; % 0 = NO (do not epoch), 1 = YES (epoch data)
task_epoch_length = [-0.75 0.75];%[-1, 1];%[-0.5 1]; % epoch length in second
% rest_epoch_length = xx; % for resting EEG continuous data will be segmented into consecutive epochs of a specified length (here 2 second) by adding dummy events
overlap_epoch = 0;     % 0 = NO (do not create overlapping epoch), 1 = YES (50% overlapping epoch)
% dummy_events ={'xxx'}; % enter dummy events name

% 10. Do you want to remove/correct baseline?
remove_baseline = 1; % 0 = NO (no baseline correction), 1 = YES (baseline correction)
baseline_window = [-0.75  0];%[];%[-0.5  0]; % baseline period in milliseconds (MS) [] = entire epoch
% baseline_window = task_epoch_length;
% 11. Do you want to remove artifact laden epoch based on voltage threshold?
voltthres_rejection = 1; % 0 = NO, 1 = YES
volt_threshold = [-250 250]; % lower and upper threshold (in uV)

% 12. Do you want to perform epoch level channel interpolation for artifact laden epoch? (see manuscript for detail)
interp_epoch = 0; % 0 = NO, 1 = YES.
frontal_channels = {'list of frontal channels'}; % If you set interp_epoch = 1, enter the list of frontal channels to check (see manuscript for detail)
% recommended list for EGI 128 channel net: {'E1', 'E8', 'E14', 'E21', 'E25', 'E32', 'E17'}

%13. Do you want to interpolate the bad channels that were removed from data?
interp_channels = 1; % 0 = NO (Do not interpolate), 1 = YES (interpolate missing channels)

% 14. Do you want to rereference your data?
rerefer_data = 1; % 0 = NO, 1 = YES
reref=[]; % Enter electrode name/s or number/s to be used for rereferencing
% For channel name/s enter, reref = {'channel_name', 'channel_name'};
% For channel number/s enter, reref = [channel_number, channel_number];
% For average rereference enter, reref = []; default is average rereference

% 15. Do you want to save interim results?
save_interim_result = 1; % 0 = NO (Do not save) 1 = YES (save interim results)

% 16. How do you want to save your data? .set or .mat
output_format = 1; % 1 = .set (EEGLAB data structure), 2 = .mat (Matlab data structure)

% ********* no need to edit beyond this point for EGI .mff data **********
% ********* for non-.mff data format edit data import function ***********
% ********* below using relevant data import plugin from EEGLAB **********


clc

datapath = '/data/liuzzil2/UMD_Flanker/bids/';

cd(datapath)
subsdir = dir;
subsdir(1:2) = [];
subslist = cell(length(subsdir ) ,1);


%% Initialize output variables

datafile_names = [];
reference_used_for_faster=[]; % reference channel used for running faster to identify bad channel/s
faster_bad_channels=[]; % number of bad channel/s identified by faster
ica_preparation_bad_channels=[]; % number of bad channel/s due to channel/s exceeding xx% of artifacted epochs
length_ica_data=[]; % length of data (in second) fed into ICA decomposition
total_ICs=[]; % total independent components (ICs)
ICs_removed=[]; % number of artifacted ICs
total_epochs_before_artifact_rejection=[];
total_epochs_after_artifact_rejection=[];
total_channels_interpolated=[]; % total_channels_interpolated=faster_bad_channels+ica_preparation_bad_channels

subject = 0;
%% Loop over all data files


% issue with adjust alogrithm
% jj 30, age 18, 
% jj 51, age 12

% 156, ahe 12, no events


for jj = 1:length(subsdir)
    sub = subsdir(jj).name(5:end);
    
    for age = [12,15,18]
        agegroup = num2str(age);
        
        subdir = ['sub-',sub];
        eegdir = [subdir,'/age-',agegroup,'/eeg/'];
        filebids = [subdir,'_task-flanker_eeg'];
        
        filename = [datapath,eegdir,filebids];
        outputfolder = sprintf('/data/liuzzil2/UMD_Flanker/derivatives/sub-%s/age-%s/',sub,agegroup);
        
        if exist([filename,'.set'],'file') %&& ~exist([outputfolder,filebids '_rejected_trials_cue.mat'],'file') %~exist([outputfolder,filebids '_rejected_trials_resp.mat'],'file')
            
            subject = subject + 1;
            datafile_names = cat(1,datafile_names,{filebids});
            disp(filebids)
            delay_egi = 36;
            if strcmp(agegroup,'12')
                delay_epr = 15;
            else
                delay_epr = 16;
            end
            
            
            stimulus_timeoffset   = (delay_egi + delay_epr); % stimulus related time offset (in milliseconds). 0 = No time offset
            response_timeoffset = delay_egi;    % response related time offset (in milliseconds). 0 = No time offset
           

            EEG = []; EEG_copy = [];
            EEG = pop_loadset([filebids '.set'] , [datapath,eegdir]);
            %% STEP 3: Adjust anti-aliasing and task related time offset
            if adjust_time_offset==1
                % adjust anti-aliasing filter time offset
                if filter_timeoffset~=0
                    for aafto=1:length(EEG.event)
                        EEG.event(aafto).latency=EEG.event(aafto).latency+(filter_timeoffset/1000)*EEG.srate;
                    end
                end
                % adjust stimulus time offset
                if stimulus_timeoffset~=0
                    for sto=1:length(EEG.event)
                        for sm=1:length(stimulus_markers)
                            if strcmp(EEG.event(sto).type, stimulus_markers{sm})
                                EEG.event(sto).latency=EEG.event(sto).latency+(stimulus_timeoffset/1000)*EEG.srate;
                            end
                        end
                    end
                end
                % adjust response time offset
                if response_timeoffset~=0
                    for rto=1:length(EEG.event)
                        for rm=1:length(response_markers)
                            if strcmp(EEG.event(rto).type, response_markers{rm})
                                EEG.event(rto).latency=EEG.event(rto).latency-(response_timeoffset/1000)*EEG.srate;
                            end
                        end
                    end
                end
            end

            %% STEP 4: Change sampling rate
            if down_sample==1
                if floor(sampling_rate) > EEG.srate
                    error ('Sampling rate cannot be higher than recorded sampling rate');
                elseif floor(sampling_rate) ~= EEG.srate
                    EEG = pop_resample( EEG, sampling_rate);
                    EEG = eeg_checkset( EEG );
                end
            end

            %% STEP 5: Delete outer layer of channels
            chans_labels=cell(1,EEG.nbchan);
            for i=1:EEG.nbchan
                chans_labels{i}= EEG.chanlocs(i).labels;
            end
            [chans,chansidx] = ismember(outerlayer_channel, chans_labels);
            outerlayer_channel_idx = chansidx(chansidx ~= 0);
            if delete_outerlayer==1
                if isempty(outerlayer_channel_idx)==1
                    error(['None of the outer layer channels present in channel locations of data.'...
                        ' Make sure outer layer channels are present in channel labels of data (EEG.chanlocs.labels).']);
                else
                    EEG = pop_select( EEG,'nochannel', outerlayer_channel_idx);
                    EEG = eeg_checkset( EEG );
                    channels_analysed = EEG.chanlocs;
                end
            end
                
           if ~exist([outputfolder, filebids '_ica_data.set'], 'file')
                toonoisy =1;
%                 %% STEP 6: Filter data
%                 % Calculate filter order using the formula: m = dF / (df / fs), where m = filter order,
%                 % df = transition band width, dF = normalized transition width, fs = sampling rate
%                 % dF is specific for the window type. Hamming window dF = 3.3
%                 
%                 high_transband = highpass; % high pass transition band
%                 low_transband = 10; % low pass transition band
%                 
%                 hp_fl_order = 3.3 / (high_transband / EEG.srate);
%                 lp_fl_order = 3.3 / (low_transband / EEG.srate);
%                 
%                 % Round filter order to next higher even integer. Filter order is always even integer.
%                 if mod(floor(hp_fl_order),2) == 0
%                     hp_fl_order=floor(hp_fl_order);
%                 elseif mod(floor(hp_fl_order),2) == 1
%                     hp_fl_order=floor(hp_fl_order)+1;
%                 end
%                 
%                 if mod(floor(lp_fl_order),2) == 0
%                     lp_fl_order=floor(lp_fl_order)+2;
%                 elseif mod(floor(lp_fl_order),2) == 1
%                     lp_fl_order=floor(lp_fl_order)+1;
%                 end
%                 
%                 % Calculate cutoff frequency
%                 high_cutoff = highpass/2;
%                 low_cutoff = lowpass + (low_transband/2);
%                 
%                 % Performing high pass filtering
%                 EEG = eeg_checkset( EEG );
%                 EEG = pop_firws(EEG, 'fcutoff', high_cutoff, 'ftype', 'highpass', 'wtype', 'hamming', 'forder', hp_fl_order, 'minphase', 0);
%                 EEG = eeg_checkset( EEG );
%                 
%                 % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%                 
%                 % pop_firws() - filter window type hamming ('wtype', 'hamming')
%                 % pop_firws() - applying zero-phase (non-causal) filter ('minphase', 0)
%                 
%                 % Performing low pass filtering
%                 EEG = eeg_checkset( EEG );
%                 EEG = pop_firws(EEG, 'fcutoff', low_cutoff, 'ftype', 'lowpass', 'wtype', 'hamming', 'forder', lp_fl_order, 'minphase', 0);
%                 EEG = eeg_checkset( EEG );
%                 
%                 % pop_firws() - transition band width: 10 Hz
%                 % pop_firws() - filter window type hamming ('wtype', 'hamming')
%                 % pop_firws() - applying zero-phase (non-causal) filter ('minphase', 0)
%                 
%                 
%                 
%                 %% STEP 8: Prepare data for ICA
%                 EEG_copy=[];
%                 EEG_copy=EEG; % make a copy of the dataset
%                 EEG_copy = eeg_checkset(EEG_copy);
%                 
%                 
%                 v = var(reshape(EEG_copy.data,[EEG_copy.nbchan, EEG_copy.pnts*EEG_copy.trials]),[],2);
%                 ica_prep_badChans = find(v >  5e3);
%                 
%                 
%                 % Perform 1Hz high pass filter on copied dataset
%                 transband = 1;
%                 fl_cutoff = transband/2;
%                 fl_order = 3.3 / (transband / EEG.srate);
%                 
%                 if mod(floor(fl_order),2) == 0
%                     fl_order=floor(fl_order);
%                 elseif mod(floor(fl_order),2) == 1
%                     fl_order=floor(fl_order)+1;
%                 end
%                 
%                 EEG_copy = pop_firws(EEG_copy, 'fcutoff', fl_cutoff, 'ftype', 'highpass', 'wtype', 'hamming', 'forder', fl_order, 'minphase', 0);
%                 EEG_copy = eeg_checkset(EEG_copy);
%                 
%                 % Create 1 second epoch
%                 EEG_copy=eeg_regepochs(EEG_copy,'recurrence', 1, 'limits',[0 1], 'rmbase', [NaN], 'eventtype', '999'); % insert temporary marker 1 second apart and create epochs
%                 EEG_copy = eeg_checkset(EEG_copy);
%                 
%                 % Find bad epochs and delete them from dataset
%                 vol_thrs = [-500 500]; % [lower upper] threshold limit(s) in uV.
%                 emg_thrs = [-100 30]; % [lower upper] threshold limit(s) in dB.
%                 emg_freqs_limit = [20 50]; % [lower upper] frequency limit(s) in Hz.
%                 
%                 % Find channel/s with xx% of artifacted 1-second epochs and delete them
%                 numEpochs =EEG_copy.trials; % find the number of epochs
%                 all_bad_channels=0;
%                 
%                 for ch=1:EEG_copy.nbchan
%                     % Find artifaceted epochs by detecting outlier voltage
%                     EEG_copy = pop_eegthresh(EEG_copy,1, ch, vol_thrs(1), vol_thrs(2), EEG_copy.xmin, EEG_copy.xmax, 0, 0);
%                     EEG_copy = eeg_checkset( EEG_copy );
%                     
%                     % 1         : data type (1: electrode, 0: component)
%                     % 0         : display with previously marked rejections? (0: no, 1: yes)
%                     % 0         : reject marked trials? (0: no (but store the  marks), 1:yes)
%                     
%                     % Find artifaceted epochs by using thresholding of frequencies in the data.
%                     % this method mainly rejects muscle movement (EMG) artifacts
%                     EEG_copy = pop_rejspec( EEG_copy, 1,'elecrange',ch ,'method','fft','threshold', emg_thrs, 'freqlimits', emg_freqs_limit, 'eegplotplotallrej', 0, 'eegplotreject', 0);
%                     
%                     % method                : method to compute spectrum (fft)
%                     % threshold             : [lower upper] threshold limit(s) in dB.
%                     % freqlimits            : [lower upper] frequency limit(s) in Hz.
%                     % eegplotplotallrej     : 0 = Do not superpose rejection marks on previous marks stored in the dataset.
%                     % eegplotreject         : 0 = Do not reject marked trials (but store the  marks).
%                     
%                     % Find number of artifacted epochs
%                     EEG_copy = eeg_checkset( EEG_copy );
%                     EEG_copy = eeg_rejsuperpose( EEG_copy, 1, 1, 1, 1, 1, 1, 1, 1);
%                     artifacted_epochs=EEG_copy.reject.rejglobal;
%                     
%                     % Find bad channel / channel with more than 20% artifacted epochs
%                     if sum(artifacted_epochs) > (numEpochs*20/100)
%                         ica_prep_badChans = cat(1,ica_prep_badChans,ch);
%                         
%                     end
%                 end
%                 ica_prep_badChans = unique(ica_prep_badChans);
%                 % If all channels are bad, save the dataset at this stage and ignore the remaining of the preprocessing.
%                 if numel(ica_prep_badChans)==EEG.nbchan || numel(ica_prep_badChans)+1==EEG.nbchan
%                     %     all_bad_channels=1;
%                     %     warning(['No usable data for datafile', datafile_names{subject}]);
%                     %     if output_format==1
%                     %         EEG = eeg_checkset(EEG);
%                     %         EEG = pop_editset(EEG, 'setname',  strrep(datafile_names{subject}, ext, '_no_usable_data_all_bad_channels'));
%                     %         EEG = pop_saveset(EEG, 'filename', strrep(datafile_names{subject}, ext, '_no_usable_data_all_bad_channels.set'),'filepath', [output_location filesep 'processed_data' filesep ]); % save .set format
%                     %     elseif output_format==2
%                     %         save([[output_location filesep 'processed_data' filesep ] strrep(datafile_names{subject}, ext, '_no_usable_data_all_bad_channels.mat')], 'EEG'); % save .mat format
%                     %     end
%                     
%                 else
%                     % Reject bad channel - channel with more than xx% artifacted epochs
%                     EEG_copy = pop_select( EEG_copy,'nochannel', ica_prep_badChans);
%                     EEG_copy = eeg_checkset(EEG_copy);
%                 end
%                 
%                 %     if numel(ica_prep_badChans)==0
%                 %         ica_preparation_bad_channels{subject}='0';
%                 %     else
%                 %         ica_preparation_bad_channels{subject}=num2str(ica_prep_badChans);
%                 %     end
%                 %
%                 %     if all_bad_channels == 1
%                 %         length_ica_data(subject)=0;
%                 %         total_ICs(subject)=0;
%                 %         ICs_removed{subject}='0';
%                 %         total_epochs_before_artifact_rejection(subject)=0;
%                 %         total_epochs_after_artifact_rejection(subject)=0;
%                 %         total_channels_interpolated(subject)=0;
%                 %         continue % ignore rest of the processing and go to next datafile
%                 %     end
%                 %
%                 % Find the artifacted epochs across all channels and reject them before doing ICA.
%                 EEG_copy = pop_eegthresh(EEG_copy,1, 1:EEG_copy.nbchan, vol_thrs(1), vol_thrs(2), EEG_copy.xmin, EEG_copy.xmax,0,0);
%                 EEG_copy = eeg_checkset(EEG_copy);
%                 
%                 % 1         : data type (1: electrode, 0: component)
%                 % 0         : display with previously marked rejections? (0: no, 1: yes)
%                 % 0         : reject marked trials? (0: no (but store the  marks), 1:yes)
%                 
%                 % Find artifaceted epochs by using power threshold in 20-40Hz frequency band.
%                 % This method mainly rejects muscle movement (EMG) artifacts.
%                 EEG_copy = pop_rejspec(EEG_copy, 1,'elecrange', 1:EEG_copy.nbchan, 'method', 'fft', 'threshold', emg_thrs ,'freqlimits', emg_freqs_limit, 'eegplotplotallrej', 0, 'eegplotreject', 0);
%                 
%                 % method                : method to compute spectrum (fft)
%                 % threshold             : [lower upper] threshold limit(s) in dB.
%                 % freqlimits            : [lower upper] frequency limit(s) in Hz.
%                 % eegplotplotallrej     : 0 = Do not superpose rejection marks on previous marks stored in the dataset.
%                 % eegplotreject         : 0 = Do not reject marked trials (but store the  marks).
%                 
%                 % Find the number of artifacted epochs and reject them
%                 EEG_copy = eeg_checkset(EEG_copy);
%                 EEG_copy = eeg_rejsuperpose(EEG_copy, 1, 1, 1, 1, 1, 1, 1, 1);
%                 reject_artifacted_epochs=EEG_copy.reject.rejglobal;
%                 
%                 
%                 if nnz(ica_prep_badChans) < 20 && (nnz(reject_artifacted_epochs) < 0.8*length(reject_artifacted_epochs))
%                     EEG_copy = pop_rejepoch(EEG_copy, reject_artifacted_epochs, 0);
%                     
%                     
%                     %% STEP 9: Run ICA
%                     %     length_ica_data(subject)=EEG_copy.trials; % length of data (in second) fed into ICA
%                     
%                     [coeff,score] = pca(reshape(EEG_copy.data,[EEG_copy.nbchan, EEG_copy.pnts*EEG_copy.trials])');
%                     v = var(score,[],1);
%                     npcs = find(cumsum(v / sum(v)) > 0.98);
%                     
%                     EEG_copy = eeg_checkset(EEG_copy);
%                     EEG_copy = pop_runica(EEG_copy, 'icatype', 'runica', 'extended', 1, 'stop', 1e-6, 'interupt','off','pca',npcs(1));
%                     % EEG_copy = pop_runica(EEG_copy, 'icatype', 'runica', 'extended', 1, 'stop', 1e-6, 'interupt','off');
%                     close all
%                     
%                     % Find the ICA weights that would be transferred to the original dataset
%                     ICA_WINV=EEG_copy.icawinv;
%                     ICA_SPHERE=EEG_copy.icasphere;
%                     ICA_WEIGHTS=EEG_copy.icaweights;
%                     ICA_CHANSIND=EEG_copy.icachansind;
%                     
%                     % If channels were removed from copied dataset during preparation of ica, then remove
%                     % those channels from original dataset as well before transferring ica weights.
%                     EEG = eeg_checkset(EEG);
%                     EEG = pop_select(EEG,'nochannel', ica_prep_badChans);
%                     
%                     % Transfer the ICA weights of the copied dataset to the original dataset
%                     EEG.icawinv=ICA_WINV;
%                     EEG.icasphere=ICA_SPHERE;
%                     EEG.icaweights=ICA_WEIGHTS;
%                     EEG.icachansind=ICA_CHANSIND;
%                     EEG = eeg_checkset(EEG);
%                     
%                     %% STEP 10: Run adjust to find artifacted ICA components
%                     %     icafile = sprintf('/data/liuzzil2/UMD_Flanker/derivatives/sub-%s/age-%s/sub-%s_age%s_ICA',sub,agegroup,sub,agegroup);
%                     %     save([icafile,'.mat'], 'EEG'); % save .mat format
%                     %     load([icafile,'.mat'], 'EEG'); % save .mat format
%                     
%                     % badICs=[]; EEG_copy =[];
%                     EEG_copy = EEG;
%                     EEG_copy =eeg_regepochs(EEG_copy,'recurrence', 1, 'limits',[0 1], 'rmbase', [NaN], 'eventtype', '999'); % insert temporary marker 1 second apart and create epochs
%                     EEG_copy = eeg_checkset(EEG_copy);
%                     
%                     icafile = sprintf('/data/liuzzil2/UMD_Flanker/derivatives/sub-%s/age-%s/sub-%s_age%s_adjust_report',sub,agegroup,sub,agegroup);
%                     % if size(EEG_copy.icaweights,1) == size(EEG_copy.icaweights,2)
%                     try
%                         badICs = adjusted_ADJUST(EEG_copy,icafile);
%                         
%                         close all;
%                         % else % if rank is less than the number of electrodes, throw a warning message
%                         %     warning('The rank is less than the number of electrodes. ADJUST will be skipped. Artefacted ICs will have to be manually rejected for this participant');
%                         % end
%                         
%                         % Mark the bad ICs found by ADJUST
%                         for ic=1:length(badICs)
%                             EEG.reject.gcompreject(1, badICs(ic))=1;
%                             EEG = eeg_checkset(EEG);
%                         end
%                         % total_ICs(subject)=size(EEG.icasphere, 1);
%                         % if numel(badICs)==0
%                         %     ICs_removed{subject}='0';
%                         % else
%                         %     ICs_removed{subject}=num2str(double(badICs));
%                         % end
%                         
%                         %% Save dataset after ICA, if saving interim results was preferred
%                         
%                         
%                         if save_interim_result==1
%                             if output_format==1
%                                 EEG = eeg_checkset(EEG);
%                                 EEG = pop_editset(EEG,'setname', [filebids '_ica_data']);
%                                 EEG = pop_saveset(EEG,'filename', [filebids '_ica_data.set'],'filepath', outputfolder ); % save .set format
%                             elseif output_format==2
%                                 save([[output_location filesep 'ica_data' filesep ] strrep(datafile_names{subject}, ext, '_ica_data.mat')], 'EEG'); % save .mat format
%                             end
%                         end
%                         toonoisy =0;
%                     catch
%                         toonoisy =1;
%                     end
%                 else
%                     toonoisy =1;
%                 end
               
            else
                
                EEG = [];
                EEG = pop_loadset([filebids '_ica_data.set'] , [outputfolder]);
                toonoisy = 0;
            end
            
            %% STEP 11: Remove artifacted ICA components from data
            if toonoisy == 0
                all_bad_ICs=0;
                ICs2remove=find(EEG.reject.gcompreject); % find ICs to remove
                
                % If all ICs and bad, save data at this stage and ignore rest of the preprocessing for this subject.
                %     if numel(ICs2remove)==total_ICs(subject)
                %         all_bad_ICs=1;
                %         warning(['No usable data for datafile', datafile_names{subject}]);
                %         if output_format==1
                %             EEG = eeg_checkset(EEG);
                %             EEG = pop_editset(EEG, 'setname',  strrep(datafile_names{subject}, ext, '_no_usable_data_all_bad_ICs'));
                %             EEG = pop_saveset(EEG, 'filename', strrep(datafile_names{subject}, ext, '_no_usable_data_all_bad_ICs.set'),'filepath', [output_location filesep 'processed_data' filesep ]); % save .set format
                %         elseif output_format==2
                %             save([[output_location filesep 'processed_data' filesep ] strrep(datafile_names{subject}, ext, '_no_usable_data_all_bad_ICs.mat')], 'EEG'); % save .mat format
                %         end
                %     else
                EEG = eeg_checkset( EEG );
                EEG = pop_subcomp( EEG, ICs2remove, 0); % remove ICs from dataset
                %     end
                
                % if all_bad_ICs==1
                %     total_epochs_before_artifact_rejection(subject)=0;
                %     total_epochs_after_artifact_rejection(subject)=0;
                %     total_channels_interpolated(subject)=0;
                %
                % end
                
                %% STEP 12: Segment data into fixed length epochs
                if epoch_data==1
                    if task_eeg ==1 % task eeg
                        EEG = eeg_checkset(EEG);
                        EEG = pop_epoch(EEG, task_event_markers, task_epoch_length, 'epochinfo', 'yes');
                    elseif task_eeg==0 % resting eeg
                        if overlap_epoch==1
                            EEG=eeg_regepochs(EEG,'recurrence',(rest_epoch_length/2),'limits',[0 rest_epoch_length], 'rmbase', [NaN], 'eventtype', char(dummy_events));
                            EEG = eeg_checkset(EEG);
                        else
                            EEG=eeg_regepochs(EEG,'recurrence',rest_epoch_length,'limits',[0 rest_epoch_length], 'rmbase', [NaN], 'eventtype', char(dummy_events));
                            EEG = eeg_checkset(EEG);
                        end
                    end
                end
                
                %     total_epochs_before_artifact_rejection(subject)=EEG.trials;
                
                %% STEP 13: Remove baseline
                if remove_baseline==1
                    EEG = eeg_checkset( EEG );
                    EEG = pop_rmbase( EEG, baseline_window);
                end
                
                %% STEP 14: Artifact rejection
                all_bad_epochs=0;
                if voltthres_rejection==1 % check voltage threshold rejection
                    if interp_epoch==1 % check epoch level channel interpolation
                        chans=[]; chansidx=[];chans_labels2=[];
                        chans_labels2=cell(1,EEG.nbchan);
                        for i=1:EEG.nbchan
                            chans_labels2{i}= EEG.chanlocs(i).labels;
                        end
                        [chans,chansidx] = ismember(frontal_channels, chans_labels2);
                        frontal_channels_idx = chansidx(chansidx ~= 0);
                        badChans = zeros(EEG.nbchan, EEG.trials);
                        badepoch=zeros(1, EEG.trials);
                        if isempty(frontal_channels_idx)==1 % check whether there is any frontal channel in dataset to check
                            warning('No frontal channels from the list present in the data. Only epoch interpolation will be performed.');
                        else
                            % find artifaceted epochs by detecting outlier voltage in the specified channels list and remove epoch if artifacted in those channels
                            for ch =1:length(frontal_channels_idx)
                                EEG = pop_eegthresh(EEG,1, frontal_channels_idx(ch), volt_threshold(1), volt_threshold(2), EEG.xmin, EEG.xmax,0,0);
                                EEG = eeg_checkset( EEG );
                                EEG = eeg_rejsuperpose( EEG, 1, 1, 1, 1, 1, 1, 1, 1);
                                badChans(ch,:) = EEG.reject.rejglobal;
                            end
                            for ii=1:size(badChans, 2)
                                badepoch(ii)=sum(badChans(:,ii));
                            end
                            badepoch=logical(badepoch);
                        end
                        
                        % If all epochs are artifacted, save the dataset and ignore rest of the preprocessing for this subject.
                        if sum(badepoch)==EEG.trials || sum(badepoch)+1==EEG.trials
                            %             all_bad_epochs=1;
                            %             warning(['No usable data for datafile', datafile_names{subject}]);
                            %             if output_format==1
                            %                 EEG = eeg_checkset(EEG);
                            %                 EEG = pop_editset(EEG, 'setname',  strrep(datafile_names{subject}, ext, '_no_usable_data_all_bad_epoch'));
                            %                 EEG = pop_saveset(EEG, 'filename', strrep(datafile_names{subject}, ext, '_no_usable_data_all_bad_epoch.set'),'filepath', [output_location filesep 'processed_data' filesep ]); % save .set format
                            %             elseif output_format==2
                            %                 save([[output_location filesep 'processed_data' filesep ] strrep(datafile_names{subject}, ext, '_no_usable_data_all_bad_epochs.mat')], 'EEG'); % save .mat format
                            %             end
                        else
                            EEG = pop_rejepoch( EEG, badepoch, 0);
                            EEG = eeg_checkset(EEG);
                        end
                        
                        
                        
                        if all_bad_epochs==1
                            %                 warning(['No usable data for datafile', datafile_names{subject}]);
                        else
                            % Interpolate artifacted data for all reaming channels
                            badChans = zeros(EEG.nbchan, EEG.trials);
                            % Find artifacted epochs by detecting outlier voltage but don't remove
                            for ch=1:EEG.nbchan
                                EEG = pop_eegthresh(EEG,1, ch, volt_threshold(1), volt_threshold(2), EEG.xmin, EEG.xmax,0,0);
                                EEG = eeg_checkset(EEG);
                                EEG = eeg_rejsuperpose(EEG, 1, 1, 1, 1, 1, 1, 1, 1);
                                badChans(ch,:) = EEG.reject.rejglobal;
                            end
                            tmpData = zeros(EEG.nbchan, EEG.pnts, EEG.trials);
                            for e = 1:EEG.trials
                                % Initialize variables EEGe and EEGe_interp;
                                EEGe = []; EEGe_interp = []; badChanNum = [];
                                % Select only this epoch (e)
                                EEGe = pop_selectevent( EEG, 'epoch', e, 'deleteevents', 'off', 'deleteepochs', 'on', 'invertepochs', 'off');
                                badChanNum = find(badChans(:,e)==1); % find which channels are bad for this epoch
                                EEGe_interp = eeg_interp(EEGe,badChanNum); %interpolate the bad channels for this epoch
                                tmpData(:,:,e) = EEGe_interp.data; % store interpolated data into matrix
                            end
                            EEG.data = tmpData; % now that all of the epochs have been interpolated, write the data back to the main file
                            
                            % If more than 10% of channels in an epoch were interpolated, reject that epoch
                            badepoch=zeros(1, EEG.trials);
                            for ei=1:EEG.trials
                                NumbadChan = badChans(:,ei); % find how many channels are bad in an epoch
                                if sum(NumbadChan) > round((10/100)*EEG.nbchan)% check if more than 10% are bad
                                    badepoch (ei)= sum(NumbadChan);
                                end
                            end
                            badepoch=logical(badepoch);
                        end
                        % If all epochs are artifacted, save the dataset and ignore rest of the preprocessing for this subject.
                        if sum(badepoch)==EEG.trials || sum(badepoch)+1==EEG.trials
                            all_bad_epochs=1;
                            %             warning(['No usable data for datafile', datafile_names{subject}]);
                            %             if output_format==1
                            %                 EEG = eeg_checkset(EEG);
                            %                 EEG = pop_editset(EEG, 'setname',  strrep(datafile_names{subject}, ext, '_no_usable_data_all_bad_epochs'));
                            %                 EEG = pop_saveset(EEG, 'filename', strrep(datafile_names{subject}, ext, '_no_usable_data_all_bad_epochs.set'),'filepath', [output_location filesep 'processed_data' filesep ]); % save .set format
                            %             elseif output_format==2
                            %                 save([[output_location filesep 'processed_data' filesep ] strrep(datafile_names{subject}, ext, '_no_usable_data_all_bad_epochs.mat')], 'EEG'); % save .mat format
                            %             end
                        else
                            EEG = pop_rejepoch(EEG, badepoch, 0);
                            EEG = eeg_checkset(EEG);
                        end
                    else % if no epoch level channel interpolation
                        EEG = pop_eegthresh(EEG, 1, (1:EEG.nbchan), volt_threshold(1), volt_threshold(2), EEG.xmin, EEG.xmax, 0, 0);
                        rejtrials = any(EEG.reject.rejthreshE,1);
                        EEG = eeg_checkset(EEG);
                        EEG = eeg_rejsuperpose( EEG, 1, 1, 1, 1, 1, 1, 1, 1);
                    end % end of epoch level channel interpolation if statement
                    
                    % If all epochs are artifacted, save the dataset and ignore rest of the preprocessing for this subject.
                    if sum(EEG.reject.rejthresh)==EEG.trials || sum(EEG.reject.rejthresh)+1==EEG.trials
                        %         all_bad_epochs=1;
                        %         warning(['No usable data for datafile', datafile_names{subject}]);
                        %         if output_format==1
                        %             EEG = eeg_checkset(EEG);
                        %             EEG = pop_editset(EEG, 'setname',  strrep(datafile_names{subject}, ext, '_no_usable_data_all_bad_epochs'));
                        %             EEG = pop_saveset(EEG, 'filename', strrep(datafile_names{subject}, ext, '_no_usable_data_all_bad_epochs.set'),'filepath', [output_location filesep 'processed_data' filesep ]); % save .set format
                        %         elseif output_format==2
                        %             save([[output_location filesep 'processed_data' filesep ] strrep(datafile_names{subject}, ext, '_no_usable_data_all_bad_epochs.mat')], 'EEG'); % save .mat format
                        %         end
                    else
                        EEG = pop_rejepoch(EEG,(EEG.reject.rejthresh), 0);
                        EEG = eeg_checkset(EEG);
                    end
                end % end of voltage threshold rejection if statement
                
                %     % if all epochs are found bad during artifact rejection
                %     if all_bad_epochs==1
                %         total_epochs_after_artifact_rejection(subject)=0;
                %         total_channels_interpolated(subject)=0;
                %         continue % ignore rest of the processing and go to next datafile
                %     else
                %         total_epochs_after_artifact_rejection(subject)=EEG.trials;
                %     end
                
                %% STEP 15: Interpolate deleted channels
                
                if interp_channels==1
                    EEG = eeg_interp(EEG, channels_analysed);
                    EEG = eeg_checkset(EEG);
                end
                % if numel(FASTbadChans)==0 && numel(ica_prep_badChans)==0
                %     total_channels_interpolated(subject)=0;
                % else
                %     total_channels_interpolated(subject)=numel(FASTbadChans)+ numel(ica_prep_badChans);
                % end
                
                %% STEP 16: Rereference data
                if rerefer_data==1
                    if iscell(reref)==1
                        reref_idx=zeros(1, length(reref));
                        for rr=1:length(reref)
                            reref_idx(rr)=find(strcmp({EEG.chanlocs.labels}, reref{rr}));
                        end
                        EEG = eeg_checkset(EEG);
                        EEG = pop_reref( EEG, reref_idx);
                    else
                        EEG = eeg_checkset(EEG);
                        EEG = pop_reref(EEG, reref);
                    end
                end
                
                %% Save processed data
                if output_format==1
                    EEG = eeg_checkset(EEG);
                    EEG = pop_editset(EEG, 'setname', [filebids '_processed_data_cue']); %_resp
                    EEG = pop_saveset(EEG, 'filename', [filebids '_processed_data_cue.set'] ,'filepath', outputfolder); %_resp, save .set format
                elseif output_format==2
                    save([[output_location filesep 'processed_data' filesep ] strrep(datafile_names{subject}, ext, '_processed_data.mat')], 'EEG'); % save .mat format
                    
                end
                save([outputfolder,filebids '_rejected_trials_cue.mat'],'rejtrials')
%                 save([outputfolder,filebids '_rejected_trials_resp.mat'],'rejtrials')
                
                
                
                
            end
        end
    end
end
%% Create the report table for all the data files with relevant preprocessing outputs.
% report_table=table(datafile_names', reference_used_for_faster', faster_bad_channels', ica_preparation_bad_channels', length_ica_data', ...
%     total_ICs', ICs_removed', total_epochs_before_artifact_rejection', total_epochs_after_artifact_rejection',total_channels_interpolated');
%
% report_table.Properties.VariableNames={'datafile_names', 'reference_used_for_faster', 'faster_bad_channels', ...
%     'ica_preparation_bad_channels', 'length_ica_data', 'total_ICs', 'ICs_removed', 'total_epochs_before_artifact_rejection', ...
%     'total_epochs_after_artifact_rejection', 'total_channels_interpolated'};
% writetable(report_table, ['MADE_preprocessing_report_', datestr(now,'dd-mm-yyyy'),'.csv']);


% Check effectivness of ICA eye removal
% Add get