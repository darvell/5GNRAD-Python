function [results, detStats, info] = run5GNRad(simConfig, stConfig, prsConfig, geometry, sensConfig,backgroundChannel,targetChannel)
% RUN5GNRAD Run PRS-based radar simulation using 5G NR signals
%   RESULTS = RUN5GNRAD(SIMCONFIG, STCONFIG, PRSCONFIG, GEOMETRY,
%   SENSCONFIG, BACKGROUNDCHANNEL) simulates a monostatic
%   radar system based on 5G NR Positioning Reference Signals (PRS). It
%   computes a range-Doppler map and estimates the position and velocity of
%   the target across a number of sensing snapshots.
%
%   Inputs:
%     SIMCONFIG          - Structure with system-level configuration (e.g., fc, bandwidth, noise figure)
%     STCONFIG           - Structure with true target states:
%                          * position: [N x 3] target positions
%                          * velocity: [N x 3] target velocities
%     PRSCONFIG          - PRS configuration structure for nrPRSConfig
%     GEOMETRY           - Structure with TX and RX geometry (fields: tx, rx)
%     SENSCONFIG         - Sensing parameters (e.g., window type, Doppler FFT size)
%     BACKGROUNDCHANNEL  - Cell array or vector with background channel impulse responses per snapshot
%
%   Output:
%     RESULTS - Structure containing per-snapshot radar performance metrics:
%         * positionError        - Euclidean error in 3D position estimate
%         * rangeError           - Error in estimated range
%         * velocityError        - Error in estimated radial velocity
%         * azimuthError         - AOA azimuth error
%         * elevationError       - AOA elevation error
%         * dopplerPeakPower     - Peak value of Doppler map
%         * dopplerpeakToAverage - Peak-to-average power ratio in Doppler domain
%         * isLos                - Boolean LOS flag from channel model
%
%   The simulation includes:
%     - PRS generation and OFDM mapping
%     - CDL-based sensing channel modeling
%     - Range-Doppler processing and geometry estimation
%
%   Example:
%     results = RUN5GNRAD(simCfg, targetCfg, prsCfg, geometry, sensCfg, bgChannel);

%   2025 NIST/CTL Steve Blandino

%   This file is available under the terms of the NIST License.


%% CONSTANTS
c = 299702547;
subcarrierRB = 12;
%maxPowerFcc_100MHz = 75; % https://www.ecfr.gov/current/title-47/chapter-I/subchapter-B/part-30/subpart-C
k = 1.380649*1e-23;
T = 297;

%% SYSTEM PARAMS
fc = simConfig.systemFc; % Carrier frequency
numberSensingSymbols = sensConfig.numberSensingSymbols;
dopplerFftLen = sensConfig.dopplerFftLen;

% Carrier
carrier = nrCarrierConfig;
carrier.SubcarrierSpacing = simConfig.carrierSubcarrierSpacing;
carrier.NSizeGrid = simConfig.carrierNSizeGrid;

%% Configure PRS
prs = nrPRSConfig;
prs.PRSResourceSetPeriod = prsConfig.PRSResourceSetPeriod;
prs.PRSResourceOffset = prsConfig.PRSResourceOffset;
prs.PRSResourceRepetition = prsConfig.PRSResourceRepetition;
prs.PRSResourceTimeGap = prsConfig.PRSResourceTimeGap;
prs.NumRB = prsConfig.NumRB;
prs.RBOffset = prsConfig.RBOffset;
prs.CombSize = prsConfig.CombSize;
prs.REOffset = prsConfig.REOffset;
prs.NPRSID = prsConfig.NPRSID;
prs.NumPRSSymbols = prsConfig.NumPRSSymbols;
prs.SymbolStart = prsConfig.SymbolStart;
prs.NumRB = carrier.NSizeGrid;

% Get the number of orthogonal frequency division multiplexing (OFDM) symbols per slot.
numSymPerSlot = carrier.SymbolsPerSlot;
numSlots = numberSensingSymbols*prs(1).PRSResourceSetPeriod(1);

%% OFDM PARAMS
ofdmInfo = nrOFDMInfo(carrier);
ofdmSymbolTime = ofdmInfo.SymbolLengths / ofdmInfo.SampleRate;  % Time in seconds
ofdmTs = mean(ofdmSymbolTime);
prsPeriodicity =  sum(ofdmSymbolTime)/carrier.SlotsPerSubframe*prs.PRSResourceSetPeriod(1);

%% DEPENDENT PARAMS
numberSubcarriers = carrier.NSizeGrid * subcarrierRB; % Total subcarriers

% Range resolution
prsRangeResolution = 1/(2*ofdmInfo.SampleRate)*c;
rangeFFTLen = ofdmInfo.Nfft;
rangeWindow = getDftWindow('hamming', numberSubcarriers);
rangeBinDestgrd = (0:rangeFFTLen*prs.CombSize-1) * prsRangeResolution * 1/prs.CombSize;

% Velocity Resolution
vosf = dopplerFftLen/numberSensingSymbols;
prsVelocityResolution = c / (2*numberSensingSymbols*prsPeriodicity*fc)/vosf;

% Velocity Bin
velocityBin = (-dopplerFftLen/2:dopplerFftLen/2-1)*prsVelocityResolution;

lambda = c/fc;

%% Antenna params
nAntH = simConfig.antennaNumH;
nAntV = simConfig.antennaNumV;
numElm = nAntH*nAntV;
transmitArray = phased.URA([nAntV nAntH], 'ElementSpacing',lambda/2);
receiveArray =  phased.URA([nAntV nAntH], 'ElementSpacing',lambda/2);
hpbwH = 0.886*2/nAntH*180/pi;
hpbwV = 0.886*2/nAntV*180/pi;
scanStepH = floor(hpbwH);
scanStepV = floor(hpbwV);
azimuthRange = -180:scanStepH:180;
elevationRange = -90:scanStepV:90;
[az, el] = meshgrid(azimuthRange, elevationRange);
scanVector = [az(:), el(:)];

% %% Parameters
% c = physconst('LightSpeed');
% fc = 3.5e9;                     % carrier
% lambda = c/fc;
%
% % 3GPP tuple: (M,N,P,Mg,Ng; Mp,Np)
% M  = 8; N  = 8; P  = 2;         % elements per panel: rows, cols, pols
% Mg = 1; Ng = 1;                 % panels grid (rows, cols)
% Mp = 4; Np = 8;                 % TXRU grid (rows, cols)  -> 32 TXRUs
%
% % Check divisibility (each TXRU controls an integer block)
% assert(mod(M,Mp)==0 && mod(N,Np)==0, 'M/Mp or N/Np must be integer.');
% blkR = M/Mp;                    % rows per TXRU block (here 2)
% blkC = N/Np;                    % cols per TXRU block (here 1)
%
% %% Dual-pol element models (±45°)
% elNeg = phased.NRAntennaElement('PolarizationModel',2,'PolarizationAngle',-45);
% elPos = phased.NRAntennaElement('PolarizationModel',2,'PolarizationAngle', 45);
%
% %% Physical array: NR rectangular panel
% arr = phased.NRRectangularPanelArray( ...
%     'ElementSet', {elNeg, elPos}, ...      % P=2
%     'Size',       [M N Mg Ng],     ...     % [element rows, element cols, panel rows, panel cols]
%     'Spacing',    [0.5*lambda 0.5*lambda 3*lambda 3*lambda] ); % elem/panel spacings (example)

%% Power
% maxPowerFcc = maxPowerFcc_100MHz - 10*log10(100e6) + 10*log10(simConfig.systemBw);
%maxPower = maxPowerFcc - (10*log10(nAntV*nAntH) + 10*log10(simConfig.antennaCouplingEfficiency));
maxPower = 52-30+8; %dBm
%% SNR DEFINITION
NF = 10^(simConfig.systemNF/10);
N = k*T*simConfig.systemBw*NF;
P = 10^((maxPower)/10);
SNR = 10*log10(P/N);

%% CHANNEL MODEL PARAMETERS
% Generate random channel gains, delays, and Doppler shifts for L objects
txPos = geometry.tx;
rxPos = geometry.rx;
scenario = simConfig.channelScenario;
stPosition = stConfig.position;

%% CHANNEL MODEL DEPENDENT PARAMETERS
velocityVectors = stConfig.velocity;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End Config
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Transmission: Generate OFDM grid
% Extract OFDM parameters
ofdmInfo    = nrOFDMInfo(carrier);
ofdmFftLen  = ofdmInfo.Nfft;
cpLengths   = ofdmInfo.CyclicPrefixLengths;  % length for each symbol in a slot

% Map the resource elements (RE) for both of the PRS resources on the carrier resource grid.
% Determine the start and end indices for the active subcarriers
startIdx = (ofdmFftLen - numberSubcarriers) / 2 + 1;
endIdx = startIdx + numberSubcarriers - 1;

maxElementsPerSlot = ofdmFftLen * numSymPerSlot;

totalMaxElements = numSlots * maxElementsPerSlot;

Iall = zeros(totalMaxElements, 1);
Jall = zeros(totalMaxElements, 1);
Vall = zeros(totalMaxElements, 1);
ptr = 1;

for slotIdx = 0:prs.PRSResourceSetPeriod(1):numSlots-1
    carrier.NSlot = slotIdx;
    indCell = nrPRSIndices(carrier,prs,'OutputResourceFormat','cell');
    symCell = nrPRS(carrier,prs,'OutputResourceFormat','cell');
    idx = indCell{1};
    val = symCell{1};

    [rowIdx, colIdx] = ind2sub([numberSubcarriers, numSymPerSlot], idx);
    rowIdx = rowIdx+startIdx-1;
    colIdx = colIdx + slotIdx * numSymPerSlot;

    rangeIdx = ptr : ptr + numberSubcarriers - 1;
    Iall(rangeIdx) = rowIdx;
    Jall(rangeIdx) = colIdx;
    Vall(rangeIdx) = val;

    ptr = ptr + numel(val);
end

Iall(ptr:end) = [];
Jall(ptr:end) = [];
Vall(ptr:end) = [];

% sparse assembly
ofdmGrid = sparse(Iall, Jall, Vall, ofdmFftLen, carrier.SymbolsPerSlot*numSlots);

% Get the column indices of non-zero elements in ofdmGrid
[~, colIdx] = find(ofdmGrid);

% Determine the slot indices by dividing column indices by numSymPerSlot
symbolIndices = unique(colIdx);

%% Transmission: generate TX waveform
txWaveform = zeros((cpLengths(2)+ofdmFftLen)*prs.NumPRSSymbols*numberSensingSymbols, 1);
for symIdx = 0:prs.NumPRSSymbols*numberSensingSymbols-1

    % Frequency-domain samples for this symbol
    freqDomainSymbol = sqrt(prs.CombSize)*ofdmGrid(:, symbolIndices(symIdx+1));

    timeDomainNoCP = ifft(full(freqDomainSymbol), ofdmFftLen) * sqrt(ofdmFftLen);

    % Determine CP length for symbol symIdx within the slot:
    cpLen = cpLengths(mod(symbolIndices(symIdx+1)-1, length(cpLengths)) + 1);

    % Extract the CP samples from the end of the symbol
    cyclicPrefix = timeDomainNoCP(end - cpLen + 1 : end);

    % Concatenate CP + time-domain symbol
    timeDomainWithCP = [cyclicPrefix; timeDomainNoCP];

    % Append to the full TX waveform
    txWaveform((cpLengths(2)+ofdmFftLen)*symIdx+1: (cpLengths(2)+ofdmFftLen)*(symIdx+1) ) = timeDomainWithCP;
end

Nst = size(stPosition,1);
Ndrop = floor(Nst/simConfig.nStDrop);
velocityError = cell(1,Ndrop);
rangeError = cell(1,Ndrop);
snrvarVec = cell(1,Nst);
timeIndex = cell(1,Ndrop);
azimuthError = cell(1,Ndrop);
elevationError= cell(1,Ndrop);
positionErrorX= cell(1,Ndrop);
positionErrorY= cell(1,Ndrop);
positionErrorZ= cell(1,Ndrop);
positionErrorV= cell(1,Ndrop);
positionErrorH= cell(1,Ndrop);
truePositive = zeros(Ndrop,1);
falseNegative = zeros(Ndrop,1);
falsePositve = zeros(Ndrop,1);
falseAlarmProb = zeros(Ndrop,1);
isLos= zeros(1,Nst);
tStart = tic;
index = setStMinDistanceConstraint(stPosition, simConfig.nStDrop, 10);
stPosition = stPosition(index,:,:);
velocityVectors = velocityVectors(index,:,:);
for q = 1:Ndrop
    %% Realize Channel
    targetIdx = (q-1)*simConfig.nStDrop+1:q*simConfig.nStDrop;

    baseArgs = { ...
        txPos, stPosition(targetIdx,:), velocityVectors(targetIdx,:), fc, ofdmTs};

    opts = struct( ...
        'bandwidth', ofdmInfo.SampleRate, ...
        'nRealization', symbolIndices(end), ...
        'transmitArray', transmitArray, ...
        'receiveArray', receiveArray, ...
        'scanvector', scanVector, ...
        'scenario', scenario, ...
        'backgroundChannel', backgroundChannel(q),...
        'angleEstimation', sensConfig.angleEstimationMethod);

    if ~isempty(targetChannel)
        opts.targetChannel = targetChannel(targetIdx);
    end

    args = [baseArgs, namedargs2cell(opts)];

    % Call function with dynamically built args
    [~,H,Hfull, tgtPwr,syncOffset] = getSensingCdl(args{:});

    %% Through the channel
    numSamplesTotal = length(txWaveform);

    snrvar = 10^(SNR/10);
    samplePointer = 0;
    waveformLen = numSamplesTotal + size(H,1) - 1;
    rxWaveform = zeros(waveformLen, numElm);

    for symIdx = 0:prs.NumPRSSymbols*numberSensingSymbols-1

        % Symbol block in TX waveform
        cpLen = cpLengths(mod(symbolIndices(symIdx+1)-1, length(cpLengths)) + 1);
        symbolLen = ofdmFftLen + cpLen;

        % Extract samples for this symbol from txWaveform
        thisSymbolTx = txWaveform(samplePointer + 1 : samplePointer + symbolLen);
        for nrx = 1:numElm
            hSym = Hfull(:,symbolIndices(symIdx+1),nrx);

            % Convolution
            thisSymbolRx = conv(thisSymbolTx, hSym);

            % Add the received symbol to the overall rxWaveform in the correct place
            rxWaveform(samplePointer + 1 : samplePointer + length(thisSymbolRx),nrx) = ...
                rxWaveform(samplePointer + 1 : samplePointer + length(thisSymbolRx),nrx) + thisSymbolRx;
        end
        % Update pointer
        samplePointer = samplePointer + symbolLen;
    end

    rxWaveform  = rxWaveform+(sqrt(1/(2*snrvar)) * (randn(waveformLen, numElm) + 1j *randn(waveformLen, numElm)));
    snrvarVec{q} =  10*log10(sum(reshape(snrvar*10.^(tgtPwr/10), [], simConfig.nStDrop),1))';

    %% Receiver Processing: retrieve OFDM grid
    rows = (1:ofdmFftLen).';
    samplePointer = 0;
    freqDomainSymbolRxStore =  zeros(prs.NumPRSSymbols*numberSensingSymbols*ofdmFftLen, 1);
    linearIdxStore = zeros(prs.NumPRSSymbols*numberSensingSymbols*ofdmFftLen, 1);
    for symIdx = 0:prs.NumPRSSymbols*numberSensingSymbols-1

        cpLen = cpLengths(mod(symbolIndices(symIdx+1)-1, length(cpLengths)) + 1);
        symbolLen = ofdmFftLen + cpLen;
        for nrx = 1:numElm

            % Extract the samples for this symbol
            thisSymbolRx = rxWaveform(samplePointer + 1 : samplePointer + symbolLen,nrx);

            % Remove CP
            thisSymbolRxNoCP = thisSymbolRx(cpLen + 1 : end);

            % Take FFT (including the sqrt(NFFT) normalization)
            freqDomainSymbolRx = (1 / sqrt(ofdmFftLen)) * fft(thisSymbolRxNoCP, ofdmFftLen);

            % Store into the received OFDM grid
            linearIdx = rows + (symbolIndices(symIdx+1) - 1) * ofdmFftLen;
            freqDomainSymbolRxStore(symIdx*ofdmFftLen+1: (symIdx+1)*ofdmFftLen,nrx ) = freqDomainSymbolRx;
            linearIdxStore(symIdx*ofdmFftLen+1: (symIdx+1)*ofdmFftLen,nrx ) = linearIdx;
        end
        % Advance pointer
        samplePointer = samplePointer + symbolLen;
    end

    %% Receiver Processing: Channel estimate
    g_tilde_kn = zeros(ofdmFftLen,carrier.SymbolsPerSlot * numSlots,numElm);
    for nrx = 1:numElm
        [rowIdx, colIdx] = ind2sub([ofdmFftLen, carrier.SymbolsPerSlot * numSlots], linearIdxStore(:,nrx));
        rxOfdmGrid = sparse(rowIdx, colIdx, freqDomainSymbolRxStore(:,nrx), ofdmFftLen,  carrier.SymbolsPerSlot * numSlots);
        g_tilde_kn(:,:,nrx) = full(rxOfdmGrid.*conj(ofdmGrid));
    end

    %% Receiver Processing: Sensing processing
    % Range processing
    rangeFft = zeros(rangeFFTLen*prs.CombSize,numberSensingSymbols,numElm);

    % Loop over PRS resource symbols in each slot
    for slotIdx = 0:prs.PRSResourceSetPeriod(1):numSlots-1
        carrier.NSlot = slotIdx;
        indCell = nrPRSIndices(carrier,prs,'OutputResourceFormat','cell');
        symIndSlot = symbolIndices(symbolIndices < (slotIdx+1)*ofdmInfo.SymbolsPerSlot &...
            symbolIndices > (slotIdx)*ofdmInfo.SymbolsPerSlot);

        for nrx = 1:numElm
            slotGridRx =  g_tilde_kn(startIdx:endIdx,(1:numSymPerSlot)+numSymPerSlot*slotIdx,nrx);
            symCellRx = full(slotGridRx(indCell{1}));
            prsSlotGridRx = reshape(symCellRx, prs.NumRB*12/prs.CombSize, []);
            prsSlotGridRxDestagrd = prsDestaggering(prsSlotGridRx, prs,mod(symIndSlot-1, ofdmInfo.SymbolsPerSlot));
            rangeFft(:, ceil(slotIdx/prs.PRSResourceSetPeriod(1))+1,nrx) = ...
                sqrt(rangeFFTLen*prs.CombSize)*ifft(prsSlotGridRxDestagrd.*rangeWindow, rangeFFTLen*prs.CombSize);
        end
    end

    rangeFft = rangeFft - mean(rangeFft,2);
    rangeLength = prs.CombSize*ofdmFftLen;
    rangeLength = min(rangeLength,find(rangeBinDestgrd>simConfig.maxRangeInterest, 1,'first'));
    rangeFft(rangeLength+1:end,:,:) = [];

    % Doppler per antenna RD cube
    rdCube = complex(zeros(rangeLength,dopplerFftLen, numElm)); 
   
    for m = 1:numElm
         rdCube(:,:,m) = fftshift(1/sqrt(dopplerFftLen) *fft(rangeFft(:,:,m), dopplerFftLen,2),2);
    end

    idxVec = getAntennaSortedIndex(receiveArray);
    rdCube = rdCube(:,:,idxVec);
    Nfft_h = sensConfig.azFftLen;
    Nfft_v = sensConfig.elFftLen;

    Xcube = reshape(rdCube, rangeLength, dopplerFftLen, nAntV, nAntH);
    RDA = fftshift(fftshift(fft(fft(Xcube, Nfft_v, 3), Nfft_h,4), 3 ),4);

    [azGrid, elGrid] = getArrayAngleGrid(Nfft_v, Nfft_h, ...
        receiveArray.ElementSpacing(1), receiveArray.ElementSpacing(2), lambda);

    % Detection
    out = detectCfar4D(RDA, sensConfig);

    % Parameter estimation
    nDetectedTarget = size(out,1);
    azEstimate_k = zeros(nDetectedTarget,1);
    elEstimate_k = zeros(nDetectedTarget,1);
    for det = 1:nDetectedTarget
        azEstimate_k(det,1) = azGrid(out(det,3),out(det,4));
        elEstimate_k(det,1) = elGrid(out(det,3),out(det,4));
    end
    rngEstimate_k = rangeBinDestgrd(out(:,1))+syncOffset;
    velParallel_k = velocityBin(out(:,2));
    gtVel = sum((-txPos+stPosition(targetIdx,:))./...
        vecnorm(txPos-stPosition(targetIdx,:), 2, 2).* (velocityVectors(targetIdx,:)),2);

    stPositionEstimate_k = estimateScatteringGeometry(txPos ,rxPos, ...
        2*rngEstimate_k(:)/c*1e9, [azEstimate_k, elEstimate_k], 'keepSectorAz', [-60 60]);

    validEstimateIndex = all(~isnan(stPositionEstimate_k),2);
    stPositionEstimate_k = stPositionEstimate_k(validEstimateIndex,:);
    velParallel_k = velParallel_k(validEstimateIndex);

    % Metrics
    metrics =scoreAssociationsPos(stPosition(targetIdx,:), stPositionEstimate_k, ...
        gtVel,velParallel_k,txPos);
    timeIndex{q} = repmat(q,simConfig.nStDrop,1);
    positionErrorX{q} = metrics.stats.pos.errXYZ(:,1);
    positionErrorY{q} = metrics.stats.pos.errXYZ(:,2);
    positionErrorZ{q} = metrics.stats.pos.errXYZ(:,3);
    positionErrorH{q} = vecnorm(metrics.stats.pos.errXYZ(:,1:2),2,2);
    positionErrorV{q} = metrics.stats.pos.errXYZ(:,3);
    rangeError{q} = metrics.stats.pos.range_err;
    velocityError{q} = metrics.stats.vel.vr_err;
    azimuthError{q} = metrics.stats.pos.az_err_deg;
    elevationError{q} = metrics.stats.pos.el_err_deg;
    truePositive(q,1) = metrics.TP;
    falseNegative(q,1) = metrics.FN;
    falsePositve(q,1) = metrics.FP;
    falseAlarmProb(q,1) = metrics.FPR;
    percent = q / Nst * 100;
    elapsed = toc(tStart);
    est_total = elapsed / q * Nst;
    eta = est_total - elapsed;

    % % Format ETA as duration
    etaDur = duration(0, 0, eta);  
    fprintf('\rProgress: %5.1f%% | ETA: %s', percent, char(etaDur));
end

results.timeIndex = vertcat(timeIndex{:});
results.positionErrorX = vertcat(positionErrorX{:});
results.positionErrorY = vertcat(positionErrorY{:});
results.positionErrorZ = vertcat(positionErrorZ{:});

results.rangeError = vertcat(rangeError{:});
results.velocityError = vertcat(velocityError{:});
results.azimuthError = vertcat(azimuthError{:});
results.elevationError = vertcat(elevationError{:});
results.positionErrorH = vertcat(positionErrorH{:});
results.positionErrorV = vertcat(positionErrorV{:});
detStats.truePositive = truePositive;
detStats.falseNegative = falseNegative;
detStats.falsePositve = falsePositve;
detStats.falseAlarmProb = falseAlarmProb;
results.snr = vertcat(snrvarVec{:});
info.K = sum(truePositive>=1); %number of drops with at least one detected object
info.cpi = prsPeriodicity*numberSensingSymbols;
info.isLos = isLos;

end