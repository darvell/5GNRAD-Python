function export_matlab_fixtures(scenarioPath)
% EXPORT_MATLAB_FIXTURES Generate validation fixtures for Python parity tests
%
%   export_matlab_fixtures(scenarioPath) generates CSV and MAT files in the
%   Output/ folder that the Python test suite uses to validate correctness.
%
%   Inputs:
%       scenarioPath - Path to scenario folder (e.g., 'examples/UMi-Av25')
%
%   Outputs (saved to scenarioPath/Output/):
%       matlab_prs_indices.csv  - Linear indices of PRS symbols (1-indexed)
%       matlab_prs_symbols.csv  - Complex PRS values (real,imag columns)
%       matlab_rd.mat           - Range-Doppler map (variable: rd)
%       matlab_rda.mat          - Range-Doppler-Angle cube (variable: rda)
%
%   Example:
%       export_matlab_fixtures('examples/UMi-Av25')
%       export_matlab_fixtures('examples3GPP/UMa-Av200-8x8-30')
%
%   2025 NIST/CTL - Python validation fixture generator

%% Load scenario configuration from .txt files with DEFAULTS
fprintf('Loading scenario: %s\n', scenarioPath);

inputDir = fullfile(scenarioPath, 'Input');

% Load with defaults
simConfig = loadSimConfig(fullfile(inputDir, 'simulationConfig.txt'));
prsConfigRaw = loadPrsConfig(fullfile(inputDir, 'prsConfig.txt'));
sensConfig = loadSensConfig(fullfile(inputDir, 'sensConfig.txt'));

%% Constants
c = 299792458.0;  % Speed of light (exact value for consistency with Python)
subcarrierRB = 12;

%% Configure carrier
carrier = nrCarrierConfig;
carrier.SubcarrierSpacing = simConfig.carrierSubcarrierSpacing;
carrier.NSizeGrid = simConfig.carrierNSizeGrid;

%% Configure PRS
prs = nrPRSConfig;
prs.PRSResourceSetPeriod = prsConfigRaw.PRSResourceSetPeriod;
prs.PRSResourceOffset = prsConfigRaw.PRSResourceOffset;
prs.PRSResourceRepetition = prsConfigRaw.PRSResourceRepetition;
prs.PRSResourceTimeGap = prsConfigRaw.PRSResourceTimeGap;
prs.NumRB = prsConfigRaw.NumRB;
prs.RBOffset = prsConfigRaw.RBOffset;
prs.CombSize = prsConfigRaw.CombSize;
prs.REOffset = prsConfigRaw.REOffset;
prs.NPRSID = prsConfigRaw.NPRSID;
prs.NumPRSSymbols = prsConfigRaw.NumPRSSymbols;
prs.SymbolStart = prsConfigRaw.SymbolStart;
prs.NumRB = carrier.NSizeGrid;  % Override with carrier grid size

%% OFDM parameters
ofdmInfo = nrOFDMInfo(carrier);
ofdmFftLen = ofdmInfo.Nfft;
cpLengths = ofdmInfo.CyclicPrefixLengths;

numberSubcarriers = carrier.NSizeGrid * subcarrierRB;
numSymPerSlot = carrier.SymbolsPerSlot;
numberSensingSymbols = sensConfig.numberSensingSymbols;
numSlots = numberSensingSymbols * prs.PRSResourceSetPeriod(1);

%% Generate PRS grid (same as run5GNRad.m lines 186-211)
startIdx = (ofdmFftLen - numberSubcarriers) / 2 + 1;

maxElementsPerSlot = ofdmFftLen * numSymPerSlot;
totalMaxElements = numSlots * maxElementsPerSlot;

Iall = zeros(totalMaxElements, 1);
Jall = zeros(totalMaxElements, 1);
Vall = zeros(totalMaxElements, 1);
ptr = 1;

for slotIdx = 0:prs.PRSResourceSetPeriod(1):numSlots-1
    carrier.NSlot = slotIdx;
    indCell = nrPRSIndices(carrier, prs, 'OutputResourceFormat', 'cell');
    symCell = nrPRS(carrier, prs, 'OutputResourceFormat', 'cell');
    idx = indCell{1};
    val = symCell{1};

    [rowIdx, colIdx] = ind2sub([numberSubcarriers, numSymPerSlot], idx);
    rowIdx = rowIdx + startIdx - 1;
    colIdx = colIdx + slotIdx * numSymPerSlot;

    rangeIdx = ptr : ptr + numel(val) - 1;
    Iall(rangeIdx) = rowIdx;
    Jall(rangeIdx) = colIdx;
    Vall(rangeIdx) = val;

    ptr = ptr + numel(val);
end

Iall(ptr:end) = [];
Jall(ptr:end) = [];
Vall(ptr:end) = [];

% Create sparse OFDM grid
ofdmGrid = sparse(Iall, Jall, Vall, ofdmFftLen, carrier.SymbolsPerSlot * numSlots);

%% Export PRS indices and symbols
% Get linear indices (1-indexed, column-major order as MATLAB uses)
[rowIdx, colIdx, vals] = find(ofdmGrid);
linearIndices = rowIdx + (colIdx - 1) * ofdmFftLen;

% Create output directory
outDir = fullfile(scenarioPath, 'Output');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

% Save PRS indices
fprintf('Saving PRS indices to: %s\n', fullfile(outDir, 'matlab_prs_indices.csv'));
writematrix(linearIndices, fullfile(outDir, 'matlab_prs_indices.csv'));

% Save PRS symbols (real,imag format)
fprintf('Saving PRS symbols to: %s\n', fullfile(outDir, 'matlab_prs_symbols.csv'));
symbolMatrix = [real(vals), imag(vals)];
writematrix(symbolMatrix, fullfile(outDir, 'matlab_prs_symbols.csv'));

%% Generate Range-Doppler map (no channel, for deterministic comparison)
fprintf('Generating Range-Doppler map...\n');

% Get symbol indices
[~, colIdx] = find(ofdmGrid);
symbolIndices = unique(colIdx);

% Calculate total TX waveform length (variable CP lengths)
totalSamples = 0;
for symIdx = 0:prs.NumPRSSymbols * numberSensingSymbols - 1
    cpLen = cpLengths(mod(symbolIndices(symIdx + 1) - 1, length(cpLengths)) + 1);
    totalSamples = totalSamples + ofdmFftLen + cpLen;
end

% Generate TX waveform with proper indexing
txWaveform = zeros(totalSamples, 1);
samplePtr = 0;
for symIdx = 0:prs.NumPRSSymbols * numberSensingSymbols - 1
    freqDomainSymbol = sqrt(prs.CombSize) * ofdmGrid(:, symbolIndices(symIdx + 1));
    timeDomainNoCP = ifft(full(freqDomainSymbol), ofdmFftLen) * sqrt(ofdmFftLen);
    cpLen = cpLengths(mod(symbolIndices(symIdx + 1) - 1, length(cpLengths)) + 1);
    cyclicPrefix = timeDomainNoCP(end - cpLen + 1 : end);
    timeDomainWithCP = [cyclicPrefix; timeDomainNoCP];
    symbolLen = length(timeDomainWithCP);
    txWaveform(samplePtr + 1 : samplePtr + symbolLen) = timeDomainWithCP;
    samplePtr = samplePtr + symbolLen;
end

% For no-channel test: rxWaveform = txWaveform (perfect channel)
rxWaveform = txWaveform;

% Receiver processing
rows = (1:ofdmFftLen).';
samplePointer = 0;
freqDomainSymbolRxStore = zeros(prs.NumPRSSymbols * numberSensingSymbols * ofdmFftLen, 1);
linearIdxStore = zeros(prs.NumPRSSymbols * numberSensingSymbols * ofdmFftLen, 1);

for symIdx = 0:prs.NumPRSSymbols * numberSensingSymbols - 1
    cpLen = cpLengths(mod(symbolIndices(symIdx + 1) - 1, length(cpLengths)) + 1);
    symbolLen = ofdmFftLen + cpLen;

    thisSymbolRx = rxWaveform(samplePointer + 1 : samplePointer + symbolLen);
    thisSymbolRxNoCP = thisSymbolRx(cpLen + 1 : end);
    freqDomainSymbolRx = (1 / sqrt(ofdmFftLen)) * fft(thisSymbolRxNoCP, ofdmFftLen);

    linearIdx = rows + (symbolIndices(symIdx + 1) - 1) * ofdmFftLen;
    freqDomainSymbolRxStore(symIdx * ofdmFftLen + 1 : (symIdx + 1) * ofdmFftLen) = freqDomainSymbolRx;
    linearIdxStore(symIdx * ofdmFftLen + 1 : (symIdx + 1) * ofdmFftLen) = linearIdx;

    samplePointer = samplePointer + symbolLen;
end

% Channel estimation (matched filter)
[rowIdx, colIdx] = ind2sub([ofdmFftLen, carrier.SymbolsPerSlot * numSlots], linearIdxStore);
rxOfdmGrid = sparse(rowIdx, colIdx, freqDomainSymbolRxStore, ofdmFftLen, carrier.SymbolsPerSlot * numSlots);
g_tilde_kn = full(rxOfdmGrid .* conj(ofdmGrid));

% Range processing
rangeFFTLen = ofdmInfo.Nfft;
rangeWindow = getDftWindow('hamming', numberSubcarriers);
dopplerFftLen = sensConfig.dopplerFftLen;

rangeFft = zeros(rangeFFTLen * prs.CombSize, numberSensingSymbols);

for slotIdx = 0:prs.PRSResourceSetPeriod(1):numSlots - 1
    carrier.NSlot = slotIdx;
    indCell = nrPRSIndices(carrier, prs, 'OutputResourceFormat', 'cell');
    symIndSlot = symbolIndices(symbolIndices < (slotIdx + 1) * ofdmInfo.SymbolsPerSlot & ...
        symbolIndices > slotIdx * ofdmInfo.SymbolsPerSlot);

    slotGridRx = g_tilde_kn(startIdx:startIdx + numberSubcarriers - 1, (1:numSymPerSlot) + numSymPerSlot * slotIdx);
    symCellRx = full(slotGridRx(indCell{1}));
    prsSlotGridRx = reshape(symCellRx, prs.NumRB * 12 / prs.CombSize, []);
    prsSlotGridRxDestagrd = prsDestaggering(prsSlotGridRx, prs, mod(symIndSlot - 1, ofdmInfo.SymbolsPerSlot));
    rangeFft(:, ceil(slotIdx / prs.PRSResourceSetPeriod(1)) + 1) = ...
        sqrt(rangeFFTLen * prs.CombSize) * ifft(prsSlotGridRxDestagrd .* rangeWindow, rangeFFTLen * prs.CombSize);
end

% Mean subtraction (clutter removal)
rangeFft = rangeFft - mean(rangeFft, 2);

% Truncate to max range of interest
prsRangeResolution = 1 / (2 * ofdmInfo.SampleRate) * c;
rangeBinDestgrd = (0:rangeFFTLen * prs.CombSize - 1) * prsRangeResolution * 1 / prs.CombSize;
rangeLength = prs.CombSize * ofdmFftLen;
if ~isempty(simConfig.maxRangeInterest) && simConfig.maxRangeInterest > 0
    idx = find(rangeBinDestgrd > simConfig.maxRangeInterest, 1, 'first');
    if ~isempty(idx)
        rangeLength = min(rangeLength, idx);
    end
end
rangeFft(rangeLength + 1:end, :) = [];

% Doppler FFT -> Range-Doppler map
rd = fftshift(1 / sqrt(dopplerFftLen) * fft(rangeFft, dopplerFftLen, 2), 2);

%% Save Range-Doppler map
fprintf('Saving Range-Doppler map to: %s\n', fullfile(outDir, 'matlab_rd.mat'));
save(fullfile(outDir, 'matlab_rd.mat'), 'rd');

%% Generate RDA cube (if multi-antenna)
nAntH = simConfig.antennaNumH;
nAntV = simConfig.antennaNumV;

if nAntH > 1 || nAntV > 1
    fprintf('Multi-antenna config detected (%dx%d)\n', nAntV, nAntH);
    fprintf('NOTE: RDA requires full channel simulation - saving placeholder\n');

    % Get FFT lengths from config
    Nfft_h = sensConfig.azFftLen;
    Nfft_v = sensConfig.elFftLen;

    % Create placeholder (single antenna in first position)
    rda = zeros(size(rd, 1), size(rd, 2), Nfft_v, Nfft_h);
    rda(:, :, 1, 1) = rd;
    fprintf('Saving RDA cube to: %s\n', fullfile(outDir, 'matlab_rda.mat'));
    save(fullfile(outDir, 'matlab_rda.mat'), 'rda');
else
    fprintf('Single antenna config - skipping RDA cube\n');
end

fprintf('\n=== Export Complete ===\n');
fprintf('Files saved to: %s\n', outDir);
fprintf('  - matlab_prs_indices.csv\n');
fprintf('  - matlab_prs_symbols.csv\n');
fprintf('  - matlab_rd.mat\n');
if nAntH > 1 || nAntV > 1
    fprintf('  - matlab_rda.mat\n');
end

end

%% Load simulation config with defaults (matches Python SimulationConfig)
function config = loadSimConfig(filepath)
    % Defaults from Python src/gnrad5/config/models.py
    config = struct();
    config.systemFc = 30e9;
    config.systemNF = 7.0;
    config.systemBw = 100e6;
    config.channelScenario = 'UMiAV';
    config.antennaNumH = 32;
    config.antennaNumV = 32;
    config.antennaCouplingEfficiency = 0.7;
    config.carrierSubcarrierSpacing = 120;
    config.carrierNSizeGrid = 66;
    config.nStDrop = 1;
    config.maxRangeInterest = 400.0;

    % Override with file values
    overrides = loadConfigTxt(filepath);
    fields = fieldnames(overrides);
    for i = 1:length(fields)
        config.(fields{i}) = overrides.(fields{i});
    end
end

%% Load PRS config with defaults (matches Python PrsConfig)
function config = loadPrsConfig(filepath)
    % Defaults from Python src/gnrad5/config/models.py
    config = struct();
    config.PRSResourceSetPeriod = [1, 0];
    config.PRSResourceOffset = 0;
    config.PRSResourceRepetition = 1;
    config.PRSResourceTimeGap = 1;
    config.NumRB = 66;
    config.RBOffset = 0;
    config.CombSize = 2;
    config.REOffset = 0;
    config.NPRSID = 0;
    config.NumPRSSymbols = 2;
    config.SymbolStart = 0;

    % Override with file values
    overrides = loadConfigTxt(filepath);
    fields = fieldnames(overrides);
    for i = 1:length(fields)
        config.(fields{i}) = overrides.(fields{i});
    end
end

%% Load sensing config with defaults (matches Python SensConfig)
function config = loadSensConfig(filepath)
    % Defaults from Python src/gnrad5/config/models.py
    config = struct();
    config.dopplerFftLen = 64;
    config.window = 'blackmanharris';
    config.windowLen = [];
    config.windowOverlap = 0.5;
    config.numberSensingSymbols = 256;
    config.cfarGrdCellRange = 0;
    config.cfarGrdCellVelocity = 0;
    config.cfarTrnCellRange = 0;
    config.cfarTrnCellVelocity = 0;
    config.cfarTrnCellAzimuth = 8;
    config.cfarTrnCellElevation = 6;
    config.cfarGrdCellAzimuth = 4;
    config.cfarGrdCellElevation = 3;
    config.cfarThreshold = 3.0;
    config.azFftLen = 64;
    config.elFftLen = 64;
    config.rdaThreshold = 20.0;
    config.nmsRadius = [2, 2, 1, 1];
    config.nmsMaxPeaks = 200;

    % Override with file values
    overrides = loadConfigTxt(filepath);
    fields = fieldnames(overrides);
    for i = 1:length(fields)
        config.(fields{i}) = overrides.(fields{i});
    end
end

%% Helper function to load tab-separated config files
function config = loadConfigTxt(filepath)
    config = struct();

    if ~exist(filepath, 'file')
        return;
    end

    fid = fopen(filepath, 'r');
    if fid == -1
        return;
    end

    % Skip header line
    fgetl(fid);

    while ~feof(fid)
        line = fgetl(fid);
        if ischar(line) && ~isempty(strtrim(line))
            parts = strsplit(line, '\t');
            if length(parts) >= 2
                key = strtrim(parts{1});
                valueStr = strtrim(parts{2});

                % Parse value
                value = parseValue(valueStr);

                % Assign to struct
                config.(key) = value;
            end
        end
    end

    fclose(fid);
end

function value = parseValue(str)
    str = strtrim(str);

    % Check for array notation [...]
    if startsWith(str, '[') && endsWith(str, ']')
        inner = str(2:end-1);
        if isempty(strtrim(inner))
            value = [];
            return;
        end
        parts = strsplit(inner, {' ', ','});
        parts = parts(~cellfun(@isempty, parts));
        value = cellfun(@parseScalar, parts);
        return;
    end

    % Check for space/comma separated values
    if contains(str, ' ') || contains(str, ',')
        parts = strsplit(str, {' ', ','});
        parts = parts(~cellfun(@isempty, parts));
        if length(parts) > 1
            value = cellfun(@parseScalar, parts);
            return;
        end
    end

    % Single value
    value = parseScalar(str);
end

function val = parseScalar(str)
    str = strtrim(str);

    % Boolean
    if strcmpi(str, 'true')
        val = true;
        return;
    elseif strcmpi(str, 'false')
        val = false;
        return;
    end

    % Try numeric
    num = str2double(str);
    if ~isnan(num)
        val = num;
        return;
    end

    % String
    val = str;
end
