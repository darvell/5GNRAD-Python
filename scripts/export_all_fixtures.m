% EXPORT_ALL_FIXTURES Run fixture export for all scenarios
%
%   Run this script from the repository root directory (5GNRad/) to generate
%   validation fixtures for all scenarios found in examples/ and examples3GPP/
%
%   Prerequisites:
%       - MATLAB with 5G Toolbox
%       - All .m files in src/ must be on the MATLAB path
%
%   Usage:
%       >> cd('/path/to/5GNRad')
%       >> addpath('src')
%       >> addpath('scripts')
%       >> run('scripts/export_all_fixtures.m')
%
%   2025 NIST/CTL

%% Setup
repoRoot = pwd;
fprintf('Repository root: %s\n\n', repoRoot);

% Add src and scripts to path if not already
srcPath = fullfile(repoRoot, 'src');
scriptsPath = fullfile(repoRoot, 'scripts');

if ~contains(path, srcPath)
    addpath(srcPath);
    fprintf('Added src/ to MATLAB path\n');
end
if ~contains(path, scriptsPath)
    addpath(scriptsPath);
    fprintf('Added scripts/ to MATLAB path\n');
end

%% Find all scenario directories
scenarios = {};

% Check examples/
examplesDir = fullfile(repoRoot, 'examples');
if exist(examplesDir, 'dir')
    d = dir(examplesDir);
    for i = 1:length(d)
        if d(i).isdir && ~startsWith(d(i).name, '.')
            scenarioPath = fullfile(examplesDir, d(i).name);
            % Check for Input folder with config files
            if exist(fullfile(scenarioPath, 'Input', 'simulationConfig.txt'), 'file')
                scenarios{end+1} = scenarioPath; %#ok<SAGROW>
            end
        end
    end
end

% Check examples3GPP/
examples3GPPDir = fullfile(repoRoot, 'examples3GPP');
if exist(examples3GPPDir, 'dir')
    d = dir(examples3GPPDir);
    for i = 1:length(d)
        if d(i).isdir && ~startsWith(d(i).name, '.')
            scenarioPath = fullfile(examples3GPPDir, d(i).name);
            if exist(fullfile(scenarioPath, 'Input', 'simulationConfig.txt'), 'file')
                scenarios{end+1} = scenarioPath; %#ok<SAGROW>
            end
        end
    end
end

fprintf('Found %d scenarios:\n', length(scenarios));
for i = 1:length(scenarios)
    fprintf('  %d. %s\n', i, scenarios{i});
end
fprintf('\n');

%% Export fixtures for each scenario
successCount = 0;
failCount = 0;
failedScenarios = {};

for i = 1:length(scenarios)
    fprintf('\n========================================\n');
    fprintf('Processing scenario %d/%d: %s\n', i, length(scenarios), scenarios{i});
    fprintf('========================================\n');

    try
        export_matlab_fixtures(scenarios{i});
        successCount = successCount + 1;
    catch ME
        fprintf('ERROR: %s\n', ME.message);
        fprintf('Stack trace:\n');
        for k = 1:length(ME.stack)
            fprintf('  %s (line %d)\n', ME.stack(k).name, ME.stack(k).line);
        end
        failCount = failCount + 1;
        failedScenarios{end+1} = scenarios{i}; %#ok<SAGROW>
    end
end

%% Summary
fprintf('\n\n========================================\n');
fprintf('EXPORT SUMMARY\n');
fprintf('========================================\n');
fprintf('Total scenarios: %d\n', length(scenarios));
fprintf('Successful:      %d\n', successCount);
fprintf('Failed:          %d\n', failCount);

if failCount > 0
    fprintf('\nFailed scenarios:\n');
    for i = 1:length(failedScenarios)
        fprintf('  - %s\n', failedScenarios{i});
    end
end

fprintf('\n');
fprintf('To run Python validation tests:\n');
fprintf('  GN5_VALIDATE_RD=1 pytest tests/test_matlab_fixtures.py -v\n');
