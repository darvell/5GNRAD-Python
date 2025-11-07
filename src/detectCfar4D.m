function detection4D = detectCfar4D(rdCube, sensConfig)
% DETECTCFAR4D  Fuse six 2-D CFAR detections into 4-D peaks with NMS
%   OUT = DETECTCFAR4D(RDCUBE, SENSCONFIG, RELTHRESHDB, NMSRADIUS, MAXPEAKS)
%   detects peaks in a 4-D range–Doppler–elevation–azimuth (R–D–El–Az) cube
%   by: (1) pre-thresholding the 4-D power, (2) projecting to six 2-D
%   planes, (3) running CFAR on each 2-D map, (4) accumulating per-plane
%   votes back into 4-D, and (5) extracting final 4-D peaks via greedy
%   non-maximum suppression (NMS) and light clustering/merging.
%
%   Inputs
%   ------
%   RDCUBE       : [R x D x El x Az] complex or real 4-D data. If complex,
%                  |RDCUBE|^2 is used as power; if real, values are treated
%                  as power directly (negative values clamped to zero).
%   SENSCONFIG   : struct with CFAR parameters (per 2-D plane):
%                  .cfarThreshold           : scalar CFAR threshold (linear or dB
%                                             inside cfar2D_fast_mod, as required)
%                  .cfarTrnCellRange        : training cells (range axis)
%                  .cfarGrdCellRange        : guard cells (range axis)
%                  .cfarTrnCellVelocity     : training cells (Doppler axis)
%                  .cfarGrdCellVelocity     : guard cells (Doppler axis)
%                  .cfarTrnCellAzimuth      : training cells (azimuth axis)
%                  .cfarGrdCellAzimuth      : guard cells (azimuth axis)
%                  .cfarTrnCellElevation    : training cells (elevation axis)
%                  .cfarGrdCellElevation    : guard cells (elevation axis)
%
%   Output
%   ------
%   OUT : [N x 4] integer indices of finalized 4-D peak locations:
%         columns = [r_idx, d_idx, el_idx, az_idx]
%
%
%   See also  PROJ2D, CFAR2D_FAST_MOD, NMS4D_GREEDY_ONV, CLUSTER_PEAKS_4D, MERGE_PEAKS_3OF4
%
%   2025 NIST/CTL Steve Blandino
%
%   This file is available under the terms of the NIST License.


%  Power cube & global pre-threshold
if ~isreal(rdCube), P = abs(rdCube).^2; else, P = rdCube; end
P = max(P, 0);
peakVal = max(P(:));
thrLin  = peakVal / (10^(sensConfig.rdaThreshold/10));
P(P < thrLin) = 0;

[R,D,El,Az] = size(P);

%% 2-D projections
maps.RD   = proj2D(P, [1 2]);  % [R x D]
maps.RAz  = proj2D(P, [1 4]);  % [R x Az]
maps.REl  = proj2D(P, [1 3]);  % [R x El]
maps.DAz  = proj2D(P, [2 4]);  % [D x Az]
maps.DEl  = proj2D(P, [2 3]);  % [D x El]
maps.AzEl = proj2D(P, [4 3]);  % [Az x El]

%% CFAR on each 2-D map
% Params
sensConfig.cfarTrnCellAzimuth = 8;
sensConfig.cfarTrnCellElevation = 6;
sensConfig.cfarGrdCellAzimuth = 4;
sensConfig.cfarGrdCellElevation = 3;
grdRD = [sensConfig.cfarGrdCellRange,    sensConfig.cfarGrdCellVelocity];
trnRD = [sensConfig.cfarTrnCellRange,    sensConfig.cfarTrnCellVelocity];

grdRAz = [sensConfig.cfarGrdCellRange,    sensConfig.cfarGrdCellAzimuth];
trnRAz = [sensConfig.cfarTrnCellRange,    sensConfig.cfarTrnCellAzimuth];

grdREl = [sensConfig.cfarGrdCellRange,    sensConfig.cfarGrdCellElevation];
trnREl = [sensConfig.cfarTrnCellRange,    sensConfig.cfarTrnCellElevation];

grdDAz = [sensConfig.cfarGrdCellVelocity, sensConfig.cfarGrdCellAzimuth];
trnDAz = [sensConfig.cfarTrnCellVelocity, sensConfig.cfarTrnCellAzimuth];

grdDEl = [sensConfig.cfarGrdCellVelocity, sensConfig.cfarGrdCellElevation];
trnDEl = [sensConfig.cfarTrnCellVelocity, sensConfig.cfarTrnCellElevation];

grdAzEl = [sensConfig.cfarGrdCellAzimuth, ...
    sensConfig.cfarGrdCellElevation];
trnAzEl = [sensConfig.cfarTrnCellAzimuth, ...
    sensConfig.cfarTrnCellElevation];

thrCFAR = sensConfig.cfarThreshold;

det = struct();
[det.RD.mask,   det.RD.noise]   = cfar2D_fast_mod(maps.RD,   grdRD,   trnRD,   thrCFAR);
[det.RAz.mask,  det.RAz.noise]  = cfar2D_fast_mod(maps.RAz,  grdRAz,  trnRAz,  thrCFAR);
[det.REl.mask,  det.REl.noise]  = cfar2D_fast_mod(maps.REl,  grdREl,  trnREl,  thrCFAR);
[det.DAz.mask,  det.DAz.noise]  = cfar2D_fast_mod(maps.DAz,  grdDAz,  trnDAz,  thrCFAR);
[det.DEl.mask,  det.DEl.noise]  = cfar2D_fast_mod(maps.DEl,  grdDEl,  trnDEl,  thrCFAR);
[det.AzEl.mask, det.AzEl.noise] = cfar2D_fast_mod(maps.AzEl, grdAzEl, trnAzEl, thrCFAR);

%% Accumulator
A = zeros(R,D,El,Az,'uint8');     % votes accumulator

peaks = pick_peaks_nms(det.RD.mask,  det.RD.noise,  thrCFAR, grdRD);
if ~isempty(peaks)
    r = peaks(:,1); d = peaks(:,2);
    [r,d] = deal(r(:), d(:));
    [r,d] = deal(max(1,min(R,r)), max(1,min(D,d)));       % bounds
    for k = 1:numel(r), A(r(k),d(k),:,:) = A(r(k),d(k),:,:) + 1; end
end

% R-Az (map: [R x Az]) -> vote over all Doppler & all Elevation at (r,az)
peaks = pick_peaks_nms(det.RAz.mask, det.RAz.noise, thrCFAR, grdRAz);
if ~isempty(peaks)
    r  = peaks(:,1); az = peaks(:,2);
    r  = max(1,min(R, r(:)));      az = max(1,min(Az, az(:)));
    for k = 1:numel(r), A(r(k),:, :, az(k)) = A(r(k),:, :, az(k)) + 1; end
end

% R-El (map: [R x El]) -> vote over all Doppler & all Azimuth at (r,el)
peaks = pick_peaks_nms(det.REl.mask, det.REl.noise, thrCFAR, grdREl);
if ~isempty(peaks)
    r  = peaks(:,1); el = peaks(:,2);
    r  = max(1,min(R,  r(:)));     el = max(1,min(El, el(:)));
    for k = 1:numel(r), A(r(k),:, el(k), :) = A(r(k),:, el(k), :) + 1; end
end

% D-Az (map: [D x Az]) -> vote over all Range & all Elevation at (d,az)
peaks = pick_peaks_nms(det.DAz.mask, det.DAz.noise, thrCFAR, grdDAz);
if ~isempty(peaks)
    d  = peaks(:,1); az = peaks(:,2);
    d  = max(1,min(D,  d(:)));     az = max(1,min(Az, az(:)));
    for k = 1:numel(d), A(:, d(k), :, az(k)) = A(:, d(k), :, az(k)) + 1; end
end

% D-El (map: [D x El]) -> vote over all Range & all Azimuth at (d,el)
peaks = pick_peaks_nms(det.DEl.mask, det.DEl.noise, thrCFAR, grdDEl);
if ~isempty(peaks)
    d  = peaks(:,1); el = peaks(:,2);
    d  = max(1,min(D,  d(:)));     el = max(1,min(El, el(:)));
    for k = 1:numel(d), A(:, d(k), el(k), :) = A(:, d(k), el(k), :) + 1; end
end

% Az-El (map: [Az x El]) -> vote over all Range & all Doppler at (az,el)
peaks = pick_peaks_nms(det.AzEl.mask, det.AzEl.noise, thrCFAR, grdAzEl);
if ~isempty(peaks)
    az = peaks(:,1); el = peaks(:,2);
    az = max(1,min(Az, az(:)));    el = max(1,min(El, el(:)));
    for k = 1:numel(az), A(:, :, el(k), az(k)) = A(:, :, el(k), az(k)) + 1; end
end

V = zeros(size(A));
toKeep = A>=3;
V(toKeep) = P(toKeep);
pk = nms4d_greedy_onV(V, sensConfig.nmsRadius, sensConfig.nmsMaxPeaks);
dims = struct('R', R, 'D', D, 'El', El, 'Az', Az, ...
    'wrapD', true, 'wrapAz', true);    % wrap Doppler/Azimuth

opts = struct();
opts.minVal    = pk(1,end)/55;        % discard weak peaks (linear units)
opts.norm      = [2 2 1 1];% normalize bin distances per axis 
opts.valWeight = 0.5;      % include value similarity (0 = ignore)
opts.valScale  = [];       % leave empty to auto (robust MAD)
opts.eps       = 6;      % clustering radius in normalized space
opts.minPts    = 1;        % at least 1 peak to form a cluster

cl = cluster_peaks_4d(pk, dims, opts);

detection4D = table2array(cl.clusters(:,[8:11 6]));
detection4D = round(suppressSidelobes(detection4D, inf, [R D El Az]));
detection4D(:,end) = [];
end

%  helpers

function M = proj2D(cube, keepDims)
% PROJ2D Project a 4-D cube onto a 2-D plane by summing across other dimensions
%   M = PROJ2D(CUBE, KEEPDIMS) reduces a 4-D array CUBE by summing along
%   all dimensions not specified in KEEPDIMS. The result M is a 2-D matrix
%   representing the projection of CUBE over the selected dimensions.
%
%   Inputs
%   ------
%   CUBE     : [R x D x El x Az] 4-D numeric array (e.g., range-Doppler-angle cube)
%   KEEPDIMS : [1 x 2] vector specifying which dimensions to preserve
%               (e.g., [1 2] for range–Doppler, [1 3] for range–elevation)
%
%   Outputs
%   -------
%   M : 2-D matrix obtained by summing CUBE along all dimensions not in KEEPDIMS.
%       The output is squeezed to remove singleton dimensions.
%
%   Example
%   -------
%       % Example with a random cube
%       cube = rand(32, 32, 8, 8);
%       M = proj2D(cube, [1 2]);   % Range–Doppler projection
%
%   2025 NIST/CTL Steve Blandino
%
%   This file is available under the terms of the NIST License.

allDims = 1:4;
dropDims = setdiff(allDims, keepDims);
M = cube;

for dd = dropDims
    M = sum(M, dd);
end

M = squeeze(M);
end

function peaks = nms4d_greedy_onV(V, rad, maxPeaks)
% NMS4D_GREEDY_ONV 
%   Works only on nonzero/above-threshold voxels instead of the full 4-D grid.

    % ---- Tunable: ignore tiny values to shrink candidate set
    minPeakVal = 1e-12;

    % ---- Pull only nonzero/strong candidates
    if ~isreal(V), A = abs(V); else, A = V; end
    mask = A > minPeakVal;
    if ~any(mask(:))
        peaks = zeros(0,5); return;
    end

    [R,D,El,Az] = size(A);
    linIdx = find(mask);                 % [M x 1] linear indices of candidates
    vals   = A(linIdx);                  % [M x 1] values of candidates

    % Convert once to subscripts (avoid ind2sub inside loop)
    [rr,dd,ee,aa] = ind2sub([R,D,El,Az], linIdx);

    % Sort by value (desc) so greedy selection becomes a single forward pass
    [vals, order] = sort(vals, 'descend');
    rr = rr(order); dd = dd(order); ee = ee(order); aa = aa(order);

    % Greedy suppression on the compact candidate list
    rR = rad(1); rD = rad(2); rEl = rad(3); rAz = rad(4);
    M  = numel(vals);
    alive = true(M,1);

    out = zeros(min(maxPeaks,M), 5);
    np  = 0;

    i = 1;
    while np < maxPeaks && i <= M
        if ~alive(i), i = i + 1; continue; end
        np = np + 1;
        out(np,:) = [rr(i), dd(i), ee(i), aa(i), vals(i)];

        % Suppress neighbors within Chebyshev (box) radius in each dim
        % Vectorized on the remaining alive candidates.
        % (Do comparisons only against alive to reduce work.)
        J = find(alive);          % indices into compact arrays
        % center values
        r0 = rr(i); d0 = dd(i); e0 = ee(i); a0 = aa(i);
        % compute suppression mask among currently alive
        kill = abs(rr(J) - r0) <= rR & ...
               abs(dd(J) - d0) <= rD & ...
               abs(ee(J) - e0) <= rEl & ...
               abs(aa(J) - a0) <= rAz;

        % Mark those as suppressed
        alive(J(kill)) = false;

        % Keep the chosen one alive -> false already included in kill
        % Move forward (no need to increment by >1; the 'alive' check will skip)
        i = i + 1;
    end

    if np == 0
        peaks = zeros(0,5);
    else
        peaks = out(1:np,:);
    end
end
