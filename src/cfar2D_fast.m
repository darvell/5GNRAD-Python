function [y,noiseEst] = cfar2D_fast(rdm, guardRadius, trainingRadius, threshold)
% CFAR2D_FAST  Vectorized 2-D CA-CFAR using box filters (O(N) total cost).
%   Y = CFAR2D_FAST(RDM, GUARDRADIUS, TRAININGRADIUS, THRESHOLD) performs
%   cell-averaging constant false-alarm rate (CA-CFAR) detection over the
%   2-D range–Doppler (or angle–range) power map RDM. For each cell under
%   test (CUT), the average noise/clutter is estimated from a surrounding
%   square "training" ring that excludes a square "guard" region. A
%   detection is declared when CUT > THRESHOLD * noise estimate.
%
%   [Y, NOISEEST] = CFAR2D_FAST(...) also returns the per-pixel noise
%   estimate computed from the training ring.
%
%   This implementation is fully vectorized by using 2-D box convolutions
%   to obtain training/guard sums, reducing complexity from per-CUT window
%   summations to O(N) for an RDM with N elements.
%
%   INPUTS
%     RDM            : MxN real, nonnegative matrix of power values
%                      (linear scale). If your data are in dB, convert to
%                      linear before calling (e.g., 10.^(RDM_dB/10)).
%
%     GUARDRADIUS    : 1x2 integer vector [gR gC] giving the half-sizes of
%                      the square guard window centered at the CUT:
%                      guard height = 2*gR+1, guard width = 2*gC+1.
%
%     TRAININGRADIUS : 1x2 integer vector [tR tC] giving the half-sizes of
%                      the square training window centered at the CUT:
%                      training height = 2*tR+1, training width = 2*tC+1.
%                      Must satisfy tR >= gR and tC >= gC. The training
%                      ring is the training square minus the guard square.
%
%     THRESHOLD      : Scalar detection multiplier applied to the estimated
%                      noise/clutter level (linear scale). For a dB
%                      threshold T_dB, use THRESHOLD = 10^(T_dB/10).
%
%   OUTPUTS
%     Y        : MxN matrix, same class as RDM. Y equals RDM at detected
%                pixels and 0 elsewhere (boolean mask applied to input).
%
%     NOISEEST : MxN matrix (double or cast-compatible with RDM) containing
%                the ring-average noise estimate at each pixel where the
%                full training window is available. Values outside the
%                valid interior are still computed from convolutions but
%                are not used for decisions (see Notes).
%
%   NOTES
%     * Border handling: Detections are evaluated only where a full
%       training window fits entirely inside RDM. CUTs within tR rows of
%       the top/bottom or tC columns of the left/right borders are masked
%       out (no detection reported). This matches common CFAR practice.
%
%     * Window geometry: The number of training cells is
%           K = (2*tR+1)*(2*tC+1) - (2*gR+1)*(2*gC+1).
%       The noise estimate is mean(training ring) = ringSum / K.
%
%     * Scaling/units: Ensure THRESHOLD and RDM are in the same (linear)
%       domain. If you want to set a threshold in dB (e.g., +12 dB above
%       clutter), convert it to linear as described above.
%
%     * Performance: Using conv2 with all-ones kernels implements summed-area
%       (box) filters, yielding O(MN) total cost independent of window size.
%
%   EXAMPLE
%     % RDM in linear power, guard 3x3, training 7x7, threshold = 12 dB
%     T_dB = 12; thr = 10^(T_dB/10);
%     [y, nEst] = cfar2D_fast(RDM, [1 1]*1, [1 1]*3, thr);
%     imagesc(y > 0); axis image; title('CFAR detections');
%
%   SEE ALSO  conv2, imboxfilt, medfilt2
%
%   2025 Steve Blandino / NIST CTL
%
%   This file is available under the terms of the NIST License.
% Radii
gR = guardRadius(1); gC = guardRadius(2);
tR = trainingRadius(1); tC = trainingRadius(2);

% Box kernels
kerT = ones(2*tR+1, 2*tC+1, 'like', rdm);
kerG = ones(2*gR+1, 2*gC+1, 'like', rdm);

% Sums over training square and guard square
sumT = conv2(rdm, kerT, 'same');
sumG = conv2(rdm, kerG, 'same');

% Training-ring sum and count
K = numel(kerT) - numel(kerG);
ringSum = sumT - sumG;
noiseEst = ringSum / K;

% Build a mask for CUT positions that have a full training window
mask = true(size(rdm));
mask([1:tR end-tR+1:end], :) = false;
mask(:, [1:tC end-tC+1:end]) = false;

% Optionally also remove borders needed for guard window (not strictly necessary
% since ringSum removed guard contribution), but to match your CUT interior:
mask([1:tR end-tR+1:end], :) = false;
mask(:, [1:tC end-tC+1:end]) = false;

% Threshold (linear scale). If your threshold is in dB, use: threshold = 10^(T_dB/10)
detect = false(size(rdm));
detect(mask) = rdm(mask) > (noiseEst(mask) * threshold);

% Output: keep original power at detections, zero elsewhere
y = zeros(size(rdm), 'like', rdm);
y(detect) = rdm(detect);
end
