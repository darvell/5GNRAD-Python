function peaks = pick_peaks_nms(rdm_pow, noiseEst, alpha, supRad)
% rdm_pow: power RDM (not magnitude)
% noiseEst: CFAR noise estimate (same size)
% alpha: CFAR multiplier (linear)
% supRad: [range_halfwidth, doppler_halfwidth] suppression radius (bins)

% 1) CFAR-thresholded binary map
B = rdm_pow > alpha .* noiseEst;

% 2) Candidate local maxima (handle plateaus)
cands = imregionalmax(rdm_pow) & B;

% 3) Sort candidates by strength (desc)
[rows, cols] = find(cands);
vals = rdm_pow(cands);
[vals, ord] = sort(vals, 'descend'); rows = rows(ord); cols = cols(ord);

% 4) NMS: keep a peak, suppress neighbors within supRad
suppressed = false(size(rdm_pow));
se = strel('rectangle', [2*supRad(1)+1, 2*supRad(2)+1]);

peaks = [];  % rows: [row, col, value]
for k = 1:numel(vals)
    r = rows(k); c = cols(k);
    if ~suppressed(r,c)
        peaks(end+1,:) = [r, c, vals(k)]; %#ok<AGROW>
        m = false(size(rdm_pow)); m(r,c) = true;
        suppressed = suppressed | imdilate(m, se);   % suppress neighborhood
    end
end
end