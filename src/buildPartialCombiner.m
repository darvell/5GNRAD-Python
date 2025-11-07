function W_RF = buildPartialCombiner(receiveArray, fc, NRF, beamAngles)
% Partially-connected analog combiner (block-diagonal), unit-modulus per phase shifter.
% beamAngles: [NRF x 2] [az, el] in degrees (one pointing per subarray)

Nrx = prod(receiveArray.Size);
assert(mod(Nrx,NRF)==0, 'Nrx must be divisible by NRF');
Ns = Nrx/NRF;

sv = phased.SteeringVector('SensorArray', receiveArray);
% Full-array steering (we will slice per subarray)
Afull = sv(fc, beamAngles.');       % [Nrx x NRF]

W_RF = zeros(Nrx, NRF);
for r = 1:NRF
    % Simple contiguous partition (first Ns elems → chain 1, next Ns → chain 2, ...)
    idx = (r-1)*Ns + (1:Ns);        % indices of elements belonging to RF chain r

    % Take the steering vector entries for those elements and keep only phase (unit-modulus)
    w_phase = exp(-1j*angle(Afull(idx, r)));   % unit-modulus phases
    W_RF(idx, r) = w_phase / sqrt(Ns);         % normalize so ||w||^2 = 1
end
end
