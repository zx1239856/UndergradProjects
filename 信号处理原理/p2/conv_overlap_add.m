function res = conv_overlap_add(x, y, optimize)
    % it is recommended that y is much shorter than x here
    M = length(x);
    N = length(y);
    blk = N;
    L = blk + N - 1; % Xblk * Yn
    if(optimize)
        L = 2 ^ nextpow2(L);
        blk = L + 1 - N;
    end
    nblk = ceil(M/blk);  % number of blks
    Mx = nblk * blk;
    % L-point FFT
    x = [x, zeros(1, Mx - M)];
    y = fft(y, L);
    i = 1;
    res = zeros(1, Mx + L - 1);
    while i <= Mx
        il = min(i + blk - 1, Mx);
        temp = ifft(fft(x(i : il), L) .* y, L); % conv of the segment
        k = min(i + L - 1, Mx);
        res(i:i+L-1) = [res(i:k), zeros(1, L - 1 - k + i)] + temp;
        i = i + blk;
    end
    res = res(1:N+M-1);
end

