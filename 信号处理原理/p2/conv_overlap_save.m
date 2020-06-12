function res = conv_overlap_save(x, y, optimize)
    M = length(x); N = length(y);
    O = N;  % overlap length
    L = N + O;
    if(optimize)
        L = 2 ^ nextpow2(L);
        O = L - N;
    end
    x = [zeros(1, O), x]; % length M + O
    y = [y, zeros(1, O)]; % length L
    y = fft(y, L);
    res = zeros(1, M + N - 1);
    i = 1;
    while i <= M + O
        idx = min(i + L - 1, M + O);  % length L or tail
        temp = ifft(fft(x(i:idx), L) .* y, L);
        pos = min(i + N - 1, M + N - 1);
        res(i : pos) = temp(O + 1 : O + pos - i + 1);  % discard [1, O]
        i = i + N;
    end
end

