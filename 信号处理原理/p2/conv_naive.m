function res = conv_naive(x, y)
   M = length(x);
   N = length(y);
   L = M + N - 1;  % \sum{x(k)y(t-k)}  1 <= k <= M, 1 <= t - k <= N
   res = zeros(1, L);
   for t = 2 : L + 1
       t1 = max(1, t - N);
       t2 = min(M, t - 1);
       for k = t1 : t2
           res(t - 1) = res(t - 1) + x(k) * y(t - k);  % note t start from 0 in the equation
       end
   end
end