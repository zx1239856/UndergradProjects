N = 10; % you can try to set N = either 10 or 1000
MIN_M = 1000;
MAX_M = 30000;
INTERVAL = 100;
END = (MAX_M - MIN_M) / INTERVAL + 1;

time_naive = zeros(1,END);
time_fft = zeros(1,END);
time_add = zeros(1,END);
time_save = zeros(1,END);
time_fft_no = zeros(1, END);
time_add_no = zeros(1, END);
time_save_no = zeros(1, END);
for i = 1 : END
    M = MIN_M + (i - 1) * INTERVAL;
    fprintf("Seq X length=%d, Y length=%d\n", M, N);
    x = rand([1, M]);
    y = rand([1, N]);
    t = clock;
    res1 = conv_naive(x, y);
    time_naive(i) = etime(clock, t);
    t = clock;
    res2 = conv_fft(x, y, false);
    time_fft_no(i) = etime(clock, t);
    t = clock;
    res3 = conv_overlap_add(x, y, false);
    time_add_no(i) = etime(clock, t);
    t = clock;
    res4 = conv_overlap_save(x, y, false);
    time_save_no(i) = etime(clock, t);
    checker(res2, res1, res3, res4);
    
    % enable power of 2 optimization
    t = clock;
    res2 = conv_fft(x, y, true);
    time_fft(i) = etime(clock, t);
    t = clock;
    res3 = conv_overlap_add(x, y, true);
    time_add(i) = etime(clock, t);
    t = clock;
    res4 = conv_overlap_save(x, y, true);
    time_save(i) = etime(clock, t);
    
    checker(res1, res2, res3, res4);
end

x = MIN_M:INTERVAL:MAX_M;
figure(1);
plot(x, time_naive, '-', x, time_fft_no, '-', x, time_add_no, '-', x, time_save_no, '-');
legend({'Naive', 'FFT', 'Overlap Add', 'Overlap Save'}, 'Location', 'northwest');
xlabel('Length of X (opt off)');
ylabel('Time Consumption/s');

figure(2);
plot(x, time_naive, '-', x, time_fft, '-', x, time_add, '-', x, time_save, '-');
legend({'Naive', 'FFT', 'Overlap Add', 'Overlap Save'}, 'Location', 'northwest');
xlabel('Length of X (opt on)');
ylabel('Time Consumption/s');

%% result checker
function checker(varargin)
    if(nargin > 1)
        for v = 2 : nargin
           for i = 1 : length(varargin{1})
               if(abs(varargin{v}(i) - varargin{1}(i)) > 1e-5)
                   fprintf("Error found in result, %f != %f\n", varargin{1}(i), varargin{v}(i));
               end
           end
        end
    end
end