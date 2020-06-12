%% Quantization Tables
Q_jpeg = [16 11 10 16 24 40 51 61;
        12 12 14 19 26 58 60 55;
        14 13 16 24 40 57 69 56;
        14 17 22 29 51 87 80 62;
        18 22 37 56 68 109 103 77;
        24 36 55 64 81 104 113 92;
        49 64 78 87 103 121 120 101;
        72 92 95 98 112 100 103 99];
    
Q_canon = [1 1 1 2 3 6 8 10;
           1 1 2 3 4 8 9 8;
           2 2 2 3 6 8 10 8;
           2 2 3 4 7 12 11 9;
           3 3 8 11 10 16 15 11;
           3 5 8 10 12 15 16 13;
           7 10 11 12 15 17 17 14;
           14 13 13 15 15 14 14 14;];

Q_nikon = [2 1 1 2 3 5 6 7;
           1 1 2 2 3 7 7 7;
           2 2 2 3 5 7 8 7;
           2 2 3 3 6 10 10 7;
           2 3 4 7 8 13 12 9;
           3 4 7 8 10 12 14 11;
           6 8 9 10 12 15 14 12;
           9 11 11 12 13 12 12 12;];


%% Program
img = rgb2gray(imread('lena.png'));

img_size = size(img);
h = img_size(1);
w = img_size(2);

blk_size = 8;
h_blks = h / blk_size;
w_blks = w / blk_size;

% a = 0.1:0.1:2;
a = 1;
vis = [0.1, 0.5, 1, 1.5, 2];
psnrs = zeros(size(a));
% for i=1:length(a)
%     Q = a(i) * Q_jpeg;
%     out_dct_2d = blk_transform(img, blk_size, @(x)dct_2d(x, Q));
%     out_idct = uint8(blk_transform(out_dct_2d, blk_size, @(x)idct_2d(x, Q)));
%     if ismember(a(i), vis)
%         figure('name', ['a: ', num2str(a(i))]),imshow(out_idct(256-31:256+32, 256-31:256+32));
%     end
%     psnrs(i) = psnr(out_idct, img);
% end
% 
% figure(),plot(a, psnrs),xlabel('x'),ylabel('PSNR');
Q = Q_jpeg;
out_dct_2d = blk_transform(img, blk_size, @(x)dct_2d(x, Q));
out_idct = uint8(blk_transform(out_dct_2d, blk_size, @(x)idct_2d(x, Q)));
fprintf("PSNR: %f\n", psnr(out_idct, img));
figure(),imshow(out_idct);

function out = dct_2d(blk, Q)
    blk = double(blk) - 128;
    out = dct2(blk);
    out = round(out./Q);
end

function out = idct_2d(blk, Q)
    out = round(idct2(blk .* Q) + 128);
end

function out = blk_transform(img, blk_size, trans_func)
    in_size = size(img);
    out = zeros(in_size);
    h_blks = in_size(1) / blk_size;
    w_blks = in_size(2) / blk_size;
    for i=1:h_blks
        for j=1:w_blks
            out((i-1)*blk_size+1:i*blk_size, (j-1)*blk_size+1:j*blk_size) ...
                = trans_func(img((i-1)*blk_size+1:i*blk_size, (j-1)*blk_size+1:j*blk_size));
        end
    end
end