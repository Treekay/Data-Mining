global trainNum testNum;
global height width;
global K; % 特征维数
global T; % 测试次数
trainNum = 7 * 40; % 训练图片数量
testNum = 3 * 40; % 测试图片数量
height = 112; % 图片高度
width = 92; % 图片宽度
xlspath = 'res/testRes.xlsx';

% 记录各个特征维数的正确率
accuracy = zeros(100, 10);
% 每个特征维数测试十次取平均
for T = 1 : 10
    % 对从 50 到 100 的 特征维数分别进行测试
    for K = 50 : 100
        randomSeq = zeros(trainNum, 10);
        trainImgs = zeros(trainNum, height, width);
        for i = 1 : 40
            % 生成一组随机排列的1-10序列
            % 序列前 7 位数字对应的图像用来训练
            % 序列后 3 位数字对应的图像用来测试
            randomSeq(i, :) = randperm(10);
            imgs = zeros(7, height, width);
            for j = 1 : 7
                imgs(j, :, :) = uint8(imread(['src/s' num2str(i) '/' num2str(randomSeq(i, j)) '.pgm']));
            end
            trainImgs((i - 1) * 7 + 1 : i * 7, :, :) = imgs;
        end
        trainImgs = uint8(trainImgs);

        % 将 280 张训练图像拉伸成1维列向量并拼接
        % 每张图像像素个数为 112 * 92 = 10304
        % 得到 10304 * 280 的数据矩阵 X
        centralizationMatrix = zeros(height * width, trainNum);
        for i = 1 : trainNum
            centralizationMatrix(:, i) = reshape(trainImgs(i, :, :), [height * width, 1]);
        end
        % 对整个数据矩阵求出均值向量
        meanMatrix = mean(centralizationMatrix, 2);
        % X 的每列减去均值向量进行中心化
        for i = 1: trainNum
           centralizationMatrix(:, i) = centralizationMatrix(:, i) - meanMatrix; 
        end
        % 求出 X 的转置和 X 的矩阵乘积得到协方差矩阵
        covarianceMatrix = centralizationMatrix' * centralizationMatrix;
        % 用 matlab 的 eig 函数求出特征向量
        [featureVector, temp] = eig(covarianceMatrix);
        % 选出前 K 大的特征值对应的特征向量 W
        featureVector = featureVector(:, trainNum - K + 1: trainNum);
        % 将中心化矩阵 X 乘上 W 映射得到基向量矩阵 V
        basisVector = centralizationMatrix * featureVector;
        % 将基向量矩阵 V 的每一列向量作为后续映射关系的一组基向量(特征脸)
        % 将中心化矩阵 X 每一列都通过基向量矩阵 V 映射到对应的特征空间中
        eigenfaces = basisVector' * centralizationMatrix;

        % 对于每个测试图像，也转成列向量，减去均值向量而中心化
        % 然后用基向量矩阵映射到特征空间中
        matchNum = 0;
        for i = 1 : 40
            for j = 8 : 10
                % 读取测试图像
                testImg = imread(['src/s' num2str(i) '/' num2str(randomSeq(i, j)) '.pgm']);
                % 中心化
                testImg = double(reshape(testImg, [height * width, 1])) - meanMatrix;
                % 计算待识别图像的投影
                testImg = basisVector' * testImg;
                [~, trainNum] = size(eigenfaces);
                % 计算待识别图像的投影和四十张训练图像在特征空间的坐标的二范数
                % 遍历搜索计算找到二范数最小的图像即为识别匹配出的图像
                maxDist = Inf;
                matchImg = 0;
                for k = 1 : trainNum
                    currentDist = norm(double(testImg) - eigenfaces(:, k), 2);
                    if maxDist > currentDist
                        matchImg = k;
                        maxDist = currentDist;
                    end
                end
                matchImg = floor((matchImg - 1) / 7) + 1;
                if matchImg == i
                    matchNum = matchNum + 1;
                end
            end
        end
        accuracy(K, T) = double(matchNum / testNum);
    end
end
xlswrite(xlspath, accuracy);
meanAccuracy = mean(accuracy, 2);
plot(meanAccuracy);
xlabel('特征维数 K');
ylabel('正确率');
axis([50 100 0.8 1]);
saveas(gcf, 'res/testRes.jpg');