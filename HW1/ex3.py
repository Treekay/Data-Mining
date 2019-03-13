import numpy as np
import csv

def func(x, y):
    return (y**2 * np.exp(-1 * y**2) + x**4 * np.exp(-1 * x**2)) / (x * np.exp(-1 * x**2))

def getBoundVal():
    maxVal = 0
    minVal = float('inf')
    for x in range(2, 5):
        for y in range(-1, 2):
            val = func(x, y)
            maxVal = val if val > maxVal else maxVal
            minVal = val if val < minVal else minVal
    return [maxVal, minVal]

if __name__ == "__main__":
    TEST_TIME = 100
    POINT_NUM = [10, 20, 30, 40, 50, 60, 70, 80, 100, 200, 500, 1000]

    # 写入文件
    csv_file = csv.writer(open('res3.csv', 'w', newline=''), dialect='excel')
    csv_file.writerow(['投点个数', '均值', '方差'])

    [maxVal, minVal] = getBoundVal()
    ## 对不同投点个数分别测试
    for k in range(len(POINT_NUM)):
        row = []
        Integration = []

        # 记录投点个数
        row.append(POINT_NUM[k])

        # 每种投点个数重复20次
        for t in range(TEST_TIME):
            count = 0
            for i in range(POINT_NUM[k]):
                x = np.random.uniform(2, 4)
                y = np.random.uniform(-1, 1)
                z = np.random.uniform(minVal, maxVal)
                # 判断随机点是否在函数曲线下方
                if z < func(x, y):
                    count += 1
            # 点数除以总点数得到比例, 比例乘上体积得到积分
            Integration.append(count / POINT_NUM[k] * 2 * 2 * (maxVal - minVal))

        # 记录均值和方差并写入表格
        row.append(np.mean(Integration))
        row.append(np.var(Integration))
        csv_file.writerow(row)

        # 输出结果
        print('POINT_NUM:', POINT_NUM[k], '  Integration:', np.mean(Integration))
