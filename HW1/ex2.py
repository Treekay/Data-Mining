import numpy as np
import csv

TEST_TIME = 100
POINT_NUM = [5, 10, 20, 30, 40, 50, 60, 70, 80, 100, 300]

# 写入文件
csv_file = csv.writer(open('res2.csv', 'w', newline=''), dialect='excel')
csv_file.writerow(['投点个数', '均值', '方差'])

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
            x = np.random.rand()
            y = np.random.rand()
            # 判断随机点是否在函数曲线下方
            if y < x**3:
                count += 1
        # 点数除以总点数得到比例, 比例乘上体积1得到积分
        Integration.append(count / POINT_NUM[k])

    # 记录均值和方差并写入表格
    row.append(np.mean(Integration))
    row.append(np.var(Integration))
    csv_file.writerow(row)

    # 输出结果
    print('POINT_NUM:', POINT_NUM[k], '  Integration:', np.mean(Integration))