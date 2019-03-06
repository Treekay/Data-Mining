import numpy as np
import csv

TEST_TIME = 20
POINT_NUM = [20, 50, 100, 200, 300, 500, 1000, 5000]

# 写入文件
csv_file = csv.writer(open('res1.csv', 'w', newline=''), dialect='excel')
csv_file.writerow(['投点个数', '均值', '方差'])

## 对不同投点个数分别测试
for k in range(len(POINT_NUM)):
    row = []
    PI = []

    # 记录投点个数
    row.append(POINT_NUM[k])

    # 每种投点个数重复20次
    for t in range(20):
        count = 0
        for i in range(POINT_NUM[k]):
            x = np.random.rand()
            y = np.random.rand()
            # 记录每个随机点到原点的距离
            dis = (x**2 + y**2) ** 0.5
            # 判断随机点是否在圆范围内, 是则计数加一
            if dis <= 1:
                count += 1
        # 用圆内点数除以总点数得到 1/4 PI
        PI.append(4 * count / POINT_NUM[k])

    # 记录均值和方差并写入表格
    row.append(np.mean(PI))
    row.append(np.var(PI))
    csv_file.writerow(row)

    # 输出结果
    print('POINT_NUM:', POINT_NUM[k], '  PI:', np.mean(PI))