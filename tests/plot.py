import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy import stats
def plotLoss(loss, goodput,queue_new,queue_qt):
    path = 'E:\chengjiyu\研究生毕设\结果图\仿真结果tcpmodel\\'
    plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
    fig = plt.figure(0)
    plt.plot(loss, goodput, label="平均吞吐量", color="red")
    plt.title('不同无线丢包下的实际平均吞吐量')
    plt.xlabel('无线丢包率[%]')  # make axis labels
    plt.ylabel('平均吞吐量')
    plt.legend(loc='best')
    plt.savefig(path + '平均吞吐量' + '.png')
    fig = plt.figure(1)
    plt.plot([i for i in range(10)], queue_new, label="the TCP NewReno queue size", color="red")
    plt.plot([i for i in range(10)], queue_qt, label="the TCP QL-WT queue size", color="blue")
    plt.title('不同队列长度占比分布')
    plt.xlabel('等待队列长度')  # make axis labels
    plt.ylabel('概率分布')
    plt.legend(loc='best')
    plt.savefig(path + '等待队列长度概率对比' + '.png')
    # fig = plt.figure(2)
    # interval_qt = []
    # interval_new = []
    # with open(path + 'interval_qt.txt', 'r') as d:
    #     for line in d:
    #         for i in line.split():
    #             interval_qt.append(float(i))
    # with open(path + 'interval_new.txt', 'r') as d:
    #     for line in d:
    #         for i in line.split():
    #             interval_new.append(float(i))
    # print(interval_new)
    # print(interval_qt)
    # sns.distplot(interval_new, hist=False, label="the TCP NewReno result", color="r")
    # sns.distplot(interval_qt,  hist=False, label="the TCP QL-WT result", color="b")
    # plt.title('The arrival interval distribution')
    # plt.xlabel('time')  # make axis labels
    # plt.ylabel('probability')
    # plt.xlim(0,20)
    # plt.legend(loc='best')
    # plt.savefig(path + 'The arrival interval distribution' + '.png')

    fig = plt.figure(3)
    new = [0.446,0.442,0.439,0.436,0.432,0.428]
    qt =  [0.485,0.482,0.481,0.478,0.474,0.471]
    improve = [8.74,9.05,9.57,9.63,9.72,10.05]
    plt.plot([i*0.1 for i in range(6)], new, label="the TCP NewReno result", color="red")
    plt.plot([i*0.1 for i in range(6)], qt, label="the TCP QL-WT result", color="blue")
    plt.title('不同丢包率的平均吞吐量')
    plt.xlabel('Wireless loss rate[%]')  # make axis labels
    plt.ylabel('Goodput')
    plt.legend(loc='best')
    plt.savefig(path + 'The average goodput' + '.png')
    fig = plt.figure(4)
    plt.plot([i*0.1 for i in range(6)], improve, label="the TCP QL-WT improved percentage", color="red")
    plt.title('QL-WT算法优化性能')
    plt.xlabel('Wireless loss rate[%]')  # make axis labels
    plt.ylabel('Improvement[%]')
    plt.legend(loc='best')
    plt.savefig(path + 'The average goodput' + '.png')
    plt.show()
    plt.close('all')
if __name__ == "__main__":
    loss = [i*0.1 for i in range(6)]
    goodput = [0.446,0.442,0.439,0.436,0.432,0.428]
    queue_new = [0.3254189944134078, 0.2579143389199255, 0.17411545623836128, 0.11126629422718808, 0.06098696461824953, 0.0335195530726257, 0.01722532588454376, 0.010707635009310988, 0.00558659217877095, 0.0032588454376163874]
    queue_qt = [0.29764549089293646, 0.22834295868502888, 0.16836961350510884, 0.11017325633051978, 0.07729897823189694, 0.04886717014660151, 0.03065304309195913, 0.020435362061306087, 0.011550422034651266, 0.006663705019991115]
    plotLoss(loss, goodput,queue_new,queue_qt)