import simpy
import numpy as np
from .log import Logger
from .gol import gol

class BaseSourceModel():
    '''The Base class for source model,
    ------------methods---------------
    get_pkt_num() : return a int that represents the amount of packets that the source generated
    get_interval() : return a float that represents the interval for the next message
    on_served() : decorator for feedback on the successful delivering
    on_dropped() : decorator for feedback on the fail delivering
    '''
    def __init__(self):
        pass

    def get_pkt_num(self):
        return 1

    def get_interval(self):
        return 1.

    def on_served(self):
        print('The source received feedback for successful delivering')

    def on_droped(self):
        print('The source received feedback for transmission failure')

class MMPPModel(BaseSourceModel):
    '''The Markov Modulated Poisson Process Source Model'''

    def __init__(self, Q, Lambda):
        assert(len(np.shape(Q)) is 2)
        assert(len(np.shape(Lambda)) is 1)
        assert(np.shape(Q)[0] == np.shape(Q)[1])
        assert(np.shape(Q)[0] == len(Lambda))
        self.__log = Logger('ACK', 'data.txt')
        self.__gol = gol()
        self.__Q = np.atleast_2d(Q)
        self.__state_transition = np.cumsum(self.__Q, axis = 1)
        self.__Lambda = np.atleast_1d(Lambda)
        self.__states = np.array([i for i in range(Lambda.shape[0])])
        self.__cur_state = np.random.randint(0, self.__states[-1]+1)      # self.__states=[0,1,2,3]; self.__cur_state=[0,1,2]
        self.__init_cwnd = 1
        self.__init_ssth = 65535
        self.__cwnd = 1400
        self.__ssth = 65535
        self.__accumulator = 0
        self.__segment = 1400       # add segment size by chengjiyu on 2016/10/9
        self.__d = "d"             # 初始化当丢包时的窗口减小因子 chengjiyu on 2017/06/05
        self.__v = "v"            # 初始化窗口的增长速率 chengjiyu on 2017/11/10

    def get_lambda(self):
        # 参数定义
        s_1 = 8.4733 * 10 ** (-4)
        s_2 = 5.0201 * 10 ** (-6)
        pi = np.mat(np.array([0.0058897, 0.9941103]))
        e = np.mat(np.ones((2, 1)))
        In = np.mat(np.eye(2))
        L = np.mat(np.array([[1.0722, 0], [0, 0.48976]]))
        Q_th = np.mat(np.array([[-s_1, s_1], [s_2, -s_2]]))
        fai = pi * L / (pi * L * e)
        l = np.mat(np.array([1.0722, 0.48976]))
        v = self.__gol.get_value(self.__v, 0.182)
        d = self.__gol.get_value(self.__d)  # d 的取值范围是 0 - 1, d 由全局变量 _global_dict 获得
        b = v * fai * (
            (Q_th - L).I ** 2 * L * (1 - d) * (In - (1 - d) * (Q_th - L).I * L).I * (Q_th - L).I ** 2 * L - (
            Q_th - L).I ** 3 * L) * e
        self.__gol.set_value("ifImproved", 1)
        if self.__gol.get_value("ifImproved"):
            Lambda = np.array((b * l).tolist()[0])   # [1.0722,0.48976]
            print("Lambda:", Lambda)
        else:
            Lambda = np.array([1.0722, 0.48976])
        return Lambda
    def get_interval(self):
        state = self.__states[self.__cur_state]
        rate = self.__Lambda[state]
        dice = np.random.random()
        self.__cur_state = np.argwhere(self.__state_transition[self.__cur_state] > dice)[0][0]
        self.__Lambda = self.get_lambda()
        # self.__gol.set_value("lambda", self.get_lambda())
        # Find the indices of array elements that are non-zero, grouped by element.
        # return position of the first meet specified condition
        return np.random.exponential(1./ rate) / rate

    @property
    def cur_state(self):
        return self.__cur_state

    @property
    def Q(self):
        return self.__Q
    # add tcp by chengjiyu on 2016/10/8
    def on_served(self):
        print('The source received feedback for successful delivering')
        self.__log.logger.info('The source received feedback for successful delivering')
        self.__gol.set_value(self.__d, 0)  # 对丢包时的窗口减小因子d赋值 chengjiyu on 2017/06/05
        # 队列长度检测算法QLD chengjiyu on 2017/11/10
        if self.__gol.get_value("lenght") < 3:
            v = 0.182
        elif self.__gol.get_value("lenght") > 6:
            v = 0.142
        else:
            v = 0.182-(0.182-0.142)*(self.__gol.get_value("lenght")-3)/3
        self.__gol.set_value(self.__v, v)  # 窗口的增长速率v赋值 chengjiyu on 2017/11/10
        if self.__cwnd <= self.__ssth:
            self.__cwnd += self.__segment
            print("Acked in Slow Start Phase")
            print("cwnd is {0}, ssth is {1}".format(self.__cwnd, self.__ssth))
            self.__log.logger.info("cwnd is {0}, ssth is {1}".format(self.__cwnd, self.__ssth))
        else:
            print("Acked in Congestion avoidance")
            adder = self.__segment * self.__segment / self.__cwnd
            adder = int(max(1.0, adder))
            # self.__accumulator += 1
            # if self.__accumulator == self.__cwnd:
            # self.__cwnd += 1
            # self.__accumulator = 0
            self.__cwnd += adder
            print("cwnd is {0}, ssth is {1}".format(self.__cwnd, self.__ssth))
            self.__log.logger.info("cwnd is {0}, ssth is {1}".format(self.__cwnd, self.__ssth))

    def on_droped(self):
        print('The source received feedback for transmission failure')
        self.__log.logger.info('The source received feedback for transmission failure')
        # 等待时间检测算法WTD chengjiyu on 2017.11.10
        if self.__gol.get_value("lenght") < 3:
            d = 0
        elif self.__gol.get_value("lenght") > 6:
            d = 0.5
        else:
            d = 0.5*(self.__gol.get_value("lenght")-3)/3
        self.__ssth = max(2 * self.__segment, self.__cwnd // 2)
        if self.__gol.get_value("ifImproved"):
            self.__cwnd = max(2 * self.__segment,(1-d)*self.__cwnd) # WTD算法
        else:
            self.__cwnd = self.__ssth# + 3 * self.__segment
        # self.__cwnd = max(self.__cwnd // 2, 1)
        # self.__ssth = max(self.__cwnd, 2)
        print("duplicate acks \ncwnd is {0}, ssth is {1}".format(self.__cwnd, self.__ssth))
        self.__log.logger.info("cwnd is {0}, ssth is {1}".format(self.__cwnd, self.__ssth))
        self.__gol.set_value(self.__d, d)         # 对丢包时的窗口减小因子赋值 chengjiyu on 2017/06/05

    def on_timeout(self):
        print('The source received feedback for time out')
        self.__log.logger.info('The source received feedback for time out')
        self.__gol.set_value(self.__d, 1)  # 对丢包时的窗口减小因子赋值 chengjiyu on 2017/06/05
        # self.__ssth = max(self.__cwnd // 2, 2)
        # self.__cwnd = 0
        self.__ssth = max(2 * self.__segment, self.__cwnd // 2)
        self.__cwnd = self.__segment
        print("time out \ncwnd is {0}, ssth is {1}".format(self.__cwnd, self.__ssth))
        self.__log.logger.info("cwnd is {0}, ssth is {1}".format(self.__cwnd, self.__ssth))


class TcpSourceModel(BaseSourceModel):
    '''The emulated TCP source model'''

    def __init__(self, rtt):
        self.__rtt = rtt
        self.__segment = 1400
        self.__cwnd = 1
        self.__ssth = 65535
        self.__acked = 0
        self.__cum = 0
        self.__log = Logger('ACK', 'data.txt')

    @property
    def cwnd(self):
        return self.__cwnd

    @property
    def ssth(self):
        return self.__ssth

    def get_interval(self):
        rate = self.__cwnd / self.__segment * self.__rtt
        # self.__segment * self.__cwnd / self.__rtt变为self.__cwnd / self.__segment * self.__rtt 2017/04/30
        return np.random.exponential(1. / rate) / rate

    def on_served(self):
        print('The source received feedback for successful delivering')
        self.__log.logger.info('The source received feedback for successful delivering')
        if self.__cwnd <= self.__ssth:
            self.__cwnd += self.__segment
            print("Acked in Slow Start Phase")
            print("cwnd is {0}, ssth is {1}".format(self.__cwnd, self.__ssth))
            self.__log.logger.info("cwnd is {0}, ssth is {1}".format(self.__cwnd, self.__ssth))
        else:
            print("Acked in Congestion avoidance")
            adder = self.__segment * self.__segment / self.__cwnd
            adder = int(max(1.0, adder))
            # self.__accumulator += 1
            # if self.__accumulator == self.__cwnd:
            # self.__cwnd += 1
            # self.__accumulator = 0
            self.__cwnd += adder
            print("cwnd is {0}, ssth is {1}".format(self.__cwnd, self.__ssth))
            self.__log.logger.info("cwnd is {0}, ssth is {1}".format(self.__cwnd, self.__ssth))

    def on_droped(self):
        print('The source received feedback for transmission failure')
        self.__log.logger.info('The source received feedback for transmission failure')
        self.__ssth = max(2 * self.__segment, self.__cwnd // 2)
        self.__cwnd = self.__ssth# + 3 * self.__segment
        # self.__cwnd = max(self.__cwnd // 2, 1)
        # self.__ssth = max(self.__cwnd, 2)
        print("duplicate acks \ncwnd is {0}, ssth is {1}".format(self.__cwnd, self.__ssth))
        self.__log.logger.info("cwnd is {0}, ssth is {1}".format(self.__cwnd, self.__ssth))

    def on_timeout(self):
        print('The source received feedback for time out')
        self.__log.logger.info('The source received feedback for time out')
        # self.__ssth = max(self.__cwnd // 2, 2)
        # self.__cwnd = 0
        self.__ssth = max(2 * self.__segment, self.__cwnd // 2)
        self.__cwnd = self.__segment
        print("time out \ncwnd is {0}, ssth is {1}".format(self.__cwnd, self.__ssth))
        self.__log.logger.info("cwnd is {0}, ssth is {1}".format(self.__cwnd, self.__ssth))

    # def on_served(self):
    #     if self.__cwnd < self.__ssth:
    #         self.__cwnd += 1
    #         print("Acked in Slow Start Phase")
    #         print("cwnd is {0}, ssth is {1}".format(self.__cwnd, self.__ssth))
    #     else:
    #         print("Acked in Congestion avoidance")
    #         self.__cum += 1
    #         if self.__cum == self.__cwnd:
    #             self.__cwnd += 1
    #             self.__cum = 0
    #         print("cwnd is {0}, ssth is {1}".format(self.__cwnd, self.__ssth))
    #
    # # add duplicate acks by chengjiyu on 2016/9/28
    # def on_droped(self):
    #     self.__cwnd = max(self.__cwnd / 2, 1)
    #     self.__ssth = max(self.__cwnd, 2)
    #     print("duplicate acks \ncwnd is {0}, ssth is {1}".format(self.__cwnd, self.__ssth))

