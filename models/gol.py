class gol(object):
    def __init__(self):#初始化
        global _global_dict
        _global_dict = {}

    def set_value(self, key, value):
        """ 定义一个全局变量 """
        # key = "d" 表示窗口在丢包时的下降程度
        # key = "v" 表示窗口的增加速率
        # key = "length" 表示当前队列的长度
        # key = "wait_time" 表示队列的等待时间
        # key = "ifImproved" 表示是否使用优化算法 ：1：优化，0：未优化
        _global_dict[key] = value

    def get_value(self, key, defValue=0):
        """获得一个全局变量,不存在则返回默认值"""
        try:
            return _global_dict[key]
        except KeyError:
            return defValue