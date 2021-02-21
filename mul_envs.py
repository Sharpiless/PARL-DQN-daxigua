from multiprocessing import Pipe, Pool
from State2NN import AI_Board


def run_game(child_conn):
    game = AI_Board()
    isdone = False
    while True:
        action = child_conn.recv()  # 队列为空时会阻塞进程
        print('-[INFO] get action:{} from queue'.format(action))
        image, reward, isdone = game.run()
        if isdone:
            game.reset()
        child_conn.send([image, reward, isdone])


class MulEnvs(object):

    def __init__(self, process_num):
        self.process_num = process_num

    def init_process(self):
        self.Process_Pool = Pool(self.process_num)
        self.Parent_Conns, self.Child_Conns = zip(
            *[Pipe() for _ in range(self.process_num)])
        for id_ in range(self.process_num):
            # 创建process_num个游戏进程
            self.Process_Pool.apply_async(
                run_game, args=(self.Child_Conns[id_],))

    def next(self, Actions):
        Images, Rewards, Isdones = [], [], []
        for id_ in range(self.process_num):
            self.Parent_Conns[id_].send(Actions[id_])

        for id_ in range(self.process_num):
            image, reward, isdone = self.Parent_Conns[id_].recv()
            print(
                '-[INFO] id_{} get action:{} from queue, reward:{}'.format(id_, action, reward))
            Images.append(image)
            Rewards.append(reward)
            Isdones.append(isdone)
            
        return Images, Rewards, Isdones
