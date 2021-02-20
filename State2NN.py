import pygame as pg
from random import random, randrange
from Fruit import create_fruit
from Game import GameBoard
import numpy as np


def sorted_index(list_):
    return sorted(range(len(list_)), key=lambda k: list_[k])


class AI_Board(GameBoard):
    def __init__(self):
        self.create_time = 1.0
        self.gravity = (0, 1800)
        GameBoard.__init__(self, self.create_time, self.gravity)
        self.action_num = 16
        self.init_segment()
        self.setup_collision_handler()

    def decode_action(self, action):

        seg = (self.WIDTH - 40) // self.action_num
        x = action * seg + 20
        # print('-[INFO] Drop down at x = {}/{}'.format(x, self.WIDTH))

        return x

    def next_frame(self, action=None, generate=False):

        try:
            reward = 0
            if not self.waiting:
                self.count += 1
            self.surface.fill(pg.Color('black'))

            self.space.step(1 / self.FPS)
            self.space.debug_draw(self.draw_options)
            if generate:
                self.i = randrange(1, 5)
                self.current_fruit = create_fruit(
                    self.i, int(self.WIDTH/2), self.init_y - 10)
                self.count = 1
                self.waiting = True

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    exit()
            if not action is None and self.i and self.waiting:
                x = self.decode_action(action)
                fruit = create_fruit(self.i, x, self.init_y)
                self.fruits.append(fruit)
                ball = self.create_ball(
                    self.space, x, self.init_y, m=fruit.r//10, r=fruit.r-fruit.r % 5, i=self.i)
                self.balls.append(ball)
                self.current_fruit = None
                self.i = 0
                self.waiting = False

            reward = self.score - self.last_score
            if reward > 0:
                self.last_score = self.score

            if not self.lock:
                for i, ball in enumerate(self.balls):
                    if ball:
                        angle = ball.body.angle
                        x, y = (int(ball.body.position[0]), int(
                            ball.body.position[1]))
                        self.fruits[i].update_position(x, y, angle)
                        self.fruits[i].draw(self.surface)

            if self.current_fruit:
                self.current_fruit.draw(self.surface)

            pg.draw.aaline(self.surface, (0, 200, 0),
                           (0, self.init_y), (self.WIDTH, self.init_y), 5)

            self.show_score()

            if self.check_fail():
                self.score = 0
                self.last_score = 0
                self.reset()

            pg.display.flip()
            self.clock.tick(self.FPS)

        except Exception as e:
            print(e)
            if len(self.fruits) > len(self.balls):
                seg = len(self.fruits) - len(self.balls)
                self.fruits = self.fruits[:-seg]
            elif len(self.balls) > len(self.fruits):
                seg = len(self.balls) - len(self.fruits)
                self.balls = self.balls[:-seg]

        return self.score, reward, self.alive

    def get_feature(self, N_class=12, Keep=15):
        # 特征工程
        c_t = self.i
        # 自身类别
        feature_t = np.zeros((1, N_class + 1), dtype=np.float)
        feature_t[0, c_t] = 1.
        feature_t[0, 0] = 0.5
        feature_p = np.zeros((Keep, N_class + 1), dtype=np.float)
        Xcs = []
        Ycs = []
        Ts = []
        for i, ball in enumerate(self.balls):
            if ball:
                x = int(ball.body.position[0])
                y = int(ball.body.position[1])
                t = self.fruits[i].type
                Xcs.append(x/self.WIDTH)
                Ycs.append(y/self.HEIGHT)
                Ts.append(t)
        sorted_id = sorted_index(Ycs)
        for i, id_ in enumerate(sorted_id):
            if i == Keep:
                break
            feature_p[i, Ts[id_]] = 1.
            feature_p[i, 0] = Xcs[id_]
            feature_p[i, -1] = Ycs[id_]
            
        image = np.concatenate((feature_t, feature_p), axis=0)
        return image

    def next(self, action=None):
        self.next_frame(generate=True)
        self.last_score = self.score
        i = self.i
        image = self.get_feature()
        _, reward, _ = self.next_frame(action=action)
        for _ in range(int(self.create_time * self.FPS)):
            _, nreward, _ = self.next_frame(action=None, generate=False)
            reward += nreward
        if reward == 0:
            reward = -i
        image = np.expand_dims(image.flatten(), axis=0)
        return image.astype(np.float32), reward, not self.alive

    def run(self):

        while True:
            action = randrange(0, self.action_num)
            print('action:', action)
            image, reward, alive = self.next(action=action)
            print(image.shape)


if __name__ == '__main__':

    game = AI_Board()
    game.run()
