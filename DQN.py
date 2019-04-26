import numpy as np
from collections import deque
import keras
import tensorflow as tf
from keras.layers import Concatenate, Input, Lambda, Masking, LSTM, Dense
import keras.backend as K
from keras.models import Model
import random
from utils import *
from logic import *
from keras.preprocessing.sequence import  pad_sequences


def build_network():
    """
    预测网络
    :return:
    """
    o_state = Input(shape=(54, 62))
    o_kick_in = Input(shape=(28,))
    o_act_in = Input(shape=(309,))

    x = Masking(mask_value=-1)(o_state)
    x = LSTM(512)(x)
    x = Dense(512)(x)
    actions = Dense(309)(x)
    kicks = Dense(28)(x)

    model = Model(inputs=[o_state, o_act_in, o_kick_in], outputs=[actions, kicks])
    model.summary()
    model.compile(optimizer=keras.optimizers.adam(lr=0.002),
                  loss=loss)
    return model


def loss(y_true, y_pred):
    act_loss = tf.reduce_mean(K.square(y_true[0] - y_pred[0]))
    kick_loss = tf.reduce_mean(K.square(y_true[1] - y_pred[1]))
    return act_loss + kick_loss


model = build_network()


class DQN():
    def __init__(self,position, state):
        self.gamma = 0.9
        self.epsilon = 0.3
        self.memory = deque()
        self.max_memory = 5000
        self.batch_size = 12
        self.position = position
        self.state = state
        self.kick_dims = 28
        self.max_len = 62

    def get_kicker(self, kick_type=None, is_random=True):
        """
        随机打出一张带牌,28x1的多标签向量
        :param kick_type:
        :param is_random:
        :return:
        """
        res = np.zeros(shape=(self.kick_dims,))
        # 三带一
        if kick_type == '!' or kick_type == '(':
            rand_num = random.randint(1, 13)
            res[rand_num-1] += 1
        # 三带二：
        elif kick_type == '@' or kick_type == ')':
            rand_num = random.randint(16, 28)
            res[rand_num-1] += 1
        # 二联飞机带单：
        elif kick_type == '#':
            rand_num = random.sample(range(1, 13), 2)
            for num in rand_num:
                res[num-1] += 1
        # 三联飞机带单
        elif kick_type == '$':
            rand_num = random.sample(range(1, 13), 3)
            for num in rand_num:
                res[num-1] += 1
        # 四联飞机带单
        elif kick_type == '%':
            rand_num = random.sample(range(1, 13), 4)
            for num in rand_num:
                res[num-1] += 1
        # 五联飞机带单
        elif kick_type == '^':
            rand_num = random.sample(range(1, 13), 5)
            for num in rand_num:
                res[num-1] += 1
        # 二联飞机带双
        elif kick_type == '&':
            rand_num = random.sample(range(16, 28), 2)
            for num in rand_num:
                res[num-1] += 1
        # 三联飞机带双
        elif kick_type == '*':
            rand_num = random.sample(range(16, 28), 3)
            for num in rand_num:
                res[num-1] += 1
        # 四联飞机带双
        elif kick_type == '?':
            rand_num = random.sample(range(16, 28), 4)
            for num in rand_num:
                res[num-1] += 1
        return res

    def random_aciton(self, has_prev=True):
        """
        随机打出一手牌 54x1的向量形式
        :param has_prev:是否有上家
        :return:打出的手牌，54x1, one-hot向量形式， 动作向量对应标号，带牌对应标号
         kick_set 保存带牌编号的列表
        """
        act_num = random.randint(1, 308)
        act_str = label2char[act_num]
        if act_str[-1] not in ['P', 'A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'X', 'D']:
            kicker = self.get_kicker(kick_type=act_str[-1])
        else:
            kicker = np.zeros(shape=(self.kick_dims,))
        vec = np.zeros(shape=(309,))
        vec[act_num] += 1
        act_vec= convert2vec(vec, kicker, self.state)
        return act_vec, act_num, kicker

    def model_predict(self, state_in, act_in, kick_in):
        """

        :param state_in:
        :param act_in:
        :param kick_in:
        :return: one hot形式的动作向量和带牌向量
        """
        state = state_in
        if len(state.shape) == 1:
            state = np.expand_dims(state, axis=-1)
        if len(state.shape) == 2:
            state = np.expand_dims(state, axis=0)
        shape_ = state.shape
        if shape_[2] != 62:
            extra = np.zeros(shape=(shape_[0], 54, 62 - shape_[2]))
            state = np.concatenate((state, extra-1), axis=-1)
        act, kick = model.predict([state, act_in, kick_in])
        print("pos: {}, Q_net_prediction: Q_act: {}, kick_act: {}".format(self.position, np.max(act), np.max(kick)))
        return act[0], kick[0]

    def e_greedy_action(self):
        """
        :return: (54x1向量的动作向量， 动作向量对应的标号， 带牌对应的标号
        """
        if random.random() < self.epsilon:
            #self.epsilon *= 0.95
            print("random out.")
            return self.random_aciton()
        else:
            act, kick = self.model_predict(self.state, np.zeros((1, 309,)), np.zeros((1, 28,)))
            act_num = np.argmax(act)
            act_vec = convert2vec(act, kick, self.state)
            return act_vec, act_num, kick

    def do_move(self, action):
        # 如果出了自己没有的牌，判定为游戏失败
        # 全部pass，此时出任何牌都算出牌成功
        if len(self.state.shape) == 1 or is_pass(self.state):
            if (action == 0).all():  # 上家全部pass 此时不能出pass
                print("Can't pass now.")
                next_state = self.update(action, True)
                return next_state, -5, True
            else:
                if is_legal(np.zeros(shape=(54,)), action, self.state, judge_prev=False):
                    next_state = self.update(action, True)
                    return next_state, 20, False
                else:
                    next_state = self.update(action, True)
                    return next_state, -5, True
        else:
            if (self.state[:, -1] == 0).all():
                prev_act = self.state[:, -2]  # 上一手牌
            else:
                prev_act = self.state[:, -1]
            if (action == 0).all():
                next_state = self.update(action, True)
                return next_state, -1, False
            if is_legal(prev_act, action, self.state) or isbomb(action):
                next_state = self.update(action, True)
                return next_state, 30, False  # 出对牌判为出牌成功
            else:
                next_state = self.update(action, True)
                return next_state, -4, True

    def update(self, action, is_myturn=True):
        action = np.expand_dims(action, -1)
        if len(self.state.shape) == 1:
            self.state = np.expand_dims(self.state, -1)
        new_state = np.concatenate((self.state, action), -1)
        if not is_myturn:
            return new_state
        else:
            for index, num in enumerate(action):
                new_state[index, 0] -= num
            return new_state

    def train(self, action, reward, next_state, done, kick_vec):
        shape_ = self.state.shape
        if len(shape_) == 1:
            state = np.expand_dims(self.state, -1)
        else:
            state = self.state
        shape_ = state.shape
        if shape_[1] != 62:
            extra_ = np.zeros(shape=(54, 62-shape_[1]))
            state = np.concatenate((state, extra_-1), axis=-1)

        shape_ = next_state.shape
        if len(shape_) == 1:
            n_state = np.expand_dims(next_state, -1)
        else:
            n_state = next_state
        shape_ = n_state.shape
        if shape_[1] != 62:
            extra_ = np.zeros(shape=(54, 62-shape_[1]))
            n_state = np.concatenate((n_state, extra_-1), axis=-1)

        self.memory.append((state, action, reward, n_state, done, kick_vec))
        if len(self.memory) > self.max_memory:
            self.memory.popleft()
        if len(self.memory) > self.batch_size:
            for i in range(1):
                mini_batch = random.sample(self.memory, self.batch_size)
                state_batch = np.array([np.array(block[0]) for block in mini_batch])
                action_batch = np.array([np.array(block[1]) for block in mini_batch])
                reward_batch = np.array([np.array(block[2]) for block in mini_batch])
                next_state_batch = np.array([np.array(block[3]) for block in mini_batch])
                kick_bacth = np.array([np.array(block[5]) for block in mini_batch])

                Q_act, Q_kick = self.model_predict(next_state_batch,
                                                   np.zeros((self.batch_size, 309,)),
                                                   np.zeros((self.batch_size, 28,)))
                y_batch = []
                for i in range(self.batch_size):
                    if mini_batch[i][4]:
                        y_batch.append([reward_batch[i], reward_batch[i]])
                    else:
                        y_batch.append([reward_batch[i] + self.gamma * (np.max(Q_act[i])),
                                        reward_batch[i] + self.gamma * (np.max(Q_kick[i]))])

                actions = np.zeros(shape=(self.batch_size, 309,))
                for i in range(self.batch_size):
                    actions[i, action_batch[i]] += 1

                act_gt = np.zeros(shape=(self.batch_size, 309,))
                for idx, item in enumerate(actions):
                    act_gt[idx] = item * y_batch[idx][0]

                kick_gt = np.zeros(shape=(self.batch_size, 28,))
                for idx, item in enumerate(kick_bacth):
                    kick_gt[idx] = item * y_batch[idx][1]

                # print("Traing ont batch Q_network......")
                #self.model.fit([state_batch, actions, kick_bacth], [act_gt, kick_gt])
                model.train_on_batch([state_batch, actions, kick_bacth], [act_gt, kick_gt])

    def save_model(self):
        model.save_weights('./lord_model_new.h5')


def reset(data):
    print("------------------------------Reseting hand_cards......------------------------------")
    game = (random.sample(data, 1))[0].split(";")
    a, b, c, lord_card = game[0], game[1], game[2], game[3]
    a += lord_card
    new_a, new_b, new_c = one2four([a, b, c])
    return new_a, new_b, new_c


def round(player_1, player_2, player_3):
    action, act_num, kick_set = player_1.e_greedy_action()
    print("#####################  pos:{}, play:{}  ##########################".format(player_1.position, vec2str(action)))
    next_state, reward, done = player_1.do_move(action)
    player_1.train(act_num, reward, next_state, done, kick_set)
    player_1.state = next_state
    player_2.state = player_2.update(action, is_myturn=False)
    player_3.state = player_3.update(action, False)

    # 游戏结束
    if done:
        return True
    else:
        return False


if __name__ == '__main__':
    data = []
    f = open('god_vision_v4.txt')
    for line in f.readlines():
        item = line.split('#')
        data.append(item[0][:-1])
    card_lord, card_1, card_2 = reset(data)
    lord = DQN(position=0, state=card_lord)     # 默认1号为地主，以后有机会在考虑其他情况
    peasant_1 = DQN(1, card_1)
    peasant_2 = DQN(2, card_2)
    model.load_weights('./lord_model.h5')
    epochs = 1000
    steps = 400
    round_sum = 0
    for epoch in range(epochs):
        round_count = 0
        for step in range(steps):
            round_count += 1
            print("epoch: {},  step: {}".format(epoch, step))
            is_over_l = round(lord, peasant_1, peasant_2)
            if is_over_l:
                print("Positon 0 failed.")
                break
            round_count += 1
            is_over_p1 = round(peasant_1, peasant_2, lord)
            if is_over_p1:
                print("Positon 1 failed.")
                break
            round_count += 1
            is_over_p2 = round(peasant_2, lord, peasant_1)
            if is_over_p2:
                print("Positon 2 failed.")
                break
            print("round over.")
        lord.state, peasant_1.state, peasant_2.state = reset(data)
        round_sum += round_count
    lord.save_model()
    print("average round: ", round_sum / epochs)
    f.close()
