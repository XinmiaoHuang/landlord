# 这个文件储存一些逻辑判断函数
import numpy as np
import random
from utils import label2char, char2label

def is_pass(act):
    """
    判断函数，判断当前状态下只能出pass
    :param act:
    :return:
    """
    act_shape = act.shape
    if len(act_shape) == 1:
        act = np.expand_dims(act, -1)
        act_shape = act.shape
    if act_shape[-1] > 2:
        act_1 = act[:, -1]
        act_2 = act[:, -2]
        if (act_1 == 0).all() and (act_2 == 0).all():
            return True
        else:
            return False
    else:
        act_1 = act[:, -1]
        return (act_1 == 0).all()


def random_out():
    """
    从已有的手牌里面随机出牌
    :param self:
    :return:
    """
    res = np.zeros(shape=(15, 1))
    rand_act = random.randint(1, 308)
    rand_str = label2char[rand_act]
    #需要带牌
    if rand_str[-1] not in ['P','A','2','3','4','5','6','7','8','9','T','J','Q','K','X','D']:
        rand_kick = get_random_kicker(rand_act)
        for word in rand_str[:-1]:
            number = char2label[word] - 1
            res[number] += 1
        for kick in rand_kick:
            number = char2label[kick] - 1
            res[number] += 1
        return res
    else:
        for word in rand_str:
            number = char2label[word] - 1
            res[number] += 1
        return res


def judge_handtype(prev_action, action):
    """
    判断出的两个牌是否为同个类型的牌且后手比前一手大
    :param self:
    :param prev_action:
    :param action:
    :return:
    """
    sorted_prev = np.sort(prev_action)
    sorted_action = np.sort(action)
    max_item = np.where(action == sorted_action[-1])
    if (sorted_action == sorted_prev).all() == True:
        if np.argmax(prev_action) < np.argmax(action):
            return True
        else:
            print("Cards not bigger than prev.")
            return False
    else:
        print("Type dont't match.")
        return False


def is_legal(prev_act, act_vec, state, judge_prev=True):
    """
    判断是否是合法出牌
    :param self:
    :param prev_act:
    :param act_vec:
    :return:
    """
    def transform(act):
        vec_p = np.zeros(shape=(15,))
        count = 0
        i = 0
        while i <= 48:
            sum = 0
            for k in range(4):
                sum += act[i+k]
            vec_p[count] = sum
            count += 1
            i = count * 4
        vec_p[13] = act[52]
        vec_p[14] = act[53]
        return vec_p
    if len(state.shape) == 1:
        state = np.expand_dims(state, -1)
    hand = state[:, 0]
    prev_act = transform(prev_act)
    t_vec = transform(act_vec)
    # 鬼牌只有一张, 且出牌不能大于四张
    if t_vec[14] > 1 or t_vec[13] > 1 or (t_vec[:13] > 4).any():
        print("Cards number out of bound.")
        return False
    # if np.argmax(prev_act) >= np.argmax(t_vec):
    #     print("Totall small cards.")
    #     return False

    if judge_prev:
        if not judge_handtype(prev_act, t_vec):
            return False

    if ((hand - act_vec) < 0).any():
        print("Play card that don't have.")
        return False
    return True


def isbomb(action):
    if np.max(action) == 4 and action.sum() == 4:
        return True
    else:
        return False
