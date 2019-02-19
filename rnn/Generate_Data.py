import random


def generate_data(difficulty, noise, length=10000, change=0.01, trend_value=5):
    """
    this function is generating data in various difficulties for architecture testing.
    :param difficulty: the difficulty of the output data, an integer when 1 is the easiest.
    :param noise: how much noise to add to the data, a number between 0 to 1
    :param length: the length of the output data
    :param change: what is the change rate when it happens
    :param trend_value: how much the generator "wants" to continue new trends
    :return: data a list of numbers
    """
    data = [1]
    if difficulty == 0:
        for i in range(length-1):
            if (i % (trend_value * 2)) < trend_value:
                data.append(data[i] * (1 + change *(1 + noise * random.uniform(-1,1))))
            else:
                data.append(data[i] * (1 - change *(1 + noise * random.uniform(-1,1))))
        return data

    # up_sequences = [0]*difficulty
    # same_sequences = [0]*difficulty
    # down_sequences = [0]*difficulty
    # up_chance = 1/3
    # same_chance = 1/3
    # down_chance = 1/3
    # trend = 1
    # for i in range(length-1):
    #     ran = random.uniform(0, 1)
    #     if ran < down_chance:
    #         data.append(data[i]*(1 - change * (1 + noise * random.uniform(-1, 1))))
    #         trend = -1
    #     elif ran < (down_chance + same_chance):
    #         data.append(data[i]*(1 + change * (noise * random.uniform(-1, 1))))
    #     else:
    #         data.append(data[i]*(1 + change * (1 + noise * random.uniform(-1, 1))))
    #         trend = 1
    #     pUp = 1
    #     pDown = 1
    #     for i in difficulty:
    #
    #
    #
    #
