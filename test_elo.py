import torch
import pathlib
import random
import numpy
import math
from muzero import *
from self_play import *

class play_to_play:
    def __init__(self, gameplayer1, gameplayer2, config, Game, seed):
        self.gameplayer1 = gameplayer1
        self.gameplayer2 = gameplayer2
        self.config = config
        self.game = Game(seed)

        # Fix random generator seed
        numpy.random.seed(seed)
        torch.manual_seed(seed)

        # Initialize the network
        self.model1 = models.MuZeroNetwork(self.config)
        self.model1.set_weights(self.gameplayer1["weights"])
        self.model1.to(torch.device("cuda" if self.config.selfplay_on_gpu else "cpu"))
        self.model1.eval()

        self.model2 = models.MuZeroNetwork(self.config)
        self.model2.set_weights(self.gameplayer2["weights"])
        self.model2.to(torch.device("cuda" if self.config.selfplay_on_gpu else "cpu"))
        self.model2.eval()

    #random_count: 开局随机几回合
    def playgame(self, temperature, temperature_threshold, random_count = []):
        game_history = GameHistory()
        observation = self.game.reset()
        game_history.action_history.append(0)
        game_history.observation_history.append(observation)
        game_history.reward_history.append(0)
        game_history.to_play_history.append(self.game.to_play())

        done = False



        for i in random_count:
            actions = self.game.legal_actions()
            l = len(actions)
            if i > l - 1 :
                i = l - 1
            observation, reward, done = self.game.step(i)
            game_history.store_search_statistics(None, self.config.action_space)

            # Next batch
            game_history.action_history.append(i)
            game_history.observation_history.append(observation)
            game_history.reward_history.append(reward)
            game_history.to_play_history.append(self.game.to_play())

        with torch.no_grad():
            while(not done and len(game_history.action_history) <= self.config.max_moves):
                stacked_observations = game_history.get_stacked_observations(
                    -1, self.config.stacked_observations, len(self.config.action_space)
                )

                if self.game == 0:
                    root, mcts_info = MCTS(self.config).run(
                        self.model1,
                        stacked_observations,
                        self.game.legal_actions(),
                        self.game.to_play(),
                        True,
                    )
                    action = self.select_action(
                        root,
                        temperature
                        if not temperature_threshold
                        or len(game_history.action_history) < temperature_threshold
                        else 0,
                    )
                else:
                    root, mcts_info = MCTS(self.config).run(
                        self.model2,
                        stacked_observations,
                        self.game.legal_actions(),
                        self.game.to_play(),
                        True,
                    )
                    action = self.select_action(
                        root,
                        temperature
                        if not temperature_threshold
                        or len(game_history.action_history) < temperature_threshold
                        else 0,
                    )
                observation, reward, done = self.game.step(action)
                game_history.store_search_statistics(root, self.config.action_space)

                # Next batch
                game_history.action_history.append(action)
                game_history.observation_history.append(observation)
                game_history.reward_history.append(reward)
                game_history.to_play_history.append(self.game.to_play())
        self.game.render()
        who_win = self.game.env.is_finished()

        return who_win


    @staticmethod
    def select_action(node, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        The temperature is changed dynamically with the visit_softmax_temperature function
        in the config.
        """
        visit_counts = numpy.array(
            [child.visit_count for child in node.children.values()], dtype="int32"
        )
        actions = [action for action in node.children.keys()]
        if temperature == 0:
            action = actions[numpy.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = numpy.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = numpy.random.choice(actions, p=visit_count_distribution)

        return action

def calculate_elo(wins: int, draws: int, losses: int, a_elo: float) -> float:
    """
    根据B对A的胜负平局数和A的Elo，计算B的Elo等级分

    参数:
        wins (int): B对A的胜利次数
        draws (int): B对A的和棋次数
        losses (int): B对A的失败次数
        a_elo (float): A的当前Elo分数

    返回:
        int: B的Elo分数（四舍五入取整）
    """
    total_games = wins + draws + losses
    if total_games == 0:
        raise ValueError("对局总数不能为0")

    # 计算实际得分率
    actual_score = wins + 0.5 * draws
    score_rate = actual_score / total_games

    if score_rate < 0 or score_rate > 1:
        raise ValueError("得分率必须在0到1之间")

    if score_rate == 1:
        return math.inf
    elif score_rate == 0:
            return -math.inf
    # 解方程计算Elo差值
    elo_diff = -400 * math.log10((1 / score_rate) - 1)
    b_elo = a_elo + elo_diff

    return round(b_elo)

if __name__ == "__main__":
    print("可用 GPU 设备 ID:", list(range(torch.cuda.device_count())))
    print("\nWelcome to MuZero! Here's a list of games:")
    # Let user pick a game
    games = [
        filename.stem
        for filename in sorted(list((pathlib.Path.cwd() / "games").glob("*.py")))
        if filename.name != "abstract_game.py"
    ]
    for i in range(len(games)):
        print(f"{i}. {games[i]}")
    choice = input("Enter a number to choose the game: ")
    valid_inputs = [str(i) for i in range(len(games))]
    while choice not in valid_inputs:
        choice = input("Invalid input, enter a number listed above: ")

    # Initialize MuZero
    choice = int(choice)
    game_name = games[choice]
    muzero_first = MuZero(game_name)
    muzero_second = MuZero(game_name)

    print("input newer model")
    load_model_menu(muzero_first, game_name)
    print("input older model")
    load_model_menu(muzero_second, game_name)

    old_elo = int(input("entry old elo:"))
    # 实现两个不同的模型对弈100局，最后输出胜负和
    ran = numpy.random.randint(10000)
    play1 = play_to_play(muzero_first.checkpoint, muzero_second.checkpoint, muzero_first.config, muzero_first.Game, ran)
    play2 = play_to_play(muzero_second.checkpoint, muzero_first.checkpoint, muzero_first.config, muzero_first.Game, ran)

    win = 0
    draw = 0
    loss = 0

    for i in range(25):
        try:
            l = []
            l.append(random.randint(0, 63))
            l.append(random.randint(0, 63))
            l.append(random.randint(0, 63))
            l.append(random.randint(0, 63))

            result1 = play1.playgame(0, 0, l)
            if result1 == 1:
                win += 1
            elif  result1 == -1:
                loss += 1
            else:
                draw += 1
            print(f"Game:{i * 2 + 1}, win:{win}, draw:{draw}, loss:{loss}, elo1:{calculate_elo(win, draw, loss, old_elo)}, elo2:{old_elo}")

            result2 = play2.playgame(0, 0, l) * -1
            if result2 == 1:
                win += 1
            elif  result2 == -1:
                loss += 1
            else:
                draw += 1
            print(f"Game:{i * 2 + 2}, win:{win}, draw:{draw}, loss:{loss}, elo1:{calculate_elo(win, draw, loss, old_elo)}, elo2:{old_elo}")
        except AssertionError as e:
            print(f"捕获到断言错误: {e}")


    ray.shutdown()
