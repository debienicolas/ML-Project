import open_spiel.python.algorithms.mcts as mcts
import time
import pyspiel



class MCTS(mcts.MCTSBot):

    def step_with_policy_training(self, state,temp):
        """Returns bot's policy and action at given state. Also returns the policy for training"""
        game = state.get_game()
        self.max_utility = game.max_utility()

        t1 = time.time()
        root = self.mcts_search(state)

        best = root.best_child()

        if self.verbose:
            seconds = time.time() - t1
            print("Finished {} sims in {:.3f} secs, {:.1f} sims/s".format(
                root.explore_count, seconds, root.explore_count / seconds))
            print("Root:")
            print(root.to_str(state))
            print("Children:")
            print(root.children_str(state))
            if best.children:
                chosen_state = state.clone()
                chosen_state.apply_action(best.action)
                print("Children of chosen:")
                print(best.children_str(chosen_state))

        mcts_action = best.action

        total_counts = sum([child.explore_count for child in root.children])

        if temp == 0:
            max_index = mcts_action
            return [1.0 if i == max_index else 0.0 for i in range(state.num_distinct_actions())], mcts_action


        ### checken of dit klopt ###
        
        policy = [(child.action,  child.explore_count/total_counts) for child in root.children]
        # add illegal move with 0 probability
        for i in range(state.num_distinct_actions()):
            if i not in [x[0] for x in policy]:
                policy.append((i,0.0))

        policy = sorted(policy, key=lambda x: x[0])
        
        policy = [x[1] for x in policy]

        return policy, mcts_action
    

