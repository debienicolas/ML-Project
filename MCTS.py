import open_spiel.python.algorithms.mcts as mcts
import time



class MCTS(mcts.MCTSBot):

    def step_with_policy_training(self, state):
        """Returns bot's policy and action at given state. Also returns the policy for training"""
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

        policy = [(child.action, child.prior) for child in root.children]
        # add illegal move with 0 probability
        for i in range(state.num_distinct_actions()):
            if i not in [x[0] for x in policy]:
                policy.append((i,0.0))

        policy = sorted(policy, key=lambda x: x[0])
        
        policy = [x[1] for x in policy]

        return policy, mcts_action
    
