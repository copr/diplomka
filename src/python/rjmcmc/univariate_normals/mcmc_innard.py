import copy

import numpy as np

def mcmc_step(previous_sample, stationary, proposal, proposal_sampler):
    '''
    Vnitrek mcmc algoritmu, prakticky to co se deje v jedne 
    iteraci tady te verze hastingse
    '''
    proposal_sample = proposal_sampler(previous_sample)
    local_previous_sample = copy.copy(previous_sample)
    final_sample = copy.copy(previous_sample)
    for j, x in enumerate(proposal_sample):
        local_previous_sample[j] = x
        down = np.prod([stationary(previous_sample),
                       proposal(local_previous_sample, previous_sample)])
        up = np.prod([stationary(local_previous_sample),
                      proposal(previous_sample, local_previous_sample)])

        u = np.random.uniform()

        # if (j == 1 or j == 3) and x > 10:
        #     print()
        #     print(up, down)
        #     print(previous_sample)
        #     print(local_previous_sample)

        if u < up/down:
            final_sample[j] = x
        else:
            local_previous_sample[j] = previous_sample[j]
    return final_sample
