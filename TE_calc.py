import numpy as np
import time
from matplotlib import pyplot as plt

steps = 100000
num_lattice = 500
num_record = 50

delta = 0.01
eps = np.arange(0, 1 + delta / 2, delta)
# eps = [0]
TE_D = np.zeros(len(eps), dtype=np.float64)
TE_U = np.zeros(len(eps), dtype=np.float64)


def te_unicoupled(previous, present, r=0.2):
    if len(present) != len(previous):
        print("Length Not Matched")
        exit(1)

    N = len(previous)

    # assemble each transition
    iij_manifold = np.vstack((present[1:N], previous[1:N], previous[0: N - 1]))
    ii_manifold = np.vstack((present[1:N], previous[1:N]))
    ij_manifold = np.vstack((previous[1:N], previous[0:N - 1]))
    i_manifold = previous[1:N]

    # probability of each observed transition
    p_iij = np.zeros(N - 1, dtype=np.float64)
    p_ii = np.zeros(N - 1, dtype=np.float64)
    p_ij = np.zeros(N - 1, dtype=np.float64)
    p_i = np.zeros(N - 1, dtype=np.float64)

    for k in range(N - 1):
        iij_blowup = iij_manifold[:, k]
        iij_blowup = iij_blowup.reshape(-1, 1)
        iij_blowup = np.repeat(iij_blowup, N - 1, axis=1)
        iij_indicator = np.abs(iij_blowup - iij_manifold)
        iij_indicator = np.where(np.amax(iij_indicator, axis=0) < r, 1, 0)
        p_iij[k] = np.sum(iij_indicator) / (N - 1)

        ij_blowup = ij_manifold[:, k]
        ij_blowup = ij_blowup.reshape(-1, 1)
        ij_blowup = np.repeat(ij_blowup, N - 1, axis=1)
        ij_indicator = np.abs(ij_blowup - ij_manifold)
        ij_indicator = np.where(np.amax(ij_indicator, axis=0) < r, 1, 0)
        p_ij[k] = np.sum(ij_indicator) / (N - 1)

        ii_blowup = ii_manifold[:, k]
        ii_blowup = ii_blowup.reshape(-1, 1)
        ii_blowup = np.repeat(ii_blowup, N - 1, axis=1)
        ii_indicator = np.abs(ii_blowup - ii_manifold)
        ii_indicator = np.where(np.amax(ii_indicator, axis=0) < r, 1, 0)
        p_ii[k] = np.sum(ii_indicator) / (N - 1)

        i_blowup = i_manifold[k]
        i_blowup = np.repeat(i_blowup, N - 1)
        i_indicator = np.abs(i_blowup - i_manifold)
        i_indicator = np.where(i_indicator < r, 1, 0)
        p_i[k] = np.sum(i_indicator) / (N - 1)

    if not True:
        p_iij = p_iij / sum(p_iij)
    if not True:
        p_ij = p_ij / sum(p_ij)
        p_ii = p_ii / sum(p_ii)
        p_i = p_i / sum(p_i)

    return np.sum(p_iij / np.sum(p_iij) * np.log2(np.divide(p_iij * p_i, p_ij * p_ii)))


def ulam(x, y, epsilon):
    return 2 - (epsilon * x + (1 - epsilon) * y) ** 2


def coupled_update(N, epsilon, initials, k):
    update = np.zeros(N)
    update[0] = np.sin(k)
    # update[0] = np.random.uniform(-2, 2)
    update[1:N] = ulam(initials[0:N - 1], initials[1:N], epsilon)
    return update


def te_single(epsilon):
    start_time = time.time()
    initials = np.random.uniform(-2, 2, num_lattice)

    # run 10^5-1 times
    for k in range(steps - 1):
        update = coupled_update(num_lattice, epsilon, initials, k)
        initials = update
    update = coupled_update(num_lattice, epsilon, initials, k)

    TE_downward = 0
    TE_upward = 0

    for k in range(num_record):
        initials = update
        update = coupled_update(num_lattice, epsilon, initials, k)
        initials = update
        update = coupled_update(num_lattice, epsilon, initials, k)
        TE_downward += te_unicoupled(initials, update)
        TE_upward += te_unicoupled(initials[::-1], update[::-1])

    TE_downward /= num_record
    TE_upward /= num_record

    print("--- epsilon  =  %s ---" % epsilon)
    print("N  =  %d" % num_lattice)
    print("Transfer Entropy downward is %1.15f" % TE_downward)
    print("Transfer Entropy upward is %1.15f" % TE_upward)

    print("--- %1.15f seconds ---" % (time.time() - start_time))

    return TE_downward, TE_upward


if __name__ == "__main__":
    count = 0
    for epsilon in eps:
        TE_D[count], TE_U[count] = te_single(epsilon)
        count = count + 1

    plt.plot(np.array(eps), TE_D, label=r'$T_{J\rightarrow I}$')
    plt.plot(np.array(eps), TE_U, label=r'$T_{I\rightarrow J}$')
    plt.legend()
    plt.xlabel(r'$\epsilon$')
    plt.ylabel('Transfer Entropy')
    plt.savefig('Ulammap_{record}.png'.format(record=num_record))

    plt.show()
