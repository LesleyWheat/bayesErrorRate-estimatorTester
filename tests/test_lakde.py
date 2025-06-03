# Imports
import pytest
import os
# --------------------------------------------------------------------
pathSave = os.path.join(os.getcwd(), 'local', 'testOutput')
LOW = -4
HIGH = 4
SIZES = 1000
SPREAD = 0.1
# --------------------------------------------------------------------

# Dataset iterator


def inf_train_gen(data, rng=None, batch_size=200):
    import numpy as np
    import sklearn
    import sklearn.datasets
    from sklearn.utils import shuffle as util_shuffle

    if rng is None:
        rng = np.random.default_rng()

    if data == "swissroll":
        data = sklearn.datasets.make_swiss_roll(
            n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        return data

    elif data == "circles":
        data = sklearn.datasets.make_circles(
            n_samples=batch_size, factor=0.5, noise=0.08
        )[0]
        data = data.astype("float32")
        data *= 3
        return data

    elif data == "rings":
        n_samples4 = n_samples3 = n_samples2 = batch_size // 4
        n_samples1 = batch_size - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, we set endpoint=False
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25

        X = (
            np.vstack(
                [
                    np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
                    np.hstack([circ4_y, circ3_y, circ2_y, circ1_y]),
                ]
            ).T
            * 3.0
        )
        X = util_shuffle(X, random_state=rng)

        # Add noise
        X = X + rng.normal(scale=0.08, size=X.shape)

        return X.astype("float32")

    elif data == "moons":
        data = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)[0]
        data = data.astype("float32")
        data = data * 2 + np.array([-1, -0.2])
        return data

    elif data == "8gaussians":
        scale = 4.0
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        ]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(batch_size):
            point = rng.random(size=2) * 0.5
            idx = rng.integers(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        return dataset

    elif data == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = batch_size // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = rng.normal(size=(num_classes * num_per_class, 2)) * np.array(
            [radial_std, tangential_std]
        )
        features[:, 0] += 1.0
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack(
            [np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)]
        )
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        return 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations))

    elif data == "2spirals":
        n = np.sqrt(rng.random((batch_size // 2, 1))) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + rng.random((batch_size // 2, 1)) * 0.5
        d1y = np.sin(n) * n + rng.random((batch_size // 2, 1)) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += rng.normal(size=x.shape) * 0.1
        return x

    elif data == "checkerboard":
        x1 = rng.random(batch_size) * 4 - 2
        x2_ = rng.random(batch_size) - rng.integers(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        return np.concatenate([x1[:, None], x2[:, None]], 1) * 2

    elif data == "line":
        x = rng.random(batch_size) * 5 - 2.5
        y = x
        return np.stack((x, y), 1)
    elif data == "cos":
        x = rng.random(batch_size) * 5 - 2.5
        y = np.sin(x) * 2.5
        return np.stack((x, y), 1)
    else:
        return inf_train_gen("8gaussians", rng, batch_size)

def plot_density_colormesh_score(
    scoreFunc,
    X,
    cmap="viridis",
    show_points=False,
    low=None,
    high=None,
    sizes=None,
    norm=None,
    ax=None,
):
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    import numpy as np
    import torch

    fig = None
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca()

    if low is None:
        low = (LOW, LOW)
    if high is None:
        high = (HIGH, HIGH)
    if sizes is None:
        sizes = (SIZES, SIZES)

    xi = torch.linspace(low[0], high[1], sizes[0])
    yi = torch.linspace(low[1], high[1], sizes[1])

    xx, yy = np.meshgrid(xi.numpy(), yi.numpy())
    xx = torch.from_numpy(xx)
    yy = torch.from_numpy(yy)

    Y = torch.stack([xx.flatten(), yy.flatten()], dim=-1).to(X.device)

    # KDE
    zi = np.exp(scoreFunc(Y).reshape((xx.shape)))

    #zi = kde.log_pred_density(X, Y).exp().reshape_as(xx).cpu()
    xi, yi = xi.numpy(), yi.numpy()

    ax.pcolormesh(xi, yi, zi, norm=norm, cmap=cmap)

    if show_points:
        Xn = X.cpu().numpy()
        ax.scatter(Xn[:, 0], Xn[:, 1], s=3, color="white")

    ax.set_xticks([])
    ax.set_yticks([])

    if fig is not None:
        fig.tight_layout()
        fig.show()
# --------------------------------------------------------------------

#@pytest.mark.lakde
def test_lakde1():
    import os
    # use cpu
    os.environ['LAKDE_NO_EXT'] = '1'

    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from lakde import SharedFullKDE, LocalKNearestKDE
    from lakde.callbacks import LikelihoodCallback
    # from datasets import inf_train_gen

    from lakde.experiments.k_radius_estimator import KNNDensityEstimator, correct_knn_density
    from lakde.experiments.utils import plot_density_colormesh, plot_density_colormesh_kde

    # Save path
    pathSave = os.path.abspath(os.path.join(
        os.getcwd(), 'local', 'testOutput'))
    name = 'pcaTest'
    os.makedirs(pathSave, exist_ok=True)

    datasets = ["pinwheel", "2spirals", "checkerboard"]
    norms = {"pinwheel": 0.25, "2spirals": 0.15, "checkerboard": 0.06}
    knn_k = {"pinwheel": 10, "2spirals": 10, "checkerboard": 10}
    knnkde_k = {"pinwheel": 38, "2spirals": 22, "checkerboard": 29}

    f, allaxes = plt.subplots(3, 4, figsize=(8, 6), dpi=200)
    first = True

    for dataset, axes in zip(datasets, allaxes):
        rng = np.random.default_rng(0)
        train_data = inf_train_gen(dataset, rng, batch_size=400)
        # val_data = inf_train_gen(dataset, rng, batch_size=100)

        X = torch.from_numpy(train_data).float()  # .cuda()

        knn = KNNDensityEstimator(k=knn_k[dataset])
        fullkde = SharedFullKDE(verbose=False)
        knnkde = LocalKNearestKDE(
        k_or_knn_graph=knnkde_k[dataset], verbose=False)

        for kde in [fullkde, knnkde]:
            # early_stop = LikelihoodCallback(
            #     #torch.from_numpy(val_data).float().cuda(),
            #     torch.from_numpy(val_data).float(),
            #     report_on="iteration",
            #     report_every=1,
            #     rtol=0,  # stop on overfit
            #     verbose=True,
            # )
            # kde.fit(X, iterations=10, callbacks=[early_stop])
            kde.fit(X, iterations=10)

        norm = plt.Normalize(vmin=0, vmax=norms[dataset], clip=True)

        knn.const = correct_knn_density(knn, X)
        plot_density_colormesh(knn, X, norm=norm, ax=axes[0])
        axes[0].set_title("KNN" if first else None)

        plot_density_colormesh(fullkde, X, norm=norm, ax=axes[1])
        axes[1].set_title("VB-Full-KDE" if first else None)

        plot_density_colormesh(knnkde, X, norm=norm, ax=axes[2])
        axes[2].set_title("VB-KNN-KDE" if first else None)

        # Basic KDE
        # plot_density_colormesh(knnkde, X, norm=norm, ax=axes[3])
        plot_density_colormesh_kde(X, norm=norm, ax=axes[3])
        axes[3].set_title("KDE" if first else None)

        first = False
        print(norm.vmin, norm.vmax)

        plt.tight_layout()
        plt.savefig(os.path.join(pathSave, "toy_comparison_0.4k.png"))



@pytest.mark.lakde
def test_lakde2():
    import os
    # use cpu
    os.environ['LAKDE_NO_EXT'] = '1'

    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from lakde import SharedFullKDE, LocalKNearestKDE
    from lakde.callbacks import LikelihoodCallback
    # from datasets import inf_train_gen

    from lakde.experiments.k_radius_estimator import KNNDensityEstimator, correct_knn_density
    from lakde.experiments.utils import plot_density_colormesh, plot_density_colormesh_kde

    from distributions import oneBlob, twoBlob, threeBlob
    from distributions import nBlobOverSphere, sphereAndBlob

    os.makedirs(pathSave, exist_ok=True)
    num = 100
    seed = 0
    rng = np.random.default_rng(seed=seed)

    # dist
    distA = oneBlob(2, 0, 0.1)
    distB = twoBlob(2, 1, 0.1)
    distC = threeBlob(2, 2, 0.1)
    distD = nBlobOverSphere(2, 2, 0.1, nBlobs=100, slackInN=5)
    distE = sphereAndBlob(2, 3, 0.1, nBlobs=100, slackInN=5)

    distList = [distA, distB, distC, distD, distE]

    f, allaxes = plt.subplots(len(distList), 5, figsize=(8, 8), dpi=200)
    loopCount=0

    for dist, axes in zip(distList, allaxes):
        first = True if loopCount == 0 else False
        rng = np.random.default_rng(seed=seed)
        # train_data = inf_train_gen(dataset, rng, batch_size=num)
        # val_data = inf_train_gen(dataset, rng, batch_size=10_000)
        dataset = dist.generate(rng, num)

        X = torch.from_numpy(dataset).float()  # .cuda()

        knn = KNNDensityEstimator(k=10)
        fullkde = SharedFullKDE(verbose=False)
        knnkde = LocalKNearestKDE(k_or_knn_graph=20, verbose=False)

        for kde in [fullkde, knnkde]:
            # early_stop = LikelihoodCallback(
            #     #torch.from_numpy(val_data).float().cuda(),
            #     torch.from_numpy(val_data).float(),
            #     report_on="iteration",
            #     report_every=1,
            #     rtol=0,  # stop on overfit
            #     verbose=True,
            # )
            kde.fit(X, iterations=10)

        norm = plt.Normalize(vmin=0, vmax=1, clip=True)

        knn.const = correct_knn_density(knn, X)
        plot_density_colormesh(knn, X, norm=norm, ax=axes[0])
        axes[0].set_title("KNN" if first else None)

        plot_density_colormesh(fullkde, X, norm=norm, ax=axes[1])
        axes[1].set_title("VB-Full-KDE" if first else None)

        plot_density_colormesh(knnkde, X, norm=norm, ax=axes[2])
        axes[2].set_title("VB-KNN-KDE" if first else None)

        # Basic KDE
        # plot_density_colormesh(knnkde, X, norm=norm, ax=axes[3])
        plot_density_colormesh_kde(X, norm=norm, ax=axes[3])
        axes[3].set_title("KDE" if first else None)

        plot_density_colormesh_score(dist.logPdf, X, norm=norm, ax=axes[4])
        axes[4].set_title("True Probability" if first else None)

        loopCount=loopCount+1
        print(norm.vmin, norm.vmax)

        plt.tight_layout()
        plt.savefig(os.path.join(pathSave, f"toy_comparison_dist_{num}n.png"))

@pytest.mark.lakde
def test_lakde3():
    import os
    import time
    # use cpu
    os.environ['LAKDE_NO_EXT'] = '1'

    import torch
    import numpy as np

    from .utils import dist_one
    from lakde import SharedFullKDE

    iterations = 10
    classA, classB = dist_one(nClassSamples=1000, seperate=True)

    nA = classA.shape[0]
    nB = classB.shape[0]

    fullkde_A = SharedFullKDE(verbose=True)
    fullkde_B = SharedFullKDE(verbose=True)

    A = torch.from_numpy(classA).float()
    B = torch.from_numpy(classB).float()

    fullkde_A.fit(A, iterations=1)
    print(fullkde_A.state_dict())
    fullkde_A.fit(A, iterations=iterations)
    print(fullkde_A.state_dict())
    fullkde_A.init_summaries_(A)
    fullkde_A.init_summaries_(B)

    print((f'{torch.mean(fullkde_A.data_log_likelihood(A)).exp()} '
           f'{torch.mean(fullkde_A.data_log_likelihood(B)).exp()}'))

    print((f'{torch.mean(fullkde_A.data_log_likelihood_no_rnm(A).exp())} '
           f'{torch.mean(fullkde_A.data_log_likelihood_no_rnm(B).exp())}'))

    fullkde_B.fit(B, iterations=iterations)
    fullkde_B.init_summaries_(B)
    fullkde_B.init_summaries_(A)

    p_b_a = fullkde_A.log_pred_density(A, B).exp().cpu()
    p_a_b = fullkde_B.log_pred_density(B, A).exp().cpu()

    p_a_a = np.zeros(nA)
    p_a_a2 = torch.zeros(nA)
    p_b_b = np.zeros(nB)
    p_b_b2 = torch.zeros(nB)

    start = time.time()

    for i in range(nA):
        x = torch.from_numpy(classA[i, :].reshape((1, -1))).float()
        A_LOO = torch.from_numpy(np.delete(classA, (i), axis=0)).float()

        p_a_a[i] = fullkde_A.log_pred_density(A_LOO, x).exp().cpu()

    t_np = time.time() - start
    start = time.time()

    for i in range(nA):
        A_LOO = A[torch.arange(1, A.shape[0]+1) != i, ...]
        x = torch.reshape(A[i, ...], (1, A.shape[1]))

        p_a_a2[i] = fullkde_A.log_pred_density(A_LOO, x).exp().cpu()

    t_ten = time.time() - start

    print(
        f'Numpy: time {t_np} val {np.average(p_a_a)} Tensor: time {t_ten} val {torch.mean(p_a_a2)}')

    start = time.time()

    for i in range(nB):
        x = torch.from_numpy(classB[i, :].reshape((1, -1))).float()
        B_LOO = torch.from_numpy(np.delete(classB, (i), axis=0)).float()
        p_b_b[i] = fullkde_B.log_pred_density(B_LOO, x).exp().cpu()

    t_np = time.time() - start
    start = time.time()

    for i in range(nB):
        B_LOO = B[torch.arange(1, B.shape[0]+1) != i, ...]
        x = torch.reshape(B[i, ...], (1, B.shape[1]))

        p_b_b2[i] = fullkde_B.log_pred_density(B_LOO, x).exp().cpu()

    t_ten = time.time() - start

    print(
        f'Numpy: time {t_np} val {np.average(p_b_b)} Tensor: time {t_ten} val {torch.mean(p_b_b2)}')

    a_b = np.average(p_a_b)
    a_a = np.average(p_a_a)
    b_b = np.average(p_b_b)
    b_a = np.average(p_b_a)

    a_a2 = torch.mean(p_a_a2)
    b_b2 = torch.mean(p_b_b2)

    j1 = (1/2)*(a_b + b_a)/(a_a + b_b)
    j2 = (1/2)*(a_b + b_a)/(a_a2 + b_b2)

    print(f'a_b: {a_b} a_a: {a_a} b_b: {b_b} b_a: {b_a} j1 {j1} j2 {j2}')

@pytest.mark.lakde
@pytest.mark.gpu
def test_lakde4():
    import os
    from .app import checkGpu

    checkGpu()

    # use cpu
    os.environ['LAKDE_NO_EXT'] = '0'

    import torch
    import numpy as np

    from .utils import dist_one
    from lakde import SharedFullKDE

    iterations = 10
    classA, classB = dist_one(seperate=True)

    nA = classA.shape[0]
    nB = classB.shape[0]

    fullkde_A = SharedFullKDE(verbose=True)
    fullkde_B = SharedFullKDE(verbose=True)

    A = torch.from_numpy(classA).float().cuda()
    B = torch.from_numpy(classB).float().cuda()

    fullkde_A.fit(A, iterations=1)
    print(fullkde_A.state_dict())
    fullkde_A.fit(A, iterations=iterations-1)
    print(fullkde_A.state_dict())
    fullkde_A.init_summaries_(A)
    fullkde_A.init_summaries_(B)

    print((f'{torch.mean(fullkde_A.data_log_likelihood(A)).exp()} '
           f'{torch.mean(fullkde_A.data_log_likelihood(B)).exp()}'))

    print((f'{torch.mean(fullkde_A.data_log_likelihood_no_rnm(A).exp())} '
           f'{torch.mean(fullkde_A.data_log_likelihood_no_rnm(B).exp())}'))

    fullkde_B.fit(B, iterations=iterations)
    fullkde_B.init_summaries_(B)
    fullkde_B.init_summaries_(A)

    p_b_a = fullkde_A.log_pred_density(A, B).exp()
    p_a_b = fullkde_B.log_pred_density(B, A).exp()

    p_a_a = torch.zeros(nA)
    p_b_b = torch.zeros(nB)

    for i in range(nA):
        x = torch.from_numpy(classA[i, :].reshape((1, -1))).float().cuda()
        A_LOO = torch.from_numpy(np.delete(classA, (i), axis=0)).float().cuda()

        new_x = A[torch.arange(1, A.shape[0]+1) != i, ...]

        p_a_a[i] = fullkde_A.log_pred_density(A_LOO, x).exp().cuda()

    for i in range(nB):
        x = torch.from_numpy(classB[i, :].reshape((1, -1))).float().cuda()
        B_LOO = torch.from_numpy(np.delete(classB, (i), axis=0)).float().cuda()
        p_b_b[i] = fullkde_B.log_pred_density(B_LOO, x).exp().cuda()

    a_b = torch.mean(p_a_b)
    a_a = torch.mean(p_a_a)
    b_b = torch.mean(p_b_b)
    b_a = torch.mean(p_b_a)

    print(f'a_b: {a_b} a_a: {a_a} b_b: {b_b} b_a: {b_a}')

    j2 = (1/2)*(a_b + b_a)/(a_a + b_b)
