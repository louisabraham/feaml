import numpy as np
import scipy.optimize as optim


from helpers import make_bounds, with_logging, LFR_compute, LFR_optim_obj


class LFR:
    def __init__(self, k=5, Ax=0.01, Ay=1, Az=50, seed=None):
        """
        Args:
            unprivileged_groups (tuple): Representation for unprivileged group.
            privileged_groups (tuple): Representation for privileged group.
            k (int, optional): Number of prototypes.
            Ax (float, optional): Input recontruction quality term weight.
            Az (float, optional): Fairness constraint term weight.
            Ay (float, optional): Output prediction error.
            seed (int, optional): Seed to make `predict` repeatable.
        """

        self.seed = seed

        self.k = k
        self.Ax = Ax
        self.Ay = Ay
        self.Az = Az

        self.learned_model = None
        self.features_dim = None

    def fit(self, X, is_protected, y, print_interval=50, **kwargs):
        """Compute the transformation parameters that leads to fair representations.

        Args:
            X: data
            is_protected: boolean array
            y: boolean array
            print_interval (int, optional): Print optimization objective value
                every print_interval iterations.
            **kwargs: will be passed to scipy.optimize.fmin_l_bfgs_b
        Returns:
            LFR: Returns self.
        """
        n, self.features_dim = X.shape
        assert is_protected.shape == (n,)
        assert is_protected.dtype == bool
        assert y.dtype == bool

        if self.seed is not None:
            np.random.seed(self.seed)

        # TODO: tweak model inits
        model_inits = np.random.uniform(
            size=self.features_dim * 2 + self.k + self.features_dim * self.k
        )
        bounds = make_bounds(self.features_dim, self.k)

        objective = with_logging(print_interval)(LFR_optim_obj)

        self.learned_model, *_ = optim.fmin_l_bfgs_b(
            objective,
            x0=model_inits,
            args=(X, is_protected, y, self.k, self.Ax, self.Ay, self.Az),
            bounds=bounds,
            approx_grad=True,
            **kwargs,
        )
        return self

    def transform(self, X, is_protected, y=None, threshold=0.5):
        """Transform the dataset using learned model parameters.

        Args:
            dataset (BinaryLabelDataset): Dataset containing labels that needs to be transformed.
            threshold(float, optional): threshold parameter used for binary label prediction.
        Returns:
            dataset (BinaryLabelDataset): Transformed Dataset.
        """
        assert self.learned_model is not None
        n, features_dim = X.shape
        assert features_dim == self.features_dim
        assert is_protected.shape == (n,)
        assert is_protected.dtype == bool

        if self.seed is not None:
            np.random.seed(self.seed)

        mapping, reconstructed, pred = LFR_compute(
            self.learned_model, X, is_protected, self.k
        )

        # transformed_labels = np.array(transformed_labels) > threshold

        # Mutated, fairer dataset with new labels
        return mapping, reconstructed, pred

    def fit_transform(self, X, is_protected, y, seed=None):
        """fit and transform methods sequentially"""

        return self.fit(X, is_protected, y).transform(X, is_protected)
