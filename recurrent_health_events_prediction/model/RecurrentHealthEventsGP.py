import torch
import gpytorch
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional, List, Dict
from sklearn.metrics import accuracy_score, roc_auc_score
from enum import Enum


class KernelType(Enum):
    RBF = "RBF"
    MATERN = "MATERN"


class GPClassifier(gpytorch.models.ApproximateGP):
    def __init__(
        self,
        inducing_points,
        time_idx: Optional[List[int]] = None,
        kernel_type: KernelType = KernelType.RBF,
    ):
        # Variational distribution and strategy
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()

        # Identify time and feature indices
        self.time_idx = time_idx or [-1]  # Use last column as time by default
        self.feature_idx = [
            i for i in range(inducing_points.shape[1]) if i not in self.time_idx
        ]

        if kernel_type == KernelType.RBF:
            # Radial Basis Function (RBF) kernel for time and others
            self.kernel_time = gpytorch.kernels.RBFKernel(active_dims=time_idx)
            self.kernel_others = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    ard_num_dims=len(self.feature_idx),
                    active_dims=self.feature_idx
                )
            )
        elif kernel_type == KernelType.MATERN:
            # Matern kernel for time and others
            self.kernel_time = gpytorch.kernels.MaternKernel(nu=2.5, active_dims=time_idx)
            self.kernel_others = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(
                    nu=2.5,
                    ard_num_dims=len(self.feature_idx),
                    active_dims=self.feature_idx
                )
            )
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")

        # Combine the kernels additively
        self.covar_module = self.kernel_time + self.kernel_others

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)  # active_dims handles slicing
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class RecurrentHealthEventsGP:
    def __init__(self, config: dict, features: Optional[List[str]] = None, kernel_type: KernelType = KernelType.RBF):
        # Configuration
        self.config = config
        self.features = features or config.get("features")
        self.kernel_type = kernel_type or KernelType(config.get("kernel_type", "RBF"))
        if not self.features:
            raise ValueError("Features must be provided in the configuration or as an argument.")
        self.time_feature = config.get("time_feature", "TIME")
        if self.time_feature not in self.features:
            raise ValueError(f"Time feature '{self.time_feature}' must be in the features list.")
        self.target_col = config.get("target_col")
        if not self.target_col:
            raise ValueError("Target column must be specified in the configuration.")
        self.batch_size = config.get("batch_size", 512)
        self.lr = config.get("lr", 0.01)
        self.epochs = config.get("epochs", 10)
        self.num_inducing = config.get("num_inducing", 128)
        self.scheduler_step_size = config.get("step_size", 20)
        self.scheduler_gamma = config.get("gamma", 0.5)
        self.device = config.get("device", "cpu")
        self.time_idx = [self.features.index(self.time_feature)]
        self.binary_features = config.get("binary_features", [])

        # Model components
        self.scaler = StandardScaler()
        self.model = None
        self.likelihood = None
        self.optimizer = None
        self.mll = None
        self.train_loss_curve = []
        self.lr_schedule = []

    def fit(self, df):
        X = df[self.features].values
        y = df[self.target_col].values.astype(np.float32)

        # Identify feature indices
        continuous_features = [f for f in self.features if f not in self.binary_features]
        continuous_idx = [self.features.index(f) for f in continuous_features]
        binary_idx = [self.features.index(f) for f in self.binary_features]

        # Scale only continuous features
        X_continuous = df[continuous_features].values
        X_binary = df[self.binary_features].values

        X_cont_scaled = self.scaler.fit_transform(X_continuous)

        # Reconstruct full X with correct order
        X_combined = np.zeros_like(X, dtype=np.float32)
        for i_cont, i_full in zip(range(len(continuous_idx)), continuous_idx):
            X_combined[:, i_full] = X_cont_scaled[:, i_cont]
        for i_bin, i_full in zip(range(len(binary_idx)), binary_idx):
            X_combined[:, i_full] = X_binary[:, i_bin]

        # Convert to torch
        X_tensor = torch.tensor(X_combined, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        # Prepare DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Setup GP model
        inducing_points = X_tensor[:self.num_inducing].clone().to(self.device)
        self.model = GPClassifier(
            inducing_points, time_idx=self.time_idx, kernel_type=self.kernel_type
        ).to(self.device)
        self.likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(self.device)
        self.model.train()
        self.likelihood.train()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.scheduler_step_size,
            gamma=self.scheduler_gamma
        )
        self.mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=len(dataset))

        # Training loop
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                self.optimizer.zero_grad()
                output = self.model(xb)
                loss = -self.mll(output, yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
    
            self.train_loss_curve.append(epoch_loss)
            last_lr = self.scheduler.get_last_lr()[0]
            self.lr_schedule.append(last_lr)

            self.scheduler.step()
            print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {epoch_loss:.4f} - LR: {last_lr:.6f}")

    def predict_proba(self, df):
        self.model.eval()
        self.likelihood.eval()

        # Separate continuous and binary features
        continuous_features = [f for f in self.features if f not in self.binary_features]
        continuous_idx = [self.features.index(f) for f in continuous_features]
        binary_idx = [self.features.index(f) for f in self.binary_features]

        # Scale continuous, preserve binary
        X_continuous = df[continuous_features].values
        X_binary = df[self.binary_features].values

        X_cont_scaled = self.scaler.transform(X_continuous)

        # Reconstruct full matrix in original feature order
        X_combined = np.zeros((len(df), len(self.features)), dtype=np.float32)
        for i_cont, i_full in zip(range(len(continuous_idx)), continuous_idx):
            X_combined[:, i_full] = X_cont_scaled[:, i_cont]
        for i_bin, i_full in zip(range(len(binary_idx)), binary_idx):
            X_combined[:, i_full] = X_binary[:, i_bin]

        # Convert to tensor
        X_tensor = torch.tensor(X_combined, dtype=torch.float32).to(self.device)

        # GP prediction
        with torch.no_grad():
            preds = self.likelihood(self.model(X_tensor))
            return preds.mean.cpu().numpy()


    def predict(self, df, threshold=0.5):
        probs = self.predict_proba(df)
        return (probs > threshold).astype(int)

    def score(self, df, threshold=0.5):
        """
        Compute accuracy and ROC AUC on labeled data.
        """
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in DataFrame.")

        y_true = df[self.target_col].values.astype(int)
        y_probs = self.predict_proba(df)
        y_pred = (y_probs > threshold).astype(int)

        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_probs)

        print(f"Accuracy: {acc:.4f}")
        print(f"ROC AUC : {auc:.4f}")

        return {"accuracy": acc, "roc_auc": auc}
