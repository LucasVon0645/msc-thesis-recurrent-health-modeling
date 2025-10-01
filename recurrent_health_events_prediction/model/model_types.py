from enum import Enum

class SurvivalModelType(Enum):
    """
    Enum class for model types.
    """
    KAPLAN_MEIER = "kaplan_meier"
    COX_PH = "cox_ph"
    COX_PH_SE = "cox_ph_se"
    LOGNORMAL_AFT = "lognormal_aft"
    WEIBULL_AFT = "weibull_aft"
    GBM = "survival_gbm"
    FRAILTY_COX_R = "frailty_cox_r"

class DistributionType(Enum):
    """
    Enum class for distribution types.
    """
    LOG_NORMAL = "log_normal"
    EXPONENTIAL = "exponential"
    POISSON = "poisson"
    GAUSSIAN = "gaussian"
    NORMAL = "normal"
    CATEGORICAL = "categorical"
    GAMMA = "gamma"
    HALF_NORMAL = "half_normal"
    BERNOULLI = "bernoulli"
    WEIBULL = "weibull"
    STUDENT_T = "student_t"