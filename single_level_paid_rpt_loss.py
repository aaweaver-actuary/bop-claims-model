import pymc as pm
from pymc.distributions import Gamma, Normal, Uniform, HalfNormal, Deterministic


def single_level_paid_rpt_loss_model(
    origin_period, development_period, observed_rpt_loss, observed_paid_loss, **kwargs
):
    coords = {
        "origin_period": origin_period,
        "development_period": development_period,
        "observed_rpt_loss": observed_rpt_loss,
        "observed_paid_loss": observed_paid_loss,
    }

    with pm.Model(coords=coords) as model:
        # ---- Define priors ----

        # Will use a lognormal distribution to model the expected ultimate loss. We will sample
        # from a Normal distribution for the log of the expected ultimate loss.
        log__expected_ult__loss = Normal(
            "log__expected_ult__loss", mu=0, sd=10, dims="origin_period"
        )

        # Expected percentage of ultimate loss reported and paid at each development period. We will
        # sample from a uniform distribution for these parameters.
        expected_pct_of_ult__rpt_loss = Uniform(
            "expected_pct_of_ult__rpt_loss",
            lower=0,
            upper=2,  # sometimes the reported loss can be greater than the ultimate loss due to conservative case reserving
            dims="development_period",
        )
        expected_pct_of_ult__paid_loss = Uniform(
            "expected_pct_of_ult__paid_loss",
            lower=0,
            upper=1.2,  # paid loss is much less likely to exceed the ultimate loss, but I don't want to rule it out
            dims="development_period",
        )

        # Reported and paid loss are assumed to be Gamma distributed, so we need to define the
        # shape and rate parameters for each distribution. We will sample from a HalfNormal
        # distribution for the standard deviation of the reported and paid loss distributions.
        sigma__rpt_loss = HalfNormal(
            "sigma__rpt_loss",
            sd=10,
            dims="origin_period",
            shape=(coords["origin_period"].shape, coords["development_period"].shape),
        )
        sigma__paid_loss = HalfNormal(
            "sigma__paid_loss",
            sd=10,
            dims="origin_period",
            shape=(coords["origin_period"].shape, coords["development_period"].shape),
        )

        # ---- Define deterministic variables ----

        # Expected ultimate loss is just the exponential of the log of the expected ultimate loss
        expected_ult__loss = Deterministic(
            "expected_ult__loss", pm.math.exp(log__expected_ult__loss)
        )

        # Expected rpt loss at each cell in the triangle is the product of the expected ultimate
        # loss and the expected pct of ultimate loss reported at that cell
        mu__rpt_loss = Deterministic(
            "mu__rpt_loss",
            expected_ult__loss * expected_pct_of_ult__rpt_loss,
            shape=(coords["origin_period"].shape, coords["development_period"].shape),
        )

        # Expected paid loss at each cell in the triangle is the product of the expected ultimate
        # loss and the expected pct of ultimate loss paid at that cell
        mu__paid_loss = Deterministic(
            "mu__paid_loss",
            expected_ult__loss * expected_pct_of_ult__paid_loss,
            shape=(coords["origin_period"].shape, coords["development_period"].shape),
        )

        # ---- Define likelihood ----
        rpt_loss = Gamma(  # noqa: F841 (variable is "unused" here, but is estimated as a model parameter)
            "rpt_loss",
            alpha=mu__rpt_loss / sigma__rpt_loss**2,
            beta=sigma__rpt_loss**2 / mu__rpt_loss,
            observed=coords["observed_rpt_loss"],
            dims=("origin_period", "development_period"),
        )
        paid_loss = Gamma(  # noqa: F841 (variable is "unused" here, but is estimated as a model parameter)
            "paid_loss",
            alpha=mu__paid_loss / sigma__paid_loss**2,
            beta=sigma__paid_loss**2 / mu__paid_loss,
            observed=coords["observed_paid_loss"],
            dims=("origin_period", "development_period"),
        )

    return model
