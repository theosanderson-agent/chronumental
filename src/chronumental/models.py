import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
import numpy as onp
from . import helpers
import collections


class ChronumentalModelBase(object):

    def __init__(self, **kwargs):
        # Sums branch times from the root to every node, by pointer jumping.
        # See helpers.make_path_sum: this replaced a sparse matrix that cost
        # memory proportional to total path length over the tree.
        self.path_sum = kwargs['path_sum']
        self.terminal_indices = kwargs['terminal_indices']
        self.branch_distances_array = kwargs['branch_distances_array']
        self.terminal_target_dates_array = kwargs[
            'terminal_target_dates_array']
        self.terminal_target_errors_array = kwargs[
            'terminal_target_errors_array']
        self.ref_point_distance = kwargs['ref_point_distance']

        # Optional, better starting point for the variational parameters,
        # estimated from the tree topology and the tip dates (see
        # input_mod.estimate_initial_times). When absent, each model falls
        # back to its old clock-rate-only initialisation.
        self.initial_branch_times_array = kwargs.get(
            'initial_branch_times_array')
        self.initial_root_date = kwargs.get('initial_root_date')

        self.set_initial_time()
        self.terminal_names = kwargs['terminal_names']

    def get_initial_root_date(self):
        if self.initial_root_date is not None:
            return self.initial_root_date
        return -365 * self.ref_point_distance / self.clock_rate

    def _date_scale(self):
        """Standard deviation of the date likelihood, per tip, in days.

        terminal_target_errors_array holds each tip's precision window: 1 for
        a full date, 30 for month-only, 365 for year-only. Multiplying that by
        --variance_dates conflates two different things. The window says how
        wide the reported interval is; --variance_dates says how far a
        reported date can be from the truth for other reasons. Multiplying
        them means raising the second inflates the first, so at the current
        default a year-only date is treated as uncertain to within ten years.

        Adding them in quadrature keeps them separate: a full date gets about
        --variance_dates, a month-only date is dominated by its own window,
        and raising --variance_dates no longer multiplies the window.
        """
        if not self.quadrature_date_scale:
            return self.variance_dates * self.terminal_target_errors_array
        window = self.terminal_target_errors_array / 2.0
        return jnp.sqrt(self.variance_dates**2 + window**2)

    def get_logging_results(self, params):
        results = collections.OrderedDict()
        times = self.get_branch_times(params)
        new_dates = self.calc_dates(times, params['root_date_mu'])
        results['date_cor'] = onp.corrcoef(self.terminal_target_dates_array,
                                           new_dates)[0, 1]
        results['date_error'] = onp.mean(
            onp.abs(self.terminal_target_dates_array -
                    new_dates))  # Average date error should be small
        results['date_error_med'] = onp.median(
            onp.abs(self.terminal_target_dates_array -
                    new_dates))  # Average date error should be small

        results['max_date_error'] = onp.max(
            onp.abs(self.terminal_target_dates_array - new_dates)
        )  # We know that there are some metadata errors, so there probably should be some big errors
        results['length_cor'] = onp.corrcoef(
            self.branch_distances_array,
            times)[0, 1]  # This correlation should be relatively high

        results['root_date'] = params['root_date_mu']
        return results


class DeltaGuideWithStrictLearntClock(ChronumentalModelBase):

    def __init__(self, **kwargs):

        self.clock_rate = kwargs['model_configuration']["clock_rate"]

        self.variance_dates = kwargs['model_configuration']['variance_dates']
        self.enforce_exact_clock = kwargs['model_configuration'][
            'enforce_exact_clock']
        self.variance_on_clock_rate = kwargs['model_configuration'][
            'variance_on_clock_rate']
        self.expected_min_between_transmissions = kwargs[
            'model_configuration']['expected_min_between_transmissions']
        self.quadrature_date_scale = kwargs['model_configuration'].get(
            'quadrature_date_scale', True)

        super().__init__(**kwargs)

    def get_logging_results(self, params):
        results = super().get_logging_results(params)
        results['mutation_rate'] = self.get_mutation_rate(params)
        return results

    def set_initial_time(self):
        if self.initial_branch_times_array is not None:
            # A small positive floor, not expected_min_between_transmissions:
            # this initial value already reflects the tip dates, so it
            # should not be dragged upward by a floor tuned for the crude
            # mutations/clock_rate estimate.
            self.initial_time = jnp.maximum(self.initial_branch_times_array,
                                            1e-3)
        else:
            self.initial_time = jnp.maximum(
                365 * (self.branch_distances_array) / self.clock_rate,
                self.expected_min_between_transmissions)

    def calc_dates(self, branch_lengths_array, root_date):

        all_node_dates = self.path_sum(branch_lengths_array)
        return all_node_dates[self.terminal_indices] + root_date

    def model(self):
        root_date = numpyro.sample("root_date",
                                   dist.Normal(loc=0.0, scale=1000.0))

        branch_times = numpyro.sample(
            "latent_time_length",
            dist.Uniform(
                low=onp.ones(self.branch_distances_array.shape[0]) * 0,
                high=onp.ones(self.branch_distances_array.shape[0]) * 365 *
                10000))

        if self.enforce_exact_clock:
            mutation_rate = self.clock_rate
        else:
            mutation_rate = numpyro.sample(
                f"latent_mutation_rate",
                dist.Uniform(low=0.0, high=self.clock_rate * 1000.0))

        branch_distances = numpyro.sample("branch_distances",
                                          dist.Poisson(mutation_rate *
                                                       branch_times / 365),
                                          obs=self.branch_distances_array)

        calced_dates = self.calc_dates(branch_times, root_date)

        final_dates = numpyro.sample(f"final_dates",
                                     dist.Normal(calced_dates,
                                                 self._date_scale()),
                                     obs=self.terminal_target_dates_array)

    def guide(self):
        root_date_mu = numpyro.param("root_date_mu",
                                     self.get_initial_root_date())

        root_date = numpyro.sample("root_date", dist.Delta(root_date_mu))

        time_length_mu = numpyro.param("time_length_mu",
                                       self.initial_time,
                                       constraint=dist.constraints.positive)

        mutation_rate_mu = numpyro.param("mutation_rate_mu",
                                         self.clock_rate,
                                         constraint=dist.constraints.positive)
        mutation_rate_sigma = numpyro.param(
            "mutation_rate_sigma",
            self.clock_rate,
            constraint=dist.constraints.positive)

        branch_times = numpyro.sample("latent_time_length",
                                      dist.Delta(time_length_mu))

        if not self.variance_on_clock_rate:
            mutation_rate = numpyro.sample("latent_mutation_rate",
                                           dist.Delta(mutation_rate_mu))
        else:
            mutation_rate = numpyro.sample(
                f"latent_mutation_rate",
                dist.TruncatedNormal(mutation_rate_mu,
                                     mutation_rate_sigma,
                                     low=0.0))

    def get_branch_times(self, params):
        return params['time_length_mu']

    def get_mutation_rate(self, params):
        if self.enforce_exact_clock:
            return self.clock_rate
        return params['mutation_rate_mu']


class HorseShoeLike(ChronumentalModelBase):

    def __init__(self, **kwargs):

        self.clock_rate = kwargs['model_configuration']["clock_rate"]

        self.variance_dates = kwargs['model_configuration']['variance_dates']
        self.enforce_exact_clock = kwargs['model_configuration'][
            'enforce_exact_clock']
        self.variance_on_clock_rate = kwargs['model_configuration'][
            'variance_on_clock_rate']
        self.expected_min_between_transmissions = kwargs[
            'model_configuration']['expected_min_between_transmissions']
        self.quadrature_date_scale = kwargs['model_configuration'].get(
            'quadrature_date_scale', True)

        self.initial_tau = kwargs['model_configuration']['initial_tau']
        self.fixed_tau = kwargs['model_configuration']['fixed_tau']
        self.hs_scale = kwargs['model_configuration']['hs_scale']

        super().__init__(**kwargs)

    def get_logging_results(self, params):
        results = super().get_logging_results(params)
        results['mutation_rate'] = self.get_mutation_rate(params)
        results['argmax_variances_param'] = self.terminal_names[onp.argmax(
            params['variances_param'])]
        results['median_variances_param'] = onp.median(
            params['variances_param'])
        results['max_variances_param'] = onp.max(params['variances_param'])
        results['tau'] = params['tau_param']
        return results

    def set_initial_time(self):
        if self.initial_branch_times_array is not None:
            # A small positive floor, not expected_min_between_transmissions:
            # this initial value already reflects the tip dates, so it
            # should not be dragged upward by a floor tuned for the crude
            # mutations/clock_rate estimate.
            self.initial_time = jnp.maximum(self.initial_branch_times_array,
                                            1e-3)
        else:
            self.initial_time = jnp.maximum(
                365 * (self.branch_distances_array) / self.clock_rate,
                self.expected_min_between_transmissions)

    def calc_dates(self, branch_lengths_array, root_date):

        all_node_dates = self.path_sum(branch_lengths_array)
        return all_node_dates[self.terminal_indices] + root_date

    def model(self):
        root_date = numpyro.sample("root_date",
                                   dist.Normal(loc=0.0, scale=1000.0))

        branch_times = numpyro.sample(
            "latent_time_length",
            dist.Uniform(
                low=onp.ones(self.branch_distances_array.shape[0]) * 0,
                high=onp.ones(self.branch_distances_array.shape[0]) * 365 *
                10000))

        if self.enforce_exact_clock:
            mutation_rate = self.clock_rate
        else:
            mutation_rate = numpyro.sample(
                f"latent_mutation_rate",
                dist.Uniform(low=0.0, high=self.clock_rate * 1000.0))

        branch_distances = numpyro.sample("branch_distances",
                                          dist.Poisson(mutation_rate *
                                                       branch_times / 365),
                                          obs=self.branch_distances_array)

        calced_dates = self.calc_dates(branch_times, root_date)

        hs_scale = self.hs_scale

        tau = numpyro.sample("tau", dist.HalfCauchy(hs_scale))

        lambda_l = numpyro.sample(
            "lambda",
            dist.HalfCauchy(hs_scale),
            sample_shape=self.terminal_target_dates_array.shape)

        # A horseshoe scale is the *product* tau * lambda_l, not the product
        # of their squares: dist.Normal's second argument is a standard
        # deviation, so squaring both factors before multiplying them
        # silently works in units of variance^2, which is not what any
        # horseshoe construction intends and is wildly miscalibrated (a
        # lambda_l/tau of, say, 2 turns into an implied std of 16, not 2).
        #
        # The known date-precision category (terminal_target_errors_array:
        # 1 for a full date, 30 for a month, 365 for a year-only date) is
        # exactly the same information the default model uses, via
        # `variance_dates * terminal_target_errors_array`. Discarding it, as
        # the old horseshoe likelihood did, throws away a strong, free
        # signal about which tips to distrust and replaces it with one the
        # model has to rediscover from scratch. Instead treat it as a floor:
        # the horseshoe term rides on top of it in quadrature, so a tip
        # whose residual is already explained by its declared precision
        # keeps the default model's tolerance for it (lambda_l shrinks
        # toward zero under the HalfCauchy prior), and only a tip whose
        # error is bigger than its category would suggest pushes lambda_l
        # up to explain the extra residual.
        baseline_scale = self.variance_dates * self.terminal_target_errors_array
        extra_scale = tau * lambda_l
        scale = jnp.sqrt(baseline_scale**2 + extra_scale**2)

        final_dates = numpyro.sample(f"final_dates",
                                     dist.Normal(calced_dates, scale),
                                     obs=self.terminal_target_dates_array)

    def guide(self):
        root_date_mu = numpyro.param("root_date_mu",
                                     self.get_initial_root_date())

        root_date = numpyro.sample("root_date", dist.Delta(root_date_mu))

        time_length_mu = numpyro.param("time_length_mu",
                                       self.initial_time,
                                       constraint=dist.constraints.positive)

        mutation_rate_mu = numpyro.param("mutation_rate_mu",
                                         self.clock_rate,
                                         constraint=dist.constraints.positive)
        mutation_rate_sigma = numpyro.param(
            "mutation_rate_sigma",
            self.clock_rate,
            constraint=dist.constraints.positive)

        # Start lambda_l near zero, not 0.2: the likelihood now adds the
        # horseshoe term to the known-precision baseline in quadrature, so
        # starting near zero means fitting begins at (approximately) the
        # default model's behaviour and only grows a tip's extra tolerance
        # if the data actually push for it, rather than starting the fit
        # already distrusting every tip by an arbitrary fixed amount.
        variances = numpyro.param(
            "variances_param",
            onp.ones(self.terminal_target_dates_array.shape) * 1e-3,
            constraint=dist.constraints.positive)

        tau_param = numpyro.param("tau_param",
                                  self.initial_tau,
                                  constraint=dist.constraints.positive)
        tau = numpyro.sample(
            "tau",
            dist.Delta(tau_param if not self.fixed_tau else self.initial_tau))

        sample_variances = numpyro.sample("lambda", dist.Delta(variances))

        branch_times = numpyro.sample("latent_time_length",
                                      dist.Delta(time_length_mu))

        if not self.variance_on_clock_rate:
            mutation_rate = numpyro.sample("latent_mutation_rate",
                                           dist.Delta(mutation_rate_mu))
        else:
            mutation_rate = numpyro.sample(
                f"latent_mutation_rate",
                dist.TruncatedNormal(mutation_rate_mu,
                                     mutation_rate_sigma,
                                     low=0.0))

    def get_branch_times(self, params):
        return params['time_length_mu']

    def get_mutation_rate(self, params):
        if self.enforce_exact_clock:
            return self.clock_rate
        return params['mutation_rate_mu']


models = {
    "DeltaGuideWithStrictLearntClock": DeltaGuideWithStrictLearntClock,
    "HorseShoeLike": HorseShoeLike
}
