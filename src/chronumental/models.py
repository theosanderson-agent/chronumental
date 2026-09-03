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

        self.initial_branch_times_array = kwargs.get(
            'initial_branch_times_array')
        self.initial_root_date = kwargs.get('initial_root_date')
        self.set_initial_time()
        self.terminal_names = kwargs['terminal_names']

    def get_initial_root_date(self):
        if self.initial_root_date is not None:
            return self.initial_root_date
        return (-helpers.DAYS_PER_YEAR * self.ref_point_distance /
                self.clock_rate)

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

    def get_date_sigmas(self):
        """Standard deviation of each tip's date likelihood, in days.

        The same quantity the fit uses, exposed because the confidence
        intervals need it: it is the whole of the tip term's curvature.
        """
        return self._date_scale()

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
        self.fix_clock_rate = kwargs['model_configuration'][
            'fix_clock_rate']
        self.variance_on_clock_rate = kwargs['model_configuration'][
            'variance_on_clock_rate']
        self.expected_min_between_transmissions = kwargs[
            'model_configuration']['expected_min_between_transmissions']
        self.quadrature_date_scale = kwargs['model_configuration'].get(
            'quadrature_date_scale', True)
        self.clock_likelihood = kwargs['model_configuration'].get(
            'clock_likelihood', 'poisson')
        self.branch_rate_cv_init = kwargs['model_configuration'].get(
            'branch_rate_cv_init', 0.3)
        self.root_date_prior_scale = kwargs['model_configuration'].get(
            'root_date_prior_scale', 36500.0)
        if self.branch_rate_cv_init <= 0:
            raise ValueError("branch_rate_cv_init must be positive")

        super().__init__(**kwargs)

    def get_logging_results(self, params):
        results = super().get_logging_results(params)
        results['mutation_rate'] = self.get_mutation_rate(params)
        if self.clock_likelihood == 'gamma-poisson':
            results['branch_rate_cv'] = params['branch_rate_cv']
        return results

    def set_initial_time(self):
        if self.initial_branch_times_array is not None:
            self.initial_time = jnp.maximum(self.initial_branch_times_array,
                                            1e-3)
            return
        self.initial_time = jnp.maximum(
            helpers.DAYS_PER_YEAR * (self.branch_distances_array) /
            self.clock_rate,
            self.expected_min_between_transmissions)

    def calc_dates(self, branch_lengths_array, root_date):

        all_node_dates = self.path_sum(branch_lengths_array)
        return all_node_dates[self.terminal_indices] + root_date

    def model(self):
        # Cauchy, not Normal. root_date is measured in days before the
        # oldest tip, so a Normal penalises a deep root quadratically: at
        # the old scale of 1000 days this prior asserted the root lay within
        # about three years of the oldest tip, which on a tree spanning
        # centuries dominated everything else in the objective and dragged
        # the root forward. A Cauchy's penalty grows logarithmically, so a
        # tree far deeper than the scale costs a little rather than a lot,
        # and the scale stops having to be guessed from the tree's depth.
        root_date = numpyro.sample(
            "root_date",
            dist.Cauchy(loc=0.0, scale=self.root_date_prior_scale))

        branch_times = numpyro.sample(
            "latent_time_length",
            dist.Uniform(
                low=onp.ones(self.branch_distances_array.shape[0]) * 0,
                high=onp.ones(self.branch_distances_array.shape[0]) * 365 *
                10000))

        if self.fix_clock_rate:
            mutation_rate = self.clock_rate
        else:
            mutation_rate = numpyro.sample(
                f"latent_mutation_rate",
                dist.Uniform(low=0.0, high=self.clock_rate * 1000.0))

        expected_mutations = (mutation_rate * branch_times /
                              helpers.DAYS_PER_YEAR)
        if self.clock_likelihood == 'gamma-poisson':
            # If a branch rate is Gamma distributed with mean mutation_rate
            # and coefficient of variation cv, integrating that rate out of
            # the Poisson count likelihood gives NegativeBinomial2 with
            # concentration 1 / cv^2. This adds one scalar parameter rather
            # than one latent rate per branch.
            branch_rate_cv = numpyro.param(
                "branch_rate_cv", self.branch_rate_cv_init,
                constraint=dist.constraints.positive)
            branch_distances_distribution = dist.NegativeBinomial2(
                mean=expected_mutations,
                concentration=1.0 / branch_rate_cv**2)
        else:
            branch_distances_distribution = dist.Poisson(expected_mutations)

        branch_distances = numpyro.sample(
            "branch_distances", branch_distances_distribution,
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

        branch_times = numpyro.sample("latent_time_length",
                                      dist.Delta(time_length_mu))

        # With the rate held fixed the model reads it straight off
        # self.clock_rate and never samples it, so the guide must not declare
        # a latent site for it either -- numpyro warns about guide sites the
        # model does not use.
        if self.fix_clock_rate:
            return

        mutation_rate_mu = numpyro.param("mutation_rate_mu",
                                         self.clock_rate,
                                         constraint=dist.constraints.positive)
        mutation_rate_sigma = numpyro.param(
            "mutation_rate_sigma",
            self.clock_rate,
            constraint=dist.constraints.positive)

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
        if self.fix_clock_rate:
            return self.clock_rate
        return params['mutation_rate_mu']


models = {"DeltaGuideWithStrictLearntClock": DeltaGuideWithStrictLearntClock}
