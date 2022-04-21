from cyipopt import minimize_ipopt
import numpy as np
import scipy.optimize
from scipy._lib._util import check_random_state
from scipy.optimize._basinhopping import Storage, BasinHoppingRunner, AdaptiveStepsize, RandomDisplacement, MinimizerWrapper, Metropolis


def basinhopping_ipopt(func, x0, niter=100, T=1.0, stepsize=0.5,
                 minimizer_kwargs=None, take_step=None, accept_test=None,
                 callback=None, interval=50, disp=False, niter_success=None,
                 seed=None, *, target_accept_rate=0.5, stepwise_factor=0.9):
    """This is a version of scipy.optimize.basinhopping that makes use of 
    cyipopt.minimize_ipopt instead of scipy.optimize.minimize
    
    Args:
    func : callable ``f(x, *args)``
        Function to be optimized.  ``args`` can be passed as an optional item
        in the dict ``minimizer_kwargs``
    x0 : array_like
        Initial guess.
    niter : integer, optional
        The number of basin-hopping iterations. There will be a total of
        ``niter + 1`` runs of the local minimizer.
    T : float, optional
        The "temperature" parameter for the accept or reject criterion. Higher
        "temperatures" mean that larger jumps in function value will be
        accepted.  For best results ``T`` should be comparable to the
        separation (in function value) between local minima.
    stepsize : float, optional
        Maximum step size for use in the random displacement.
    minimizer_kwargs : dict, optional
        Extra keyword arguments to be passed to the local minimizer
        ``scipy.optimize.minimize()`` Some important options could be:
            method : str
                The minimization method (e.g. ``"L-BFGS-B"``)
            args : tuple
                Extra arguments passed to the objective function (``func``) and
                its derivatives (Jacobian, Hessian).
    take_step : callable ``take_step(x)``, optional
        Replace the default step-taking routine with this routine. The default
        step-taking routine is a random displacement of the coordinates, but
        other step-taking algorithms may be better for some systems.
        ``take_step`` can optionally have the attribute ``take_step.stepsize``.
        If this attribute exists, then ``basinhopping`` will adjust
        ``take_step.stepsize`` in order to try to optimize the global minimum
        search.
    accept_test : callable, ``accept_test(f_new=f_new, x_new=x_new, f_old=fold, x_old=x_old)``, optional
        Define a test which will be used to judge whether or not to accept the
        step.  This will be used in addition to the Metropolis test based on
        "temperature" ``T``.  The acceptable return values are True,
        False, or ``"force accept"``. If any of the tests return False
        then the step is rejected. If the latter, then this will override any
        other tests in order to accept the step. This can be used, for example,
        to forcefully escape from a local minimum that ``basinhopping`` is
        trapped in.
    callback : callable, ``callback(x, f, accept)``, optional
        A callback function which will be called for all minima found. ``x``
        and ``f`` are the coordinates and function value of the trial minimum,
        and ``accept`` is whether or not that minimum was accepted. This can
        be used, for example, to save the lowest N minima found. Also,
        ``callback`` can be used to specify a user defined stop criterion by
        optionally returning True to stop the ``basinhopping`` routine.
    interval : integer, optional
        interval for how often to update the ``stepsize``
    disp : bool, optional
        Set to True to print status messages
    niter_success : integer, optional
        Stop the run if the global minimum candidate remains the same for this
        number of iterations.
    seed : {None, int, `numpy.random.Generator`,
            `numpy.random.RandomState`}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
        Specify `seed` for repeatable minimizations. The random numbers
        generated with this seed only affect the default Metropolis
        `accept_test` and the default `take_step`. If you supply your own
        `take_step` and `accept_test`, and these functions use random
        number generation, then those functions are responsible for the state
        of their random number generator.
    target_accept_rate : float, optional
        The target acceptance rate that is used to adjust the `stepsize`.
        If the current acceptance rate is greater than the target,
        then the `stepsize` is increased. Otherwise, it is decreased.
        Range is (0, 1). Default is 0.5.
        .. versionadded:: 1.8.0
    stepwise_factor : float, optional
        The `stepsize` is multiplied or divided by this stepwise factor upon
        each update. Range is (0, 1). Default is 0.9.
        .. versionadded:: 1.8.0
    
    Returns:
    res : OptimizeResult
        The optimization result represented as a ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, ``fun`` the value
        of the function at the solution, and ``message`` which describes the
        cause of the termination. The ``OptimizeResult`` object returned by the
        selected minimizer at the lowest minimum is also contained within this
        object and can be accessed through the ``lowest_optimization_result``
        attribute.  See `OptimizeResult` for a description of other attributes.
    """
    if target_accept_rate <= 0. or target_accept_rate >= 1.:
        raise ValueError('target_accept_rate has to be in range (0, 1)')
    if stepwise_factor <= 0. or stepwise_factor >= 1.:
        raise ValueError('stepwise_factor has to be in range (0, 1)')

    x0 = np.array(x0)

    # set up the np.random generator
    rng = check_random_state(seed)

    # set up minimizer
    if minimizer_kwargs is None:
        minimizer_kwargs = dict()
    wrapped_minimizer = MinimizerWrapper(minimize_ipopt, func,
                                         **minimizer_kwargs)

    # set up step-taking algorithm
    if take_step is not None:
        if not callable(take_step):
            raise TypeError("take_step must be callable")
        # if take_step.stepsize exists then use AdaptiveStepsize to control
        # take_step.stepsize
        if hasattr(take_step, "stepsize"):
            take_step_wrapped = AdaptiveStepsize(
                take_step, interval=interval,
                accept_rate=target_accept_rate,
                factor=stepwise_factor,
                verbose=disp)
        else:
            take_step_wrapped = take_step
    else:
        # use default
        displace = RandomDisplacement(stepsize=stepsize, random_gen=rng)
        take_step_wrapped = AdaptiveStepsize(displace, interval=interval,
                                             accept_rate=target_accept_rate,
                                             factor=stepwise_factor,
                                             verbose=disp)

    # set up accept tests
    accept_tests = []
    if accept_test is not None:
        if not callable(accept_test):
            raise TypeError("accept_test must be callable")
        accept_tests = [accept_test]

    # use default
    metropolis = Metropolis(T, random_gen=rng)
    accept_tests.append(metropolis)

    if niter_success is None:
        niter_success = niter + 2

    bh = BasinHoppingRunner(x0, wrapped_minimizer, take_step_wrapped,
                            accept_tests, disp=disp)

    # The wrapped minimizer is called once during construction of
    # BasinHoppingRunner, so run the callback
    if callable(callback):
        callback(bh.storage.minres.x, bh.storage.minres.fun, True)

    # start main iteration loop
    count, i = 0, 0
    message = ["requested number of basinhopping iterations completed"
               " successfully"]
    for i in range(niter):
        new_global_min = bh.one_cycle()

        if callable(callback):
            # should we pass a copy of x?
            val = callback(bh.xtrial, bh.energy_trial, bh.accept)
            if val is not None:
                if val:
                    message = ["callback function requested stop early by"
                               "returning True"]
                    break

        count += 1
        if new_global_min:
            count = 0
        elif count > niter_success:
            message = ["success condition satisfied"]
            break

    # prepare return object
    res = bh.res
    res.lowest_optimization_result = bh.storage.get_lowest()
    res.x = np.copy(res.lowest_optimization_result.x)
    res.fun = res.lowest_optimization_result.fun
    res.message = message
    res.nit = i + 1
    res.success = res.lowest_optimization_result.success
    return res