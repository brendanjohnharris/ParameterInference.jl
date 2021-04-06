using NonstationaryProcesses
using Plots
using Changepoints # Need to add Changepoints#master
using Distributions

# First make a time series with mean shifts
sim = shiftyNoiseSim(
        parameter_profile = (constantParameter, stepNoise),
        parameter_profile_parameters = [(1.0,), ((0.0, 1000.0), 100.0, 2.0, 0.0)],
        transient_t0 = 0.0,
        t0 = 0.0,
        savedt = 1,
        tmax = 1000.0)
S = simulate(sim)
𝐱 = timeseries(S)
# Run binary segmentation
# The cost function is twice the negative log-likelihood, by default
bs_cps = @BS Array(𝐱) Normal(:?, 1.0)
changepoint_plot(𝐱, bs_cps[1])


# Now try something that is not a mean shift
# sim = fmWaveSim(
#     parameter_profile = (stepNoise),
#     parameter_profile_parameters = ((0.0, 100.0), 10.0, 0.5, 0.0),
#     transient_t0 = 0.0,
#     t0 = 0.0,
#     savedt = 0.01,
#     tmax = 100.0)
# S = simulate(sim)
# 𝐱 = timeseries(S)
# # Run binary segmentation
# # The cost function is twice the negative log-likelihood, by default
# bs_cps = @BS Array(𝐱) Normal(:?, 1.0)
# changepoint_plot(𝐱, bs_cps[1])