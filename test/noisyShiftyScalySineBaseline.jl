using NonstationaryProcesses
using Catch22
using Distributions
using StatsPlots
pyplot()
sim = noisyShiftyScalySineSim(
    X0 = [0.0],
    parameter_profile = (constant, constant, constant), # (η, C, A)
    parameter_profile_parameters = ((1.0), (1.0,), (1.0,)),
    t0 = 0.0,
    savedt = 0.1,
    tmax = 1000.0/20.0
    # Assuming the default N of windows is 20, and other sampling parameters are the same
    )

# Generate a high-dimensional baseline from this system
N = 1000
𝐃 = [Uniform(0.1, 3.0), Uniform(0.1, 3.0), Uniform(0.1, 3.0)];
𝐩 = [rand.(𝐃) for i in 1:N];
𝒳0 = [randn() for i in 1:N]; # In case the IC changes a feature
𝒮 = [simulate(sim(ps = 𝐩[i], X0 = [𝒳0[i]])) for i ∈ 1:N];
𝒳 = timeseries.(𝒮, 1);
F = (catch24∘Array).(𝒳);
ℱ  = Catch22.featureMatrix(hcat(F...), Catch22.featureDims(F[1]), 𝐩)

# Generate a constant  (low-dimensional) baseline
𝒮′ = [simulate(sim(X0 = [𝒳0[i]])) for i ∈ 1:N];
𝒳′ = timeseries.(𝒮′);
F′ = (catch24∘Array).(𝒳′);
ℱ′  = Catch22.featureMatrix(hcat(F′...), Catch22.featureDims(F′[1]))

# First, compare the baselines
featureviolin(ℱ, ℱ′, normalise=true)

# Clearly certain features are not being driven by the parameters.
ℱ̂ = reZero(ℱ′)(ℱ)
featureviolin(ℱ̂, ℱ′, normalise=true)

# Produce a baseline
𝑏 = reStandardise(ℱ̂)

# Run an inference, without incorporating the baseline
S = simulate(noisyShiftyScalySineSim(tmax=1000))
I = infer(S, parameters=3)
parameterestimate(I, normalisef=false)

# Run an inference, incorporating the baseline
I = infer(S, baseline=𝑏, normalisation=_self, parameters=3)
parameterestimate(I, normalisef=false)
