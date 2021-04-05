using NonstationaryProcesses
using Catch22

sim = shcalySineSim(
    X0 = [0.0],
    profile = constantParameter,
    parameters = [1.0],
    transient = 0.0,
    𝑡₀ = 0.0,
    δ𝑡 = 0.01,
    Δ𝑡 = 0.01,
    𝑇 = 50.0
    )

# Run lots of baseline simulations, and save the results in a vector
# The baseline for this example is just noise of the same amplitude.
# This exposes any variance in features that really shouldn't be there.
𝐩 = fill(1.0, (100,));
𝐗0 = 0.01.*randn(100).+1.0 # So we don't have exactly the same time series repeated 100 times
𝒮 = [simulate(sim(ps = [p])) for p ∈ 𝐩];
𝒳 = timeseries.(𝒮);

# Calculate features for each element of 𝒳
ℱ = (catch24∘Array).(𝒳);
ℱ  = Catch22.featureMatrix(hcat(ℱ...), Catch22.featureDims(ℱ[1]), 𝐩)

# Produce a baseline
𝑏 = reStandardise(ℱ)

# Run an inference, without incorporating the baseline
S = simulate(shcalySineSim)
I = infer(S)
parameterestimate(I, normalisef=false, tswindows=false)

# Run an inference, incorporating the baseline
I = infer(S, baseline=𝑏, normalisation=_self)
parameterestimate(I, normalisef=false)
