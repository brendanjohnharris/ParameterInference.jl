using NonstationaryProcesses
using Catch22

sim = gaussianBimodalSim(
    parameter_profile = (constantParameter, constantParameter, constantParameter),
    parameter_profile_parameters = ((5.0,), (1.0,), (0.1)),  # (μ, σ, α)
    transient_t0 = 0.0,
    t0 = 0.0,
    savedt = 1,
    tmax = 100.0
)
S = sim(tmax=1000); tsdensity(S)
# Run lots of baseline simulations, and save the results in a vector
𝐩 = fill(5.0, (100,));
#𝐗0 = 0.01.*randn(100).+1.0 # So we don't have exactly the same time series repeated 100 times
𝒮 = [simulate(sim(ps = (p, (1.0), (0.2)))) for p ∈ 𝐩];
𝒳 = timeseries.(𝒮);

# Calculate features for each element of 𝒳
ℱ = (catch24∘Array).(𝒳);
ℱ  = Catch22.featureMatrix(hcat(ℱ...), Catch22.featureDims(ℱ[1]), 𝐩)

# Produce a baseline
𝑏 = reStandardise(ℱ)

# Run an inference, without incorporating the baseline
S = simulate(gaussianBimodalSim)
I = infer(S)
parameterestimate(I, normalisef=false)

# Run an inference, incorporating the baseline
I = infer(S, baseline=𝑏, normalisation=_self)
parameterestimate(I, normalisef=false)
