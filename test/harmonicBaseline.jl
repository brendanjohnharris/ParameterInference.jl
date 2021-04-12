using NonstationaryProcesses
using Catch22

sim = shcalySineSim(
    X0 = [1.0, 0.0],
    profile = constantParameter,
    parameters = [4π],
    transient = -10.0,
    𝑡₀ = 0.0,
    δ𝑡 = 0.0001,
    Δ𝑡 = 0.001,
    𝑇 = 10.0,
    alg = RK4()
    )

# Run lots of baseline simulations, and save the results in a vector
𝐩 = fill(4π, (100,));
𝒮 = [simulate(sim(ps = [p])) for p ∈ 𝐩];
𝒳 = timeseries.(𝒮);

# Calculate features for each element of 𝒳
ℱ = (catch24∘Array).(𝒳);
ℱ  = Catch22.featureMatrix(hcat(ℱ...), Catch22.featureDims(ℱ[1]), 𝐩)

# Produce a baseline
𝑏 = reStandardise(ℱ)

# Run an inference, without incorporating the baseline
S = simulate(harmonicSim(profile=sineWave, ps=(50.0, 1.0, 0.0, 20.0)))
I = infer(S)
parameterestimate(I, normalisef=false, tswindows=false)

# Run an inference, incorporating the baseline
I = infer(S, baseline=𝑏, normalisation=_self)
parameterestimate(I, normalisef=false)
