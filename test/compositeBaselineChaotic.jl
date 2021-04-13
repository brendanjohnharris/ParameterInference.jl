using NonstationaryProcesses
using ParameterInference
using Catch22
using Distributions
using StatsPlots
pyplot()
sim = simplestChaoticFlowSim(
    X0 = [0.05, 0.05, 0.05], # As in Sprott's paper
    parameter_profile = constant,
    parameter_profile_parameters = (2.032,),
    transient_t0 = -100.0,
    t0 = 0.0,
    dt = 0.0001,
    savedt = 0.5,
    tmax = 5000.0/20,
    alg = AutoVern7(Rodas5()),
    solver_opts = Dict(:adaptive => true, :reltol => 1e-10))

# Generate a high-dimensional baseline from noisyShiftyScalySine
N = 1000
𝐃 = [Uniform(0.1, 3.0), Uniform(0.1, 3.0), Uniform(0.1, 3.0)];
𝐩 = [rand.(𝐃) for i in 1:N];
𝒳0 = [randn() for i in 1:N]; # In case the IC changes a feature
𝒮₁ = [simulate(noisyShiftyScalySineSim(ps = 𝐩[i], X0 = [𝒳0[i]], tmax=1000.0/20, profiles = (constant, constant, constant))) for i ∈ 1:N];
𝒳₁ = timeseries.(𝒮₁, 1);
F1 = (catch23∘Array).(𝒳₁);
ℱ₁  = Catch22.featureMatrix(hcat(F1...), Catch22.featureDims(F1[1]), 𝐩);


# Try adding another system (chaotic)
N = 1000
𝐩 = rand(Uniform(2.018, 2.082), N);
𝒮₂ = [simulate(simplestChaoticFlowSim(profile=constantParameter, ps=[𝐩[i]], tmax=1000.0./20, Δt=0.1)) for i ∈ 1:N];
𝒳₂ = timeseries.(𝒮₂, 1);
F2 = (catch23∘Array).(𝒳₂);
ℱ₂ = Catch22.featureMatrix(hcat(F2...), Catch22.featureDims(F2[1]), 𝐩);


# And another (AR)
N = 1000
𝐃 = [Uniform(0.0, 0.25), Uniform(0.0, 0.25), Uniform(0.0, 0.25), Uniform(0.0, 0.25)];
# If all coefficients are less than 0.25 the process cannot be unstable
𝐩 = [rand.(𝐃) for i in 1:N];
#𝒳0 = [randn() for i in 1:N]; # In case the IC changes a feature
𝒮₃ = [simulate(arSim(profile=ntuple(x->constantParameter, 4), ps = 𝐩[i], T = 10000/20)) for i ∈ 1:N];
𝒳₃ = timeseries.(𝒮₃, 1);
F3 = (catch23∘Array).(𝒳₃);
ℱ₃  = Catch22.featureMatrix(hcat(F3...), Catch22.featureDims(F3[1]), 𝐩);


# Compare the baselines, and then merge
featureviolin(ℱ₁, ℱ₂, normalise=:feature)
featureviolin(ℱ₁, ℱ₃, normalise=:feature)
ℱ  = Catch22.featureMatrix(hcat(ℱ₁, ℱ₂, ℱ₃), Catch22.featureDims(F1[1]))


# Generate a constant  (low-dimensional) baseline
𝒳0 = [0.05.+0.0001.*randn(3) for i in 1:N];
𝒮′ = [simulate(sim(X0 = 𝒳0[i])) for i ∈ 1:N];
𝒳′ = timeseries.(𝒮′);
F′ = (catch23∘Array).(𝒳′);
ℱ′  = Catch22.featureMatrix(hcat(F′...), Catch22.featureDims(F′[1]))

# First, compare the baselines
featureviolin(ℱ, ℱ′, normalise=:feature)

# Apply the constant baseline
ℱ̂ = reZero(ℱ′)(ℱ)
#featureviolin(ℱ̂, ℱ′, normalise=:feature)

# Produce a total baseline function
𝑏 = reStandardise(ℱ̂)

# Run an inference, without incorporating the baseline
S = simulate(simplestChaoticFlowSim(tmax=5000, Δt=0.05))
I = infer(S, 1, features=catch23)
parameterestimate(I, normalisef=false)

# Run an inference, incorporating the baseline
I = infer(S, 1, baseline=𝑏, normalisation=_self, features=catch23)
parameterestimate(I, normalisef=false)

# And with only the high dim baseline
I = infer(S, 1, baseline=reStandardise(ℱ), normalisation=_self, features=catch23)
parameterestimate(I, normalisef=false)

# And only the low
I = infer(S, 1, baseline=reStandardise(ℱ′), normalisation=_self, features=catch23)
parameterestimate(I, normalisef=false)