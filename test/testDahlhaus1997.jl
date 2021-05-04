using ParameterInference
using NonstationaryProcesses
using Catch22

T = 10000
ϕ₁(t) = 1.8*cos(1.5-cos(4π*t./T))
ϕ₂(t) = -0.81

S = arSim(X0 = zeros(3),
parameter_profile = (ϕ₁, ϕ₂),
parameter_profile_parameters = ([], []),
t0 = 0,
dt = 1,
savedt = 1,
tmax = T)

# Constant Baseline
ϕ₁′(t) = 0.0
𝒮′ = [simulate(S(profiles=(ϕ₁′, ϕ₂), tmax=T/20)) for i ∈ 1:100];
𝒳′ = timeseries.(𝒮′);
F′ = (catch24∘Array).(𝒳′);
F′  = Catch22.featureMatrix(hcat(F′...), Catch22.featureDims(F′[1]))

#I = infer(S(), baseline=noconstantrows∘reStandardise(F′), normalisation=_self) #() resets the RNG seed
I = infer(S(), baseline=_self, normalisation=nonanrows∘noconstantrows∘standardise)
parameterestimate(I, normalisef=false)
