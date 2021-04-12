using ParameterInference
using NonstationaryProcesses

S = simulate(simplestChaoticFlowSim)

# Generate a constant (low-dimensional) baseline
# Jitter IC so time series are not identical
N = 100
𝒮′ = [simulate(S(X0 = S.X0 .+ 0.0005.*randn(length(S.X0)), tmax=S.tmax/20, profile=constantParameter, ps=[2.032], transient=-1000.0)) for i ∈ 1:N];
𝒳′ = timeseries.(𝒮′, 1);
F′ = (catch24∘Array).(𝒳′);
ℱ′  = Catch22.featureMatrix(hcat(F′...), Catch22.featureDims(F′[1]))

𝑏 = reStandardise(ℱ′)

# No baseline
I = infer(S)
parameterestimate(I, normalisef=false)

# Baseline
I = infer(S, baseline=𝑏, normalisation=_self)
parameterestimate(I, normalisef=false)
