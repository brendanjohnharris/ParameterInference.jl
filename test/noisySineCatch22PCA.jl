using NonstationaryProcesses
using Catch22
using MultivariateStats
using Plots
gr()

# Will need to work through this script to add convenience functions, plot recipes and fix nasty types (catch22)

sim = noisySineSim()

# Simulate
x = timeseries(sim, 1)
p1 = plot(x, xlabel="t", label=nothing, title="Time series")

# Window
X = slidingWindow(x, 500, 0, rect)

# Feature transformation
F = Array{Float64}(catch22(X)) # Will need to find an alternative to NamedArrays later, and make catch22 output type homogeneous to Float64

# Preprocessing
F = zscore(F)
F = nonanrows(F)

# Projection
M = project(F, principalComponents)

# Estimate # params
σ² = residualVariance(M, F)
ξ² = explainedVariance(M)

p2 = plot(plot(1:length(σ²), σ², seriestype=:line, seriescolor=:black, markersize=5, marker=:circle, label=nothing, ylabel="Residual Variance", xlabel="# PCs"), plot(1:length(ξ²), ξ², seriestype=:line, seriescolor=:red, markersize=5, marker=:circle, label=nothing, ylabel="Prop. Explained Variance", xlabel="# PCs"))


# Plot parameter estimate/s
nPs = 1
ps = embed(M, F, 1:nPs)
p3 = plot(unitInterval(-ps[:]), xlabel="Window/time", ylabel="Parameter (a.u.)", label="Estimate", legend=:bottomright)
p = NonstationaryProcesses.tuplef2ftuple(sim.parameter_profile, sim.parameter_profile_parameters) # Add parameters() func to NonstationaryProcesses)
t = sim.t0:sim.savedt:sim.tmax
dasht = length(ps).*((sim.t0:sim.savedt:sim.tmax).-sim.t0)./sim.tmax
plot!(dasht, unitInterval(p(t)), seriescolor=:red, label="True")

# p1, p2, p3

# Plot trajectory in first two PCs
ps2 = embed(M, F, 1:2)
#plot(ps2[1, :], ps2[:, 2], seriestype=:scatter, marker_z=unitInterval(p(t)))