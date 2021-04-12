using NonstationaryProcesses
using ParameterInference
using Printf
using Plots
using DimensionalData
pyplot() # Is most consistent
# In VSCode you should step through this script with ctrl+enter

# ------------- A harmonic oscillator with a potential that becomes steeper over time ------------ #
# This means that the frequency increases, but the amplitude decreases

# What does this look like for a constant parameter, say 1 Hz (2π angular frequency)
S = simulate(harmonicSim(parameter_profile=constantParameter, parameter_profile_parameters=(2π,)));
plot(S, vars=1, xlims=(0, 10))
# Just a sine wave...

# What sort of parameter estimate can we get from doubling frequency?
S = simulate(harmonicSim(parameter_profile=unitStep, parameter_profile_parameters=(50.0, 2π, 4π)));
I = infer(S);
parameterestimate(I)
# Easy... Note that the amplitude doesn't change since the parameter change happens at the extrema


# What if we have a ramping parameter value? Surely it can't be much worse:
S = simulate(harmonicSim(parameter_profile=ramp, parameter_profile_parameters=(2π*0.02, 2π, 0.0))); # Ramp from 2π to 4π over 100s
I = infer(S);
parameterestimate(I)
# Not too shabby either

# Nonlinear feature responses typicaly occur with large changes in parameters
# Or near asymptotic, bifurcation or endpoint parameter values
# For example, starting from a low frequency (0)
S = simulate(harmonicSim(parameter_profile=ramp, parameter_profile_parameters=(2π*0.02, π/2, 0.0))); # Ramp from  to π/2 to π/2 + 2π
I = infer(S);
parameterestimate(I)


# What feature best represents the first principal component?
PC1 = PCfeatureWeights(I, 1);
f = PC1[argmax(abs.(PC1)), :];
println("The heaviest feature in the first principle component is $(val(refdims(f))[1]) (w=$(@sprintf("%.2f", f[1])))")
# plot(times(S)[Int.((I.windowEdges[1:end-1].+I.windowEdges[2:end])./2)], I.F[val(refdims(f))[1], :], seriescolor=:black, size=(600, 400), right_margin=20Plots.mm, left_margin=10Plots.mm, box=true)
# plot!(xguide="Time", yguide="$(val(refdims(f))[1])", legend=nothing)
# plot!(twinx(), times(S), parameterseries(S), seriescolor=:red, yguide="ω", legend=nothing, guidecolor=:red)
plot(parameterseries(S)[Int.((I.windowEdges[1:end-1].+I.windowEdges[2:end])./2)],  I.F[val(refdims(f))[1], :], xguide="ω ∼ t", yguide="$(val(refdims(f))[1])", seriescolor=:black, legend=nothing, box=true)


# Can we improve the estimate using nonlinear dimensionality reduction
# We will use a longer time series, since something like Isomap (especially with the default N.N. of 12) works best with at least > 20 points
# These steps may take a minute
S = simulate(harmonicSim(parameter_profile=ramp, parameter_profile_parameters=(2π*0.02, π/2, 0.0), tmax=1000.0));
I_pca = infer(S, windows=x -> slidingWindow(x, length(x)÷(200)))
I_isomap = infer(S, dimensionalityReduction=isomap, windows=x -> slidingWindow(x, length(x)÷(200)))
parameterestimate(I_pca)
parameterestimate(I_isomap)