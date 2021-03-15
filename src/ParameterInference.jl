module ParameterInference
using NonstationaryProcesses
include("Windows.jl")
include("FeatureRepresentations.jl")
include("Normalisation.jl")
include("LowDimensionalProjections.jl")
include("FeatureClustering.jl")


# windows:      x -> Array of windows
# features:     TS array -> feature array
# projections:  feature array -> low dim projection array
# estimates:    projection array --> parameter estimates

Base.@kwdef struct Inference
    timeseries
    parameters=timeseries .+ NaN
    windows
    windowEdges
    features
    normalisation
    dimensionalityReduction
    F
    model
    F′
    estimates
end
export Inference

function infer(x::AbstractVector; windows::Function=slidingWindow, features::Function=catch22, normalisation::Function=nonanrows∘zscore, dimensionalityReduction::Function=principalComponents, parameters=x.+NaN)

    X, windowIdxs = windows(x)
    F = features(X)
    F̂ = normalisation(F)
    M = project(Array(F̂), dimensionalityReduction)
    F′ = embed(M, Array(F̂))
    estimates = embed(M, Array(F̂), [1]) # One parameter for now, think about more

    Inference(timeseries=x, windows=windows, windowEdges=windowIdxs, features=features, normalisation=normalisation, dimensionalityReduction=dimensionalityReduction, model=M, F=F, F′=F′, estimates=estimates, parameters=parameters)
end

function infer(P::NonstationaryProcesses.Process, dim::Int=1; kwargs...)
    infer(timeseries(P, dim), parameters=parameters(P); kwargs...)
end
export infer



include("Plotting.jl")
end
