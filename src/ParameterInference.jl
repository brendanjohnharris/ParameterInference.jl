module ParameterInference
using NonstationaryProcesses

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
    F̂ = normalisation(F)
    F′
    model
    estimates
end
export Inference


include("Features.jl")
include("Windows.jl")
include("FeatureRepresentations.jl")
include("Normalisation.jl")
include("LowDimensionalProjections.jl")
include("FeatureClustering.jl")


function infer(x::AbstractVector; windows::Function=slidingWindow, features::Function=catch24, normalisation::Function=nonanrows∘standardise∘noconstantrows, dimensionalityReduction::Function=principalComponents, parameters=x.+NaN)

    X, windowIdxs = windows(x)
    F = features(X)
    F̂ = normalisation(F)
    M = project(Array(F̂), dimensionalityReduction)
    F′ = embed(M, Array(F̂))
    estimates = embed(M, Array(F̂), [1]) # One parameter for now, think about more

    Inference(timeseries=x, windows=windows, windowEdges=windowIdxs, features=features, normalisation=normalisation, dimensionalityReduction=dimensionalityReduction, model=M, F=F, F′=F′, estimates=estimates, parameters=parameters)
end

function infer(P::NonstationaryProcesses.Process, dim::Int=1; parameters::Int=1, kwargs...)
    infer(timeseries(P, dim), parameters=NonstationaryProcesses.parameters(P, p=parameters); kwargs...)
end
export infer



include("Plotting.jl")
end
