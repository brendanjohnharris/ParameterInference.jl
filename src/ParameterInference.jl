module ParameterInference
using Catch22: timeseriesDims
using NonstationaryProcesses

# windows:      x -> Array of windows
# features:     TS array -> feature array
# projections:  feature array -> low dim projection array
# estimates:    projection array --> parameter estimates
_self(arg) = arg
export _self

Base.@kwdef struct Inference
    # These you can set
    timeseries
    windows = slidingWindow
    features = catch24
    baseline = _self
    filter = nonanrows∘noconstantrows∘noinfrows
    normalisation = _self#standardise
    dimensionalityReduction = principalcomponents
    parameters = timeseries .+ NaN
    # These you should leave to calculate
    windowedTimeseries = (windows)(timeseries)
    windowEdges = windowedTimeseries[2]
    windowCentres = (windowEdges[1:end-1] + windowEdges[2:end])./2
    F = ((features isa AbstractArray) ? features : features(windowedTimeseries[1])) # Maybe you supplied some pre-computed features?
    F̂ = (filter∘baseline∘filter∘normalisation∘filter)(F) #! Too many filters?
    model = project(Array(F̂), dimensionalityReduction)
    F′ = embed(model, Array(F̂))
    estimates = embed(model, Array(F̂), [1])
    corrtype = corspearman
    ρ = corrtype(parameters[Int.(round.(windowCentres))], estimates)
end
export Inference

include("Baseline.jl")
include("FeatureClustering.jl")
include("FeatureRepresentations.jl")
include("LowDimensionalProjections.jl")
include("Normalisation.jl")
include("Plotting.jl")
include("Windows.jl")
include("DistributionTests.jl")
include("Information.jl")

function infer(x::AbstractVector; kwargs...)
    Inference(timeseries=x; kwargs...)
end

function infer(P::NonstationaryProcesses.Process, dim::Int=1; parameters::Int=1, kwargs...)
    infer(timeseries(P, dim), parameters=NonstationaryProcesses.parameterseries(P, p=parameters); kwargs...)
end
export infer

end
