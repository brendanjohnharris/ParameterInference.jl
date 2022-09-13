module ParameterInference
using Catch22
using NonstationaryProcesses

# windows:      x -> Array of windows
# features:     TS array -> feature array
# projections:  feature array -> low dim projection array
# estimates:    projection array --> parameter estimates

Base.@kwdef struct Inference
    # These you can set
    timeseries
    windows = slidingwindow
    features = catch24
    baseline = identity
    filter = identity #nonanrows∘noconstantrows∘noinfrows
    normalisation = identity#standardise
    dimensionalityReduction = principalcomponents
    parameters = timeseries .+ NaN
    # These you should leave to calculate
    windowedTimeseries = (windows)(timeseries)
    windowTimes = (windows)(1:length(timeseries))
    windowEdges = [windowTimes[1, :][:]..., windowTimes[end]]
    windowCentres = mean(windowTimes, dims=1)[:]
    F = (!(features isa AbstractFeatureSet) && !(features isa Vector{Function}) ? features : features(windowedTimeseries)) # Maybe you supplied some pre-computed features?
    F̂ = (filter∘baseline∘filter∘normalisation∘filter)(F) #! Too many filters?
    model = project(Array(F̂), dimensionalityReduction)
    F′ = embed(model, Array(F̂))
    estimates = embed(model, Array(F̂), [1])
    corrtype = corspearman
    ρ = corrtype(mean((windows)(parameters), dims=1)[:], estimates)
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
include("BaselineFilter.jl")
# include("BoxCox.jl")
include("TestBaselines.jl")

Base.identity(x, y) = identity

function infer(x::AbstractVector; kwargs...)
    Inference(timeseries=x; kwargs...)
end

function infer(P::NonstationaryProcesses.Process, dim::Int=1; parameters::Int=1, kwargs...)
    infer(timeseries(P, dim), parameters=NonstationaryProcesses.parameterseries(P, p=parameters); kwargs...)
end
export infer

end
