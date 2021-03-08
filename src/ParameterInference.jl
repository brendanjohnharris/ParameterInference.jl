module ParameterInference

include("Windows.jl")
include("FeatureRepresentations.jl")
include("Normalisation.jl")
include("Filter.jl")
include("LowDimensionalProjections.jl")


# windows:      x -> Array of windows
# features:     TS array -> feature array
# projections:  feature array -> low dim projection array
# estimates:    projection array --> parameter estimates
function infer(x::Vector; windows::Function, features::Function, normalisation::Function, projections::Function, estimates::Function)

end



end
