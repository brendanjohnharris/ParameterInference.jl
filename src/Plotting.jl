using Plots
using DimensionalData
using StatsBase
import StatsBase.std

function pysafelabel(s)
    if typeof(s) <: String
        replace(s, r"_" => s"\\_")
    else
        return s
    end
end
export pysafelabel

include("Plotting/ParameterEstimate.jl")
include("Plotting/DimensionalityEstimate.jl")
include("Plotting/TimeSeriesPlots.jl")
include("Plotting/FeatureDistributions.jl")
include("Plotting/LowDimensionalPlots.jl")
include("Plotting/StatisticalFilter.jl")
include("Plotting/ToyVarianceMaps.jl")
include("Plotting/FeatureImage.jl")

