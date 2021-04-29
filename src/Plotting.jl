using Plots
using DimensionalData
using StatsBase
import StatsBase.std


# ------------------------------------------------------------------------------------------------ #
#                                       Plot a feature matrix                                      #
# ------------------------------------------------------------------------------------------------ #
# @recipe function f(::Type{Val{:featurematrix}}, x, y, z)
#     # seriescolor := cgrad(:RdYlBu_11, 7, categorical = true)
#     #yticks := nothing
#     #colorbar --> nothing
#     # xticks := (x, string.(x))
#     # yticks := (y, string.(y))
#     seriestype := :heatmap
#     x := x
#     y := y
#     z := z
#     #(x, y, z)
#     ()
# end

# @recipe function f(fm::Type{Val{:featurematrix}}, F::DimArray, args...)
#     seriestype := featurematrix
#     features := String.(dims(F, :feature).val)
#     timeseries := String.(dims(F, :timeseries).val)
#     F --> Array(F)
#     ()
# end

include("Plotting/ParameterEstimate.jl")
include("Plotting/DimensionalityEstimate.jl")
include("Plotting/TimeSeriesPlots.jl")
include("Plotting/FeatureDistributions.jl")
include("Plotting/LowDimensionalPlots.jl")
include("Plotting/StatisticalFilter.jl")
include("Plotting/ToyVarianceMaps.jl")
