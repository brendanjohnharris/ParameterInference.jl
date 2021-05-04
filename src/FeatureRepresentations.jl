using Catch22
using StatsBase



catch24Names = vcat(:mean, :standard_deviation, Catch22.featureNames...)
function catch24(𝐱::AbstractVector{Float64})
    𝐟 = vcat(StatsBase.mean(𝐱), StatsBase.std(𝐱), catch22(𝐱)...)
    Catch22.featureVector(𝐟, catch24Names)
end
catch24(X::AbstractArray{Float64, 2}) = Catch22.featureMatrix(mapslices(catch24, X, dims=[1]), catch24Names)
export catch24

catch23Names = setdiff(catch24Names, [:FC_LocalSimple_mean1_tauresrat])
catch21Names = setdiff(Catch22.featureNames, [:FC_LocalSimple_mean1_tauresrat])
function catch23(𝐱::AbstractVector{Float64}) # Hot fix
    𝐟 = vcat(StatsBase.mean(𝐱), StatsBase.std(𝐱), catch22(𝐱, catch21Names)...)
    Catch22.featureVector(𝐟, catch23Names)
end
catch23(X::AbstractArray{Float64, 2}) = Catch22.featureMatrix(mapslices(catch23, X, dims=[1]), catch23Names)
export catch23

IntFeatures = [:CO_f1ecac,
:CO_FirstMin_ac, :IN_AutoMutualInfoStats_40_gaussian_fmmi, :SB_BinaryStats_diff_longstretch0, :SB_BinaryStats_mean_longstretch1,
:PD_PeriodicityWang_th0_01,
:FC_LocalSimple_mean1_tauresrat,
:CO_Embed2_Dist_tau_d_expfit_meandiff,
:SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1]
catch24sansIntnames = setdiff(catch24Names, IntFeatures)
catch22sansIntnames = setdiff(Catch22.featureNames, IntFeatures)
function catch24sansInt(𝐱::AbstractVector{Float64})
    𝐟 = vcat(StatsBase.mean(𝐱), StatsBase.std(𝐱), catch22(𝐱, catch22sansIntnames)...)
    Catch22.featureVector(𝐟, catch24sansIntnames)
end
catch24sansInt(X::AbstractArray{Float64, 2}) = Catch22.featureMatrix(mapslices(catch24sansInt, X, dims=[1]), catch24sansIntnames)
export catch24sansInt


function featureRepresentation(X, featureFunc::Function)
    F = featureFunc(X)
end
export featureRepresentation


function NonstationaryProcesses.forcemat(x::DimArray)
    if typeof(x) <: AbstractVector
        x = Catch22.featureMatrix(forcemat(Array(x)), Catch22.featureDims(x))
    end
    return x
end

function intersectFeatures(X::DimArray, Y::DimArray)
    # Will be stable in the order of features in X
    fx, fy = Catch22.featureDims(X), Catch22.featureDims(Y)
    if any(.!in.(fx, (fy,)))
        error("The features of Y are neither the same as nor a superset of the features in X")
    end
    fs = intersect(Catch22.featureDims(X), Catch22.featureDims(Y))
    return (X[fs, :], forcemat(Y)[fs, :])
end
function intersectFeatures(X::DimArray, Y::Array)
    # Label Y with the same features as X
    if size(X, 1) != size(Y, 1)
        @error "X and Y do not have the same number of rows"
    end
    fs = Catch22.featureDims(X)
    Y = DimensionalData.DimArray(Y, (Dim{:feature}(fs), Dim{:timeseries}(1:size(Y, 2))))
    return (X, Y)
end
export intersectFeatures