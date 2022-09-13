using Catch22
using StatsBase

Catch22.getnames(𝐟::Vector{<:Catch22.Features.AbstractFeature}) = Catch22.getname.(𝐟)

catch2 = FeatureSet([StatsBase.mean, StatsBase.std], [:mean, :standard_deviation], [["distribution","centre"], ["distribution","spread"]], ["Arithmetic mean", "Sample standard deviation"])
export catch2

catch24Names = vcat(:mean, :standard_deviation, Catch22.featurenames...)
catch24 = catch2+catch22
export catch24

catch23 = catch24\FeatureSet(FC_LocalSimple_mean1_tauresrat)
catch23Names = getnames(catch23)
export catch23

catch21 = catch22\FeatureSet(FC_LocalSimple_mean1_tauresrat)
catch21Names = getnames(catch21)
export catch21

IntFeatures = FeatureSet([CO_f1ecac,
                        CO_FirstMin_ac, IN_AutoMutualInfoStats_40_gaussian_fmmi,  SB_BinaryStats_diff_longstretch0,  SB_BinaryStats_mean_longstretch1,
                        PD_PeriodicityWang_th0_01,
                        FC_LocalSimple_mean1_tauresrat,
                        CO_Embed2_Dist_tau_d_expfit_meandiff,
                        SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1])
catch24sansInt = catch24\IntFeatures
catch24sansIntnames = getnames(catch24sansInt)
catch22sansInt = catch22\IntFeatures
catch22sansIntnames = getnames(catch22sansInt)
export catch24sansInt, catch22sansInt


function featureRepresentation(X, featureFunc::Function)
    F = featureFunc(X)
end
export featureRepresentation


function NonstationaryProcesses.forcemat(x::AbstractFeatureArray)
    if x isa AbstractFeatureVector
        x = Catch22.featureMatrix(forcemat(Array(x)), Catch22.featureDims(x))
    end
    return x
end

function intersectFeatures(X::AbstractFeatureArray, Y::AbstractFeatureArray)
    # Will be stable in the order of features in X
    fx, fy = Catch22.featureDims(X), Catch22.featureDims(Y)
    if any(.!in.(fx, (fy,)))
        error("The features of Y are neither the same as nor a superset of the features in X")
    end
    fs = intersect(Catch22.featureDims(X), Catch22.featureDims(Y))
    return (X[fs, :], forcemat(Y)[fs, :])
end
function intersectFeatures(X::AbstractFeatureArray, Y::Array)
    # Label Y with the same features as X
    if size(X, 1) != size(Y, 1)
        @error "X and Y do not have the same number of rows"
    end
    fs = Catch22.featureDims(X)
    Y = FeatureMatrix(Y, fs)
    return (X, Y)
end
function intersectFeatures(X, Y, Z, otherarrays...)
    otherarrays = [Y, Z, otherarrays...]
    for i ∈ 1:lastindex(otherarrays)
        X, otherarrays[i] = intersectFeatures(X, otherarrays[i])
    end
    # Then do it again in case we missed any
    for i ∈ 1:lastindex(otherarrays)
        X, otherarrays[i] = intersectFeatures(X, otherarrays[i])
    end
    return (X, otherarrays...)
end
export intersectFeatures
