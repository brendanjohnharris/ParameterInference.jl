using Catch22
using DimensionalData
using StatsBase
using Clustering
# ------------------------------------------------------------------------------------------------ #
#                                       Plot a feature matrix                                      #
# ------------------------------------------------------------------------------------------------ #
@userplot FeatureImage
@recipe function f(FI::FeatureImage)
    if length(FI.args) == 3
        x = FI.args[1]
        y = FI.args[2]
        F = FI.args[3]
    elseif length(FI.args) == 2
        F = FI.args[2]
        x = FI.args[1]
        if typeof(F) <: DimArray
            y = string.(Catch22.featureDims(F))
        else
            y = string.(1:size(F, 1))
        end
    elseif length(FI.args) == 1
        F = FI.args[1]
        if typeof(F) <: DimArray
            y = string.(Catch22.featureDims(F))
        else
            y = string.(1:size(F, 1))
        end
        x = 1:size(F, 2)#string.(dims(F, :timeseries).val)
    end
    @series begin
        seriestype := :heatmap
        framestyle --> :box
        xaxis --> nothing
        yticks --> :all
        seriescolor := cgrad(:RdYlBu_11, 7, categorical = true)
        clim = max(abs.(extrema(F))...)
        clims := (-clim, clim)
        (x, y, X) = (pysafelabel.(x), pysafelabel.(y), Array(F))
    end
end


@userplot DistanceImage
@recipe function f(DI::DistanceImage; metric=StatsBase.corspearman, annotatedist=false)
    if length(DI.args) == 2
        F = DI.args[2]
        f = DI.args[1]
    elseif length(DI.args) == 1
        F = DI.args[1]
        if typeof(F) <: DimArray
            f = string.(Catch22.featureDims(F))
        else
            f = string.(1:size(F, 1))
        end
    end

    Df = 1.0.-abs.(metric(F'))
    idxs = Clustering.hclust(Df; linkage=:average, branchorder=:optimal).order
    Df′ = Df[idxs, idxs]

    #plot!(yticks=(1:size(Df, 1), replace.(string.(Catch22.featureNames[idxs]), '_'=>"\\_")), size=(800, 400), xlims=[0.5, size(Df, 1)+0.5], ylims=[0.5, size(Df, 1)+0.5], box=:on, )
    ann = [(x-0.5, y-0.5, text("$(round(Df′[x, y], digits=2))", :white)) for x ∈ 1:length(idxs) for y ∈ 1:length(idxs)]

    @series begin
        seriestype := :heatmap
        framestyle --> :box
        xaxis --> nothing
        aspect_ratio --> :equal
        size --> (800, 400)
        if length(f) ≤ 50
            yticks --> :all
        end
        if annotatedist == true
            annotations --> ann
            colorbar --> nothing
        end
        seriescolor --> cgrad(:Greys_9, rev=true)
        colorbar_title --> "1-|ρ|"
        clims --> (0.0, 1.0)
        (x, y, X) = (pysafelabel.(f[idxs]), pysafelabel.(f[idxs]), Df′)
    end
end
