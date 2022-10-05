using Catch22
using StatsBase
using Colors
using LinearAlgebra
using ColorVectorSpace
using Clustering
using Tullio

# function horizontal_legend(N, ax=PyPlot.gca())
#     PyPlot.svg(true)
#     l = PyPlot.gca().legend_
#     # https://stackoverflow.com/questions/23689728/how-to-modify-matplotlib-legend-after-it-has-been-created
#     defaults = Dict(:loc => l._loc,
#         :numpoints => l.numpoints,
#         :markerscale => l.markerscale,
#         :scatterpoints => l.scatterpoints,
#         :scatteryoffsets => l._scatteryoffsets,
#         :prop => l.prop,
#         # fontsize = None,
#         :borderpad => l.borderpad,
#         :labelspacing => l.labelspacing,
#         :handlelength => l.handlelength,
#         :handleheight => l.handleheight,
#         :handletextpad => l.handletextpad,
#         :borderaxespad => l.borderaxespad,
#         :columnspacing => l.columnspacing,
#         :ncol => N,
#         :mode => l._mode,
#         :shadow => l.shadow,
#         :title => l._legend_title_box.get_visible() ? l.get_title().get_text() : nothing,
#         :framealpha => l.get_frame().get_alpha(),
#         :bbox_to_anchor => l.get_bbox_to_anchor()._bbox,
#         :bbox_transform => l.get_bbox_to_anchor()._transform)
#     PyPlot.legend(("1", "2", "3"), defaults)
#     f = PyPlot.gcf()
# end


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
        if F isa AbstractFeatureArray
            y = string.(featureDims(F))
        else
            y = string.(1:size(F, 1))
        end
    elseif length(FI.args) == 1
        F = FI.args[1]
        if F isa AbstractFeatureArray
            y = string.(featureDims(F))
        else
            y = string.(1:size(F, 1))
        end
        x = 1:size(F, 2)#string.(dims(F, :timeseries).val)
    end
    @series begin
        seriestype := :heatmap
        framestyle --> :box
        yflip --> true
        xaxis --> nothing
        yticks --> :all
        clim = max(abs.(extrema(F))...)
        seriescolor --> palette(:Blues_9, 7)
        if min(F...) > 0.0
            clims := (0.0, clim)
        elseif max(F...) < 0.0
            clims := (-clim, 0.0)
        else
            clims := (-clim, clim)
            seriescolor --> palette(:RdYlBu_11, 7)
        end
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
        if F isa AbstractFeatureArray
            f = string.(featureDims(F))
        else
            f = string.(1:size(F, 1))
        end
    end

    Df = 1.0.-abs.(metric(F'))
    idxs = Clustering.hclust(Df; linkage=:average, branchorder=:optimal).order
    Df′ = Df[idxs, idxs]

    #plot!(yticks=(1:size(Df, 1), replace.(string.(Catch22.featurenames[idxs]), '_'=>"\\_")), size=(800, 400), xlims=[0.5, size(Df, 1)+0.5], ylims=[0.5, size(Df, 1)+0.5], box=:on, )
    ann = [(x-0.5, y-0.5, text("$(round(Df′[x, y], digits=2))", :white)) for x ∈ 1:length(idxs) for y ∈ 1:length(idxs)]

    @series begin
        seriestype := :heatmap
        framestyle --> :box
        xaxis --> nothing
        yflip --> true
        lims --> (0, length(idxs))
        aspect_ratio --> :equal
        size --> (800, 400)
        if length(f) ≤ 60
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
