using Catch22
using StatsBase
using Colors
using LinearAlgebra
using ColorVectorSpace
using Clustering
using TensorCast

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
        if typeof(F) <: AbstractFeatureArray
            y = string.(Catch22.featureDims(F))
        else
            y = string.(1:size(F, 1))
        end
    elseif length(FI.args) == 1
        F = FI.args[1]
        if typeof(F) <: AbstractFeatureArray
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
        seriescolor --> cgrad(:RdYlBu_11, 7, categorical = true)
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
        if typeof(F) <: AbstractFeatureArray
            f = string.(Catch22.featureDims(F))
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


# ------------------------------------------------------------------------------------------------ #
#                                         Covariance Matrix                                        #
# ------------------------------------------------------------------------------------------------ #
# Plot the covariance matrix in a fancy way, from either the feature matrix or a precomputed covariance matrix. Can supply featurenames as first argument, if wanted, and either a feature matrix (not square and symmetric) or a covariance matrix (square and symmetric) as arg. 2.
# Use pyplot()
# colormode can be :top, :all, or :raw

@userplot CovarianceMatrix
@recipe function f(g::CovarianceMatrix;  metric=StatsBase.cor, palette=[:cornflowerblue, :crimson, :forestgreen], colormode=:top)
    @assert 1 ≤ length(g.args) ≤ 2 && typeof(g.args[end]) <: AbstractMatrix

    if typeof(g.args[1]) <: AbstractFeatureArray
        f = Catch22.featureDims(g.args[1])
        Σ² = Array(g.args[1])
    else
        if length(g.args) == 2
            f = g.args[1]
            Σ² = g.args[2]
        else
            f = string.(1:size(g.args[1], 1))
            Σ² = g.args[1]
        end
    end
    if !issymmetric(Σ²)
        Σ² = StatsBase.cov(Σ²')
    end
    σ⁻¹ = sqrt(Diagonal(Σ²))^-1
    #r = round.(σ⁻¹*Σ²*σ⁻¹, sigdigits=10) # Don't want asymmetry because of floating point error
    #Dr = 1.0.-abs.(r)
    Dr = 1.0.-abs.(Σ²)
    if issymmetric(Dr)
        idxs = Clustering.hclust(Dr; linkage=:average, branchorder=:optimal).order
    else
        @warn "Correlation distance matrix is not symmetric, so not clustering"
        idxs = 1:size(Dr, 1)
    end

    Σ̂² = Σ²[idxs, idxs] # r[idxs, idxs]#
    A = abs.(Σ̂²)./max(abs.(Σ̂²)...)
    f̂ = f[idxs]
    if colormode != :raw
        if colormode == :top
            P = abs.(eigvecs(Symmetric(Array(Σ̂²))))[:, end:-1:end-length(palette)+1]
            P̂ = P./sum(P, dims=2)#unitInterval(P)
            𝑓′ = parse.(XYZ, palette);
        elseif colormode == :all
            P = abs.(eigvecs(Symmetric(Array(Σ̂²))))[:, end:-1:1]
            Σ̂′² = Diagonal(abs.(eigvals(Symmetric(Array(Σ̂²))))[end:-1:1])
            P̂ = P./sum(P, dims=2)#unitInterval(P)
            p = fill(:black, size(P, 2))
            p[1:length(palette)] = palette
            𝑓′ = parse.(XYZ, p);
            [𝑓′[i] = Σ̂′²[i, i]*𝑓′[i] for i ∈ 1:length(𝑓′)]
        end

        𝑓 = P̂*𝑓′
        H = Array{XYZA}(undef, size(Σ̂²))
        for (i, j) ∈ Tuple.(CartesianIndices(H))
            J = (𝑓[i] + 𝑓[j])/2
            H[i, j] = XYZA(J.x, J.y, J.z, A[i, j])
        end
        H = convert.((RGBA,), H)
    else
        H = abs.(Σ̂²)
        colorbar --> true
    end
    @series begin
        seriestype := :heatmap
        (H,)
    end
    # Plot the dummy data and set attributes
    @series begin
        colorbar --> true
        seriestype := :scatter
        markersize := 0.0
        label := nothing
        legend := :none
        marker_z := [0, max(abs.(Σ²)...)]
        if colormode != :raw
            markercolor := :binary
        end
        (zeros(2), zeros(2))
    end
    if colormode != :raw
        for i ∈ 1:length(palette)
            @series begin
                seriestype := :shape
                label := "PC$i"
                legend := :bottomright
                colorbar_title := "Σ²"
                colorbar_titlefontsize := 14
                line_width := 20
                xticks := :none
                size --> (800, 400)
                lims := (0.5, size(H, 1)+0.5)
                aspect_ratio := :equal
                legendfontsize := 8
                yticks := (1:size(H, 1), pysafelabel.(String.(f̂)))
                grid := :none
                framestyle := :box
                seriescolor := palette[i]
                (Shape([0.0;], [0.0;]))
            end
        end
    end
end
