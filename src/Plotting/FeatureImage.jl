using Catch22
using StatsBase
using Colors
using LinearAlgebra
using ColorVectorSpace
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
# Plot the covariance matrix in a fancy way, from either the feature matrix or a precomputed covariance matrix. Can supply featurenames as first argument, if wanted, and either a feature matrix (not square and symmetric) or a covariance matrix (square and symmetric) as arg. 2

@userplot CovarianceMatrix
@recipe function f(g::CovarianceMatrix;  metric=StatsBase.cor, palette=[:cornflowerblue, :crimson, :forestgreen])
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
    r = round.(σ⁻¹*Σ²*σ⁻¹, sigdigits=10) # Don't want asymmetry because of floating point error
    Dr = 1.0.-abs.(r)
    if issymmetric(Dr)
        idxs = Clustering.hclust(Dr; linkage=:average, branchorder=:optimal).order
    else
        @warn "Correlation distance matrix is not symmetric, so not clustering"
        idxs = 1:size(Dr, 1)
    end
    Σ̂² = Σ²[idxs, idxs] # r[idxs, idxs]#
    f̂ = f[idxs]
    P = abs.(eigvecs(Array(Σ̂²)))[:, 1:length(palette)]
    P̂ = P./sum(P, dims=2)#unitInterval(P)
    rgb = parse.(RGBA, palette);
    colours = [i - RGBA(0.0, 0.0, 0.0, i.alpha) for i ∈ rgb]
    C = 0.5.*colours'*P̂ .+ 0.5.*P̂*colours
    A = RGBA.((0.0,), (0.0,), (0.0,), Σ̂²./max(Σ̂²...))
    #C = fill(RGBA(0.0, 0.0, 0.0, 0.0), size(A))
    Z = C + A
    @series begin
        seriestype := :heatmap
        framestyle := :box
        xticks := :none
        colorbar := :none
        lims := (1.0, size(Z, 1))
        aspect_ratio := :equal
        label := :none
        #categorical := true
        #seriescolor := cgrad(Z[:])
        legend := :none
        yticks := (LinRange(1.0, size(Z, 1), size(Z, 1)+1)[1:end-1].+((size(Z, 1))-1)/size(Z, 1)/2, f̂)
        grid := :none
        (Z,)
    end
end

# @recipe function f(::Type{Val{:covariancematrix}}, plt::AbstractPlot;)
#     f, Σ² = plotattributes[:x], Array(plotattributes[:z])

# end