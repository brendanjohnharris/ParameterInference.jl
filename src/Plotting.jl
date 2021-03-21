using Plots
using DimensionalData
import StatsBase


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




# ------------------------------------------------------------------------------------------------ #
#                        Plot inference (consider changing to series recipe)                       #
# ------------------------------------------------------------------------------------------------ #
# This one takes a time series and moves all the way through to esimation, plotting relevant stuff
# Plots best in pyplot()
@userplot ParameterEstimate
@recipe function f(P::ParameterEstimate)
    I = P.args[1]
    legend --> false
    #link := :x
    grid --> false
    layout --> @layout [
        x
        F
        P
    ]
    size --> (800, 500)
    left_margin --> 5Plots.mm
    right_margin --> 5Plots.mm

    windowCentres = (I.windowEdges[1:end-1] + I.windowEdges[2:end])./2

# ------------------------------------------ Time series ----------------------------------------- #

    @series begin
        seriestype := :line
        subplot := 1
        seriescolor := :black
        x = I.timeseries
    end

# ------------------------------------------- Features ------------------------------------------- #

    @series begin
        seriestype := :heatmap
        subplot := 2
        seriescolor := cgrad(:RdYlBu_11, 7, categorical = true)
        yticks := nothing
        F = I.normalisation(I.F)

        clusterF = clusterReorder(F, CorrDist(), linkageMetric=:average, branchOrder=:optimal, dim=1)

        # D = 1.0.-abs.(StatsBase.corspearman(F'))
        # clusterF = clusterReorder(Array(F), D, linkageMetric=:average, branchOrder=:optimal, dim=1)

        # ρ = abs.(mapslices(x -> corspearman(x, I.parameters[Int.(windowCentres)]), F, dims=2))
        # idxs = sortperm(ρ[:])
        # clusterF = F[idxs, :]

        colorbar --> nothing
        (x, y, X) = (dims(I.timeseries, Ti).val[Int.(windowCentres)], 1:size(clusterF, 1), sigmoidNormalise(clusterF))
    end


# ------------------------------------------ Parameters ------------------------------------------ #

    @series begin
        seriestype := :line
        subplot := 3
        yaxis := nothing
        seriescolor := :black
        (x, y) = (dims(I.timeseries, Ti).val, I.parameters)
    end

    @series begin
        seriestype := :scatter
        seriescolor := :black
        inset_subplots := (3, bbox(0,0,1,1))
        yaxis := nothing
        xaxis := nothing
        background_color_inside := nothing
        background_color_subplot := nothing
        p = I.estimates
        if StatsBase.corspearman(p, I.parameters[Int.(windowCentres)]) < 0
            @warn "It looks like the estimated parameters are the negative of the true parameters. This is not unexpected, so flipping for visualisation."
            p = -p
        end
        xlims := (0, length(p)) # These go in centres of windows
        (x, y) = (0.5:1:length(p)-0.5, p)
    end


# ---------------------------- Add windows; how to remove boilerplate? --------------------------- #
# Edit most axes properties here

    @series begin
        seriescolor := nothing
        seriestype := :vline
        inset_subplots := (1, bbox(0,0,1,1))
        yaxis := nothing
        xaxis := nothing
        yguide --> "Time Series"
        framestyle := :box
        background_color_inside := nothing
        background_color_subplot := nothing
        xlims := extrema(dims(I.timeseries, Ti).val)
        x = dims(I.timeseries, Ti).val[I.windowEdges]
    end
    @series begin
        seriescolor := :black
        seriestype := :vline
        inset_subplots := (2, bbox(0,0,1,1))
        yaxis := nothing
        xaxis := nothing
        clims := (0.0, 1.0)
        yguide --> "Features"
        framestyle := :box
        background_color_inside := nothing
        background_color_subplot := nothing
        xlims := extrema(dims(I.timeseries, Ti).val)
        x = dims(I.timeseries, Ti).val[I.windowEdges]
    end

    @series begin
        seriescolor := nothing
        seriestype := :vline
        inset_subplots := (3, bbox(0,0,1,1))
        yaxis := nothing
        #foreground_color_axis := nothing
        xguide --> "Time"
        yguide --> "Parameters"
        framestyle := :box
        background_color_inside := nothing
        background_color_subplot := nothing
        xlims := extrema(dims(I.timeseries, Ti).val)
        x = dims(I.timeseries, Ti).val[I.windowEdges]
    end
end

@userplot DimensionalityEstimate
@recipe function f(D::DimensionalityEstimate)
    I = D.args[1]
    legend --> false
    #link := :x
    right_margin --> 10Plots.mm
    grid --> false
    size --> (1000, 400)
    framestyle := :box
    layout --> @layout [ev rv]

    σ² = residualVariance(I.model, I.F̂)
    ξ² = explainedVariance(I.model)
    @series begin
        seriestype := :line
        markersize --> 5
        subplot := 1
        marker --> :circle
        label --> nothing
        seriescolor --> :black
        xguide --> "Principal Components"
        yguide --> "Residual Variance"
        (x, y) = (0:length(σ²), [1, σ²...])
    end

    @series begin
        seriestype := :line
        markersize --> 5
        subplot := 2
        marker --> :circle
        label --> nothing
        seriescolor --> :red
        xguide --> "Principal Components"
        yguide --> "Explained Variance"
        (x, y) = (0:length(ξ²), [0, ξ²...])
    end

end